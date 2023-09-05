use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::rc::Rc;

use crate::compiler::ast::{BinOp, NodeRef, RefLen, UnOp};
use crate::compiler::registers::{Register, Registers};
use crate::prototype::{CIndex16, CIndex8, Instruction, Prototype, RIndex};
use crate::value::Value;
use crate::{PString, StringInterner};

type Ast = crate::compiler::sem_check::Ast2<PString>;
type Node = crate::compiler::ast::Node<PString>;
type Stat = crate::compiler::ast::Stat<PString>;
type Expr = crate::compiler::ast::Expr<PString>;
type Binding = Rc<crate::compiler::sem_check::Binding<PString>>;

#[derive(Debug)]
pub enum CodeGenError {
    UnexpectedNode,
    BadStateTransfer,
    MissingBinding,
    MissingLocalRegister,
    NoRegistersAvailable,
    ConstantPoolFull,
}

pub type Result<T> = std::result::Result<T, CodeGenError>;

pub fn code_gen<I: StringInterner<String = PString>>(
    ast: Ast,
    mut strings: I,
) -> Result<Prototype> {
    let name = strings.intern(b"test");
    let current_function = PrototypeBuilder {
        name,
        registers: Registers::new(),
        instructions: Vec::with_capacity(64),
        constants_map: HashMap::with_capacity(32),
        constants: Vec::with_capacity(32),
    };

    let mut code_gen = CodeGen {
        nodes: &ast.ast.nodes,
        refs: &ast.ast.refs,
        ref_cursor: 0,
        bindings: &ast.bindings,

        strings,

        state: Vec::with_capacity(32),

        current_function,
    };

    let root = NodeRef(0);
    code_gen.push_state(State::EnterStat(root));

    code_gen.visit()?;
    code_gen.finish()?;

    Ok(code_gen.current_function.build())
}

struct PrototypeBuilder {
    name: PString,
    registers: Registers,
    instructions: Vec<Instruction>,
    constants_map: HashMap<Value, CIndex>,
    constants: Vec<Value>,
}

impl PrototypeBuilder {
    fn build(self) -> Prototype {
        Prototype {
            name: self.name,
            stack_size: self.registers.stack_size().saturating_sub(1) as u8,
            instructions: self.instructions.into_boxed_slice(),
            constants: self.constants.into_boxed_slice(),
        }
    }
}

#[derive(Debug)]
enum State {
    // statements
    EnterStat(NodeRef),
    ExitStat,

    ContinueCompoundStat(RefLen),
    ExitCompoundStat(u16),
    ExitVarDecl,
    ExitExprStat,

    // expressions
    EnterExprAnywhere(NodeRef),
    EnterExpr(NodeRef, ExprDest),
    ExitExpr(RegisterOrConstant16),

    ExitVariableExpr(Register, ExprDest),
    ExitReturnExpr(ExprDest),
    ExitUnaryExpr(UnOp, ExprDest),
    ContinueBinaryExpr(BinOp, NodeRef, ExprDest),
    ExitBinaryExpr(BinOp, RegisterOrConstant16, ExprDest),
}

impl State {
    fn enter<I: StringInterner<String = PString>>(
        &mut self,
        from: Option<State>,
        code_gen: &mut CodeGen<I>,
    ) -> Result<()> {
        macro_rules! fail_transfer {
            () => {{
                dbg!(&from);
                Err(CodeGenError::BadStateTransfer)
            }};
        }

        match self {
            // statements
            State::EnterStat(statement) => match from {
                // only valid as long as this is the root
                None
                | Some(State::ExitStat)
                | Some(State::EnterStat(..))
                | Some(State::ContinueCompoundStat(..)) => code_gen.enter_statement(*statement),
                _ => fail_transfer!(),
            },
            State::ExitStat => match from {
                Some(State::EnterStat(..))
                | Some(State::ExitStat)
                | Some(State::ExitCompoundStat(..))
                | Some(State::ExitVarDecl)
                | Some(State::ExitExprStat) => Ok(()),
                _ => fail_transfer!(),
            },

            State::ContinueCompoundStat(len) => match from {
                Some(State::EnterStat(..)) | Some(State::ExitStat) => {
                    code_gen.continue_compound_statement(*len)
                }
                _ => fail_transfer!(),
            },
            State::ExitCompoundStat(stack_top) => match from {
                Some(State::EnterStat(..)) | Some(State::ExitStat) => {
                    code_gen.exit_compound_statement(*stack_top)
                }
                _ => fail_transfer!(),
            },
            State::ExitVarDecl => match from {
                Some(State::ExitExpr(register)) => code_gen.exit_variable_declaration(register),
                _ => fail_transfer!(),
            },
            State::ExitExprStat => match from {
                Some(State::ExitExpr(register)) => code_gen.exit_expression_statement(register),
                _ => fail_transfer!(),
            },

            // expressions
            State::EnterExprAnywhere(expression) => match from {
                Some(State::EnterStat(..))
                | Some(State::EnterExprAnywhere(..))
                | Some(State::EnterExpr(..))
                | Some(State::ContinueBinaryExpr(..)) => {
                    code_gen.enter_expression_anywhere(*expression)
                }
                _ => fail_transfer!(),
            },
            State::EnterExpr(expression, dest) => match from {
                Some(State::EnterStat(..))
                | Some(State::EnterExprAnywhere(..))
                | Some(State::EnterExpr(..))
                | Some(State::ContinueBinaryExpr(..)) => {
                    code_gen.enter_expression(*expression, *dest)
                }
                _ => fail_transfer!(),
            },
            State::ExitExpr(..) => match from {
                Some(State::EnterExprAnywhere(..))
                | Some(State::EnterExpr(..))
                | Some(State::ExitReturnExpr(..))
                | Some(State::ExitUnaryExpr(..))
                | Some(State::ExitBinaryExpr(..)) => Ok(()),
                _ => fail_transfer!(),
            },

            State::ExitVariableExpr(local, dest) => match from {
                Some(State::ExitExpr(..)) => code_gen.exit_variable_expression(*local, *dest),
                _ => fail_transfer!(),
            },
            State::ExitReturnExpr(dest) => match from {
                Some(State::ExitExpr(right)) => code_gen.exit_return_expression(right, *dest),
                _ => fail_transfer!(),
            },
            State::ExitUnaryExpr(op, dest) => match from {
                Some(State::ExitExpr(register)) => {
                    code_gen.exit_unary_expression(*op, register, *dest)
                }
                _ => fail_transfer!(),
            },
            State::ContinueBinaryExpr(op, right, dest) => match from {
                Some(State::ExitExpr(left)) => {
                    code_gen.continue_binary_expression(*op, left, *right, *dest)
                }
                _ => fail_transfer!(),
            },
            State::ExitBinaryExpr(op, left, dest) => match from {
                Some(State::ExitExpr(right)) => {
                    code_gen.exit_binary_expression(*op, *left, right, *dest)
                }
                _ => fail_transfer!(),
            },
        }
    }
}

struct CodeGen<'ast, I: StringInterner<String = PString>> {
    nodes: &'ast [Node],
    refs: &'ast [NodeRef],
    ref_cursor: usize,
    bindings: &'ast HashMap<NodeRef, Binding>,

    strings: I,

    state: Vec<State>,

    current_function: PrototypeBuilder,
}

impl<'ast, I: StringInterner<String = PString>> CodeGen<'ast, I> {
    fn get_node(&self, index: NodeRef) -> &'ast Node {
        &self.nodes[index.0 as usize]
    }

    fn get_statement(&self, index: NodeRef) -> Result<&'ast Stat> {
        match self.get_node(index) {
            Node::Stat(node) => Ok(node),
            _ => Err(CodeGenError::UnexpectedNode),
        }
    }

    fn get_expression(&self, index: NodeRef) -> Result<&'ast Expr> {
        match self.get_node(index) {
            Node::Expr(node) => Ok(node),
            _ => Err(CodeGenError::UnexpectedNode),
        }
    }

    fn get_next_ref(&mut self) -> NodeRef {
        let index = self.ref_cursor;
        self.ref_cursor += 1;
        self.refs[index]
    }

    fn push_state(&mut self, state: State) {
        self.state.push(state);
    }

    fn pop_state(&mut self) -> Option<State> {
        self.state.pop()
    }

    fn allocate(&mut self, dest: ExprDest) -> Result<MaybeTempRegister> {
        Ok(match dest {
            ExprDest::Register(register) => MaybeTempRegister::Protected(register),
            ExprDest::Anywhere => MaybeTempRegister::Temporary(
                self.current_function
                    .registers
                    .allocate_any()
                    .ok_or(CodeGenError::NoRegistersAvailable)?,
            ),
        })
    }

    fn free_temp(&mut self, register: impl IntoTempRegister) {
        if let Some(register) = register.into_temp_register() {
            self.current_function.registers.free(register);
        }
    }

    fn flatten_constant(
        &mut self,
        value: RegisterOrConstant16,
        dest: ExprDest,
    ) -> Result<RegisterOrConstant8> {
        Ok(match value {
            RegisterOrConstant16::Protected(r) => RegisterOrConstant8::Protected(r),
            RegisterOrConstant16::Temporary(r) => RegisterOrConstant8::Temporary(r),
            RegisterOrConstant16::Constant8(c) => RegisterOrConstant8::Constant8(c),
            RegisterOrConstant16::Constant16(c) => {
                let register = self.allocate(dest)?;
                self.push_instruction(Instruction::LoadC {
                    destination: register.into(),
                    constant: c,
                });
                register.into()
            }
        })
    }

    fn push_instruction(&mut self, instruction: Instruction) {
        self.current_function.instructions.push(instruction);
    }

    fn push_constant(&mut self, constant: Value) -> Result<CIndex> {
        Ok(match self.current_function.constants_map.entry(constant) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                const MAX_8: usize = CIndex8::MAX as usize;
                const MIN_16: usize = MAX_8 + 1;
                const MAX_16: usize = CIndex16::MAX as usize;

                let index = self.current_function.constants.len();
                let index = match index {
                    0..=MAX_8 => CIndex::Constant8(index as CIndex8),
                    MIN_16..=MAX_16 => CIndex::Constant16(index as CIndex16),
                    _ => return Err(CodeGenError::ConstantPoolFull),
                };
                self.current_function.constants.push(constant);
                entry.insert(index);
                index
            }
        })
    }

    fn finish(&mut self) -> Result<()> {
        // implicit return
        let constant = self.push_constant(Value::Integer(0))?;
        self.push_instruction(Instruction::ReturnC {
            constant: constant.into(),
        });

        Ok(())
    }

    // visit

    fn visit(&mut self) -> Result<()> {
        let mut previous = None;
        while let Some(mut state) = self.pop_state() {
            state.enter(previous, self)?;
            previous = Some(state);
        }

        Ok(())
    }

    // statements

    fn enter_statement(&mut self, node: NodeRef) -> Result<()> {
        let statement = self.get_statement(node)?;
        match statement {
            Stat::Compound { len } => {
                let stack_top = self.current_function.registers.stack_top();
                self.push_state(State::ExitCompoundStat(stack_top));
                self.push_state(State::ContinueCompoundStat(*len));

                Ok(())
            }
            Stat::VarDecl { def, .. } => {
                let binding = self
                    .bindings
                    .get(&node)
                    .ok_or(CodeGenError::MissingBinding)?;
                let local = binding.index;
                let register = self
                    .current_function
                    .registers
                    .allocate_any()
                    .ok_or(CodeGenError::NoRegistersAvailable)?;
                self.current_function
                    .registers
                    .assign_local(local, register);

                match def {
                    Some(definition) => {
                        self.push_state(State::ExitVarDecl);
                        self.push_state(State::EnterExpr(
                            *definition,
                            ExprDest::Register(register),
                        ));
                    }
                    None => {
                        let constant = self.push_constant(Value::Integer(0))?;
                        self.push_instruction(Instruction::LoadC {
                            destination: register.into(),
                            constant: constant.into(),
                        });
                        self.exit_variable_declaration(RegisterOrConstant16::Protected(register))?;
                    }
                }

                Ok(())
            }
            Stat::Expr { expr } => {
                self.push_state(State::ExitExprStat);
                self.push_state(State::EnterExprAnywhere(*expr));

                Ok(())
            }
        }
    }

    fn continue_compound_statement(&mut self, len: RefLen) -> Result<()> {
        if len.0 > 1 {
            let new_len = len.0 - 1;
            self.push_state(State::ContinueCompoundStat(RefLen(new_len)));
        }

        let next_statement = self.get_next_ref();
        self.push_state(State::EnterStat(next_statement));

        Ok(())
    }

    fn exit_compound_statement(&mut self, stack_top: u16) -> Result<()> {
        self.current_function.registers.pop_to(stack_top);
        self.push_state(State::ExitStat);
        Ok(())
    }

    fn exit_variable_declaration(&mut self, register: RegisterOrConstant16) -> Result<()> {
        assert!(
            matches!(register, RegisterOrConstant16::Protected(_)),
            "Register must be protected"
        );
        self.push_state(State::ExitStat);
        Ok(())
    }

    fn exit_expression_statement(&mut self, register: RegisterOrConstant16) -> Result<()> {
        self.free_temp(register);
        self.push_state(State::ExitStat);
        Ok(())
    }

    // expressions

    fn enter_expression_anywhere(&mut self, node: NodeRef) -> Result<()> {
        let expression = self.get_expression(node)?;
        match *expression {
            Expr::Integer(v) => {
                let constant = self.push_constant(Value::Integer(v))?;
                self.push_state(State::ExitExpr(constant.into()));

                Ok(())
            }
            Expr::Float(v) => {
                let constant = self.push_constant(Value::Float(v))?;
                self.push_state(State::ExitExpr(constant.into()));

                Ok(())
            }
            Expr::Var { assignment, .. } => {
                let binding = self
                    .bindings
                    .get(&node)
                    .ok_or(CodeGenError::MissingBinding)?;
                let local = binding.index;
                let local = self
                    .current_function
                    .registers
                    .address_of_local(local)
                    .ok_or(CodeGenError::MissingLocalRegister)?;
                match assignment {
                    None => {
                        self.push_state(State::ExitExpr(RegisterOrConstant16::Protected(local)));
                    }
                    Some(assignment) => {
                        self.push_state(State::EnterExpr(assignment, ExprDest::Register(local)));
                    }
                }
                Ok(())
            }
            Expr::Return { .. } => {
                // Execution can't continue after this,
                // no need to allocate an actual new register
                let register = Register::default();
                self.enter_expression(node, ExprDest::Register(register))?;

                Ok(())
            }
            _ => {
                self.enter_expression(node, ExprDest::Anywhere)?;

                Ok(())
            }
        }
    }

    fn enter_expression(&mut self, node: NodeRef, dest: ExprDest) -> Result<()> {
        let expression = self.get_expression(node)?;
        match *expression {
            Expr::Integer(v) => {
                let dest = self.allocate(dest)?;

                const MIN_16: i64 = i16::MIN as i64;
                const MAX_16: i64 = i16::MAX as i64;
                match v {
                    MIN_16..=MAX_16 => {
                        self.push_instruction(Instruction::LoadI {
                            destination: dest.into(),
                            integer: v as i16,
                        });
                    }
                    _ => {
                        let constant = self.push_constant(Value::Integer(v))?;
                        self.push_instruction(Instruction::LoadC {
                            destination: dest.into(),
                            constant: constant.into(),
                        });
                    }
                }
                self.push_state(State::ExitExpr(dest.into()));

                Ok(())
            }
            Expr::Float(v) => {
                let dest = self.allocate(dest)?;

                let v_as_integer = v as i64;
                const MIN_16: i64 = i16::MIN as i64;
                const MAX_16: i64 = i16::MAX as i64;
                match v_as_integer {
                    MIN_16..=MAX_16 if v_as_integer as f64 == v => {
                        self.push_instruction(Instruction::LoadI {
                            destination: dest.into(),
                            integer: v as i16,
                        });
                    }
                    _ => {
                        let constant = self.push_constant(Value::Float(v))?;
                        self.push_instruction(Instruction::LoadC {
                            destination: dest.into(),
                            constant: constant.into(),
                        });
                    }
                }

                self.push_state(State::ExitExpr(dest.into()));

                Ok(())
            }
            Expr::Var { assignment, .. } => {
                let binding = self
                    .bindings
                    .get(&node)
                    .ok_or(CodeGenError::MissingBinding)?;
                let local = binding.index;
                let local = self
                    .current_function
                    .registers
                    .address_of_local(local)
                    .ok_or(CodeGenError::MissingLocalRegister)?;
                match assignment {
                    Some(assignment) => {
                        self.push_state(State::ExitVariableExpr(local, dest));
                        self.push_state(State::EnterExpr(assignment, ExprDest::Register(local)));
                    }
                    None => {
                        self.exit_variable_expression(local, dest)?;
                    }
                }

                Ok(())
            }
            Expr::Return { right } => match right {
                Some(right) => {
                    self.push_state(State::ExitReturnExpr(dest));
                    self.push_state(State::EnterExprAnywhere(right));
                    Ok(())
                }
                None => {
                    let right = self.allocate(ExprDest::Anywhere)?;
                    let constant = self.push_constant(Value::Integer(0))?;
                    self.push_instruction(Instruction::LoadC {
                        destination: right.into(),
                        constant: constant.into(),
                    });
                    self.exit_return_expression(right.into(), dest)?;

                    Ok(())
                }
            },
            Expr::UnOp { op, right } => {
                self.push_state(State::ExitUnaryExpr(op, dest));
                self.push_state(State::EnterExprAnywhere(right));

                Ok(())
            }
            Expr::BinOp { op, left, right } => {
                self.push_state(State::ContinueBinaryExpr(op, right, dest));
                self.push_state(State::EnterExprAnywhere(left));

                Ok(())
            }
        }
    }

    fn exit_variable_expression(&mut self, local: Register, dest: ExprDest) -> Result<()> {
        let dest = self.allocate(dest)?;
        if local != dest.into() {
            self.push_instruction(Instruction::LoadR {
                destination: dest.into(),
                from: local.into(),
            });
        }
        self.push_state(State::ExitExpr(dest.into()));

        Ok(())
    }

    fn exit_return_expression(
        &mut self,
        right: RegisterOrConstant16,
        dest: ExprDest,
    ) -> Result<()> {
        self.free_temp(right);
        self.push_instruction(Instruction::ret(right));
        let dest = self.allocate(dest)?;
        self.push_state(State::ExitExpr(dest.into()));

        Ok(())
    }

    fn exit_unary_expression(
        &mut self,
        op: UnOp,
        right: RegisterOrConstant16,
        dest: ExprDest,
    ) -> Result<()> {
        self.free_temp(right);
        let dest = self.allocate(dest)?;
        let instruction = match op {
            UnOp::Neg => Instruction::neg(dest.into(), right),
        };
        self.push_instruction(instruction);
        self.push_state(State::ExitExpr(dest.into()));

        Ok(())
    }

    fn continue_binary_expression(
        &mut self,
        op: BinOp,
        left: RegisterOrConstant16,
        right: NodeRef,
        dest: ExprDest,
    ) -> Result<()> {
        self.push_state(State::ExitBinaryExpr(op, left, dest));
        self.push_state(State::EnterExprAnywhere(right));

        Ok(())
    }

    fn exit_binary_expression(
        &mut self,
        op: BinOp,
        left: RegisterOrConstant16,
        right: RegisterOrConstant16,
        dest: ExprDest,
    ) -> Result<()> {
        let left = self.flatten_constant(left, ExprDest::Anywhere)?;
        let right = self.flatten_constant(right, ExprDest::Anywhere)?;
        self.free_temp(left);
        self.free_temp(right);
        let dest = self.allocate(dest)?;
        let instruction = match op {
            BinOp::Add => Instruction::add(dest.into(), left, right),
            BinOp::Sub => Instruction::sub(dest.into(), left, right),
            BinOp::Mul => Instruction::mul(dest.into(), left, right),
            BinOp::Div => Instruction::div(dest.into(), left, right),
        };
        self.push_instruction(instruction);
        self.push_state(State::ExitExpr(dest.into()));

        Ok(())
    }
}

trait IntoTempRegister {
    fn into_temp_register(self) -> Option<Register>;
}

#[derive(Copy, Clone, Debug)]
enum RegisterOrConstant16 {
    Protected(Register),
    Temporary(Register),
    Constant8(CIndex8),
    Constant16(CIndex16),
}

impl IntoTempRegister for RegisterOrConstant16 {
    fn into_temp_register(self) -> Option<Register> {
        match self {
            RegisterOrConstant16::Temporary(r) => Some(r),
            _ => None,
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum RegisterOrConstant8 {
    Protected(Register),
    Temporary(Register),
    Constant8(CIndex8),
}

impl IntoTempRegister for RegisterOrConstant8 {
    fn into_temp_register(self) -> Option<Register> {
        match self {
            RegisterOrConstant8::Temporary(r) => Some(r),
            _ => None,
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum MaybeTempRegister {
    /// Should not be freed after use
    Protected(Register),
    /// Should be freed after use
    Temporary(Register),
}

impl IntoTempRegister for MaybeTempRegister {
    fn into_temp_register(self) -> Option<Register> {
        match self {
            MaybeTempRegister::Temporary(r) => Some(r),
            _ => None,
        }
    }
}

impl From<MaybeTempRegister> for RegisterOrConstant16 {
    fn from(value: MaybeTempRegister) -> Self {
        match value {
            MaybeTempRegister::Protected(r) => RegisterOrConstant16::Protected(r),
            MaybeTempRegister::Temporary(r) => RegisterOrConstant16::Temporary(r),
        }
    }
}

impl From<MaybeTempRegister> for RegisterOrConstant8 {
    fn from(value: MaybeTempRegister) -> Self {
        match value {
            MaybeTempRegister::Protected(r) => RegisterOrConstant8::Protected(r),
            MaybeTempRegister::Temporary(r) => RegisterOrConstant8::Temporary(r),
        }
    }
}

impl From<MaybeTempRegister> for Register {
    fn from(value: MaybeTempRegister) -> Self {
        match value {
            MaybeTempRegister::Protected(r) | MaybeTempRegister::Temporary(r) => r,
        }
    }
}

impl From<MaybeTempRegister> for RIndex {
    fn from(value: MaybeTempRegister) -> Self {
        match value {
            MaybeTempRegister::Protected(r) | MaybeTempRegister::Temporary(r) => r.into(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum CIndex {
    Constant8(CIndex8),
    Constant16(CIndex16),
}

impl From<CIndex> for RegisterOrConstant16 {
    fn from(value: CIndex) -> Self {
        match value {
            CIndex::Constant8(c) => RegisterOrConstant16::Constant8(c),
            CIndex::Constant16(c) => RegisterOrConstant16::Constant16(c),
        }
    }
}

impl From<CIndex> for CIndex16 {
    fn from(value: CIndex) -> Self {
        match value {
            CIndex::Constant8(c) => c.into(),
            CIndex::Constant16(c) => c,
        }
    }
}

#[derive(Copy, Clone, Debug)]
#[must_use]
enum ExprDest {
    /// Existing register
    Register(Register),
    /// Allocated a register anywhere
    Anywhere,
}

impl Instruction {
    fn ret(right: RegisterOrConstant16) -> Instruction {
        match right {
            RegisterOrConstant16::Protected(r) | RegisterOrConstant16::Temporary(r) => {
                Instruction::ReturnR { register: r.into() }
            }
            RegisterOrConstant16::Constant8(c) => Instruction::ReturnC { constant: c.into() },
            RegisterOrConstant16::Constant16(c) => Instruction::ReturnC { constant: c },
        }
    }

    // fn load(dest: Register, from: RegisterOrConstant16) -> Instruction {
    //     match from {
    //         RegisterOrConstant16::Protected(r) | RegisterOrConstant16::Temporary(r) => {
    //             Instruction::LoadR {
    //                 destination: dest.into(),
    //                 from: r.into(),
    //             }
    //         }
    //         RegisterOrConstant16::Constant8(c) => Instruction::LoadC {
    //             destination: dest.into(),
    //             constant: c.into(),
    //         },
    //         RegisterOrConstant16::Constant16(c) => Instruction::LoadC {
    //             destination: dest.into(),
    //             constant: c,
    //         },
    //     }
    // }

    fn neg(dest: Register, right: RegisterOrConstant16) -> Instruction {
        match right {
            RegisterOrConstant16::Protected(r) | RegisterOrConstant16::Temporary(r) => {
                Instruction::NegR {
                    destination: dest.into(),
                    right: r.into(),
                }
            }
            RegisterOrConstant16::Constant8(c) => Instruction::NegC {
                destination: dest.into(),
                right: c.into(),
            },
            RegisterOrConstant16::Constant16(c) => Instruction::NegC {
                destination: dest.into(),
                right: c,
            },
        }
    }

    fn add(dest: Register, left: RegisterOrConstant8, right: RegisterOrConstant8) -> Instruction {
        match (left, right) {
            (
                RegisterOrConstant8::Protected(l) | RegisterOrConstant8::Temporary(l),
                RegisterOrConstant8::Protected(r) | RegisterOrConstant8::Temporary(r),
            ) => Instruction::AddRR {
                destination: dest.into(),
                left: l.into(),
                right: r.into(),
            },
            (
                RegisterOrConstant8::Protected(l) | RegisterOrConstant8::Temporary(l),
                RegisterOrConstant8::Constant8(r),
            ) => Instruction::AddRC {
                destination: dest.into(),
                left: l.into(),
                right: r,
            },
            (
                RegisterOrConstant8::Constant8(l),
                RegisterOrConstant8::Protected(r) | RegisterOrConstant8::Temporary(r),
            ) => Instruction::AddCR {
                destination: dest.into(),
                left: l,
                right: r.into(),
            },
            (RegisterOrConstant8::Constant8(l), RegisterOrConstant8::Constant8(r)) => {
                Instruction::AddCC {
                    destination: dest.into(),
                    left: l,
                    right: r,
                }
            }
        }
    }

    fn sub(dest: Register, left: RegisterOrConstant8, right: RegisterOrConstant8) -> Instruction {
        match (left, right) {
            (
                RegisterOrConstant8::Protected(l) | RegisterOrConstant8::Temporary(l),
                RegisterOrConstant8::Protected(r) | RegisterOrConstant8::Temporary(r),
            ) => Instruction::SubRR {
                destination: dest.into(),
                left: l.into(),
                right: r.into(),
            },
            (
                RegisterOrConstant8::Protected(l) | RegisterOrConstant8::Temporary(l),
                RegisterOrConstant8::Constant8(r),
            ) => Instruction::SubRC {
                destination: dest.into(),
                left: l.into(),
                right: r,
            },
            (
                RegisterOrConstant8::Constant8(l),
                RegisterOrConstant8::Protected(r) | RegisterOrConstant8::Temporary(r),
            ) => Instruction::SubCR {
                destination: dest.into(),
                left: l,
                right: r.into(),
            },
            (RegisterOrConstant8::Constant8(l), RegisterOrConstant8::Constant8(r)) => {
                Instruction::SubCC {
                    destination: dest.into(),
                    left: l,
                    right: r,
                }
            }
        }
    }

    fn mul(dest: Register, left: RegisterOrConstant8, right: RegisterOrConstant8) -> Instruction {
        match (left, right) {
            (
                RegisterOrConstant8::Protected(l) | RegisterOrConstant8::Temporary(l),
                RegisterOrConstant8::Protected(r) | RegisterOrConstant8::Temporary(r),
            ) => Instruction::MulRR {
                destination: dest.into(),
                left: l.into(),
                right: r.into(),
            },
            (
                RegisterOrConstant8::Protected(l) | RegisterOrConstant8::Temporary(l),
                RegisterOrConstant8::Constant8(r),
            ) => Instruction::MulRC {
                destination: dest.into(),
                left: l.into(),
                right: r,
            },
            (
                RegisterOrConstant8::Constant8(l),
                RegisterOrConstant8::Protected(r) | RegisterOrConstant8::Temporary(r),
            ) => Instruction::MulCR {
                destination: dest.into(),
                left: l,
                right: r.into(),
            },
            (RegisterOrConstant8::Constant8(l), RegisterOrConstant8::Constant8(r)) => {
                Instruction::MulCC {
                    destination: dest.into(),
                    left: l,
                    right: r,
                }
            }
        }
    }

    fn div(dest: Register, left: RegisterOrConstant8, right: RegisterOrConstant8) -> Instruction {
        match (left, right) {
            (
                RegisterOrConstant8::Protected(l) | RegisterOrConstant8::Temporary(l),
                RegisterOrConstant8::Protected(r) | RegisterOrConstant8::Temporary(r),
            ) => Instruction::DivRR {
                destination: dest.into(),
                left: l.into(),
                right: r.into(),
            },
            (
                RegisterOrConstant8::Protected(l) | RegisterOrConstant8::Temporary(l),
                RegisterOrConstant8::Constant8(r),
            ) => Instruction::MulRC {
                destination: dest.into(),
                left: l.into(),
                right: r,
            },
            (
                RegisterOrConstant8::Constant8(l),
                RegisterOrConstant8::Protected(r) | RegisterOrConstant8::Temporary(r),
            ) => Instruction::MulCR {
                destination: dest.into(),
                left: l,
                right: r.into(),
            },
            (RegisterOrConstant8::Constant8(l), RegisterOrConstant8::Constant8(r)) => {
                Instruction::MulCC {
                    destination: dest.into(),
                    left: l,
                    right: r,
                }
            }
        }
    }
}
