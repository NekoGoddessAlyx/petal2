use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::rc::Rc;

use crate::compiler::ast::{BinOp, NodeRef, RefLen, UnOp};
use crate::compiler::registers::{Register, Registers};
use crate::prototype::{ConstantIndex, Instruction, Prototype};
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
        ref_cursor: ast.ast.refs.len(),
        bindings: &ast.bindings,

        strings,

        state: Vec::with_capacity(32),

        current_function,
    };

    let root = ast.ast.root();
    code_gen.push_state(State::EnterStat(root));

    code_gen.visit()?;
    code_gen.finish()?;

    Ok(code_gen.current_function.build())
}

struct PrototypeBuilder {
    name: PString,
    registers: Registers,
    instructions: Vec<Instruction>,
    constants_map: HashMap<Value, ConstantIndex>,
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

    ExitCompoundStat(u16),
    ExitVarDecl,
    ExitExprStat,

    // expressions
    EnterExprAnywhere(NodeRef),
    EnterExpr(NodeRef, ExprDest),
    ExitExpr(MaybeTempRegister),

    ExitReturnExpr(ExprDest),
    ExitUnaryExpr(UnOp, ExprDest),
    ContinueBinaryExpr(BinOp, NodeRef, ExprDest),
    ExitBinaryExpr(BinOp, MaybeTempRegister, ExprDest),
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
                None | Some(State::ExitStat) | Some(State::EnterStat(..)) => {
                    code_gen.enter_statement(*statement)
                }
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
                | Some(State::ContinueBinaryExpr(..)) => {
                    code_gen.enter_expression_anywhere(*expression)
                }
                _ => fail_transfer!(),
            },
            State::EnterExpr(expression, dest) => match from {
                Some(State::EnterStat(..))
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

    fn get_refs(&mut self, len: RefLen) -> &'ast [NodeRef] {
        let end_index = self.ref_cursor;
        let start_index = end_index - len.0 as usize;
        self.ref_cursor = start_index;
        &self.refs[start_index..end_index]
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

    fn free_temp(&mut self, register: MaybeTempRegister) {
        if let MaybeTempRegister::Temporary(register) = register {
            self.current_function.registers.free(register);
        }
    }

    fn push_instruction(&mut self, instruction: Instruction) {
        self.current_function.instructions.push(instruction);
    }

    fn push_constant(&mut self, constant: Value) -> Result<ConstantIndex> {
        Ok(match self.current_function.constants_map.entry(constant) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let index = self.current_function.constants.len();
                if index > ConstantIndex::MAX as usize {
                    return Err(CodeGenError::ConstantPoolFull);
                }
                let index = index as ConstantIndex;
                self.current_function.constants.push(constant);
                entry.insert(index);
                index
            }
        })
    }

    fn finish(&mut self) -> Result<()> {
        // got ahead of myself

        // implicit return of... 0
        // for meow
        // let register = self
        //     .registers
        //     .allocate_any()
        //     .ok_or(CodeGenError::NoRegistersAvailable)?;
        // let constant = self.push_constant(Value::Integer(0))?;
        // self.push_instruction(Instruction::LoadConstant {
        //     destination: register.into(),
        //     constant,
        // });
        self.push_instruction(Instruction::Return { register: 0 });
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
            Stat::Compound(len) => {
                let stack_top = self.current_function.registers.stack_top();
                self.push_state(State::ExitCompoundStat(stack_top));

                let statements = self.get_refs(*len).iter().rev();
                for statement in statements {
                    self.push_state(State::EnterStat(*statement));
                }

                Ok(())
            }
            Stat::VarDecl(_, definition) => {
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

                match definition {
                    Some(definition) => {
                        self.push_state(State::ExitVarDecl);
                        self.push_state(State::EnterExpr(
                            *definition,
                            ExprDest::Register(register),
                        ));
                    }
                    None => {
                        let constant = self.push_constant(Value::Integer(0))?;
                        self.push_instruction(Instruction::LoadConstant {
                            destination: register.into(),
                            constant,
                        });
                        self.exit_variable_declaration(MaybeTempRegister::Protected(register))?;
                    }
                }
                Ok(())
            }
            Stat::Expr(expression) => {
                self.push_state(State::ExitExprStat);
                self.push_state(State::EnterExprAnywhere(*expression));
                Ok(())
            }
        }
    }

    fn exit_compound_statement(&mut self, stack_top: u16) -> Result<()> {
        self.current_function.registers.pop_to(stack_top);
        self.push_state(State::ExitStat);
        Ok(())
    }

    fn exit_variable_declaration(&mut self, register: MaybeTempRegister) -> Result<()> {
        assert!(
            matches!(register, MaybeTempRegister::Protected(_)),
            "Register must be protected"
        );
        self.push_state(State::ExitStat);
        Ok(())
    }

    fn exit_expression_statement(&mut self, register: MaybeTempRegister) -> Result<()> {
        self.free_temp(register);
        self.push_state(State::ExitStat);
        Ok(())
    }

    // expressions

    fn enter_expression_anywhere(&mut self, node: NodeRef) -> Result<()> {
        let expression = self.get_expression(node)?;
        match *expression {
            Expr::Var(_) => {
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
                self.push_state(State::ExitExpr(MaybeTempRegister::Protected(local)));
                Ok(())
            }
            Expr::Return(..) => {
                // Execution can't continue after this,
                // no need to allocate an actual new register
                let register = Register::default();
                self.enter_expression(node, ExprDest::Register(register))
            }
            _ => self.enter_expression(node, ExprDest::Anywhere),
        }
    }

    fn enter_expression(&mut self, node: NodeRef, dest: ExprDest) -> Result<()> {
        let expression = self.get_expression(node)?;
        match *expression {
            Expr::Integer(v) => {
                let dest = self.allocate(dest)?;
                let constant = self.push_constant(Value::Integer(v))?;
                self.push_instruction(Instruction::LoadConstant {
                    destination: dest.into(),
                    constant,
                });
                self.push_state(State::ExitExpr(dest));
                Ok(())
            }
            Expr::Float(v) => {
                let dest = self.allocate(dest)?;
                let constant = self.push_constant(Value::Float(v))?;
                self.push_instruction(Instruction::LoadConstant {
                    destination: dest.into(),
                    constant,
                });
                self.push_state(State::ExitExpr(dest));
                Ok(())
            }
            Expr::Var(_) => {
                let dest = self.allocate(dest)?;
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
                if local != dest.into() {
                    self.push_instruction(Instruction::Move {
                        destination: dest.into(),
                        from: local.into(),
                    });
                }
                self.push_state(State::ExitExpr(dest));
                Ok(())
            }
            Expr::Return(right) => match right {
                Some(right) => {
                    self.push_state(State::ExitReturnExpr(dest));
                    self.push_state(State::EnterExprAnywhere(right));
                    Ok(())
                }
                None => {
                    let right = self.allocate(ExprDest::Anywhere)?;
                    let constant = self.push_constant(Value::Integer(0))?;
                    self.push_instruction(Instruction::LoadConstant {
                        destination: right.into(),
                        constant,
                    });
                    self.exit_return_expression(right, dest)
                }
            },
            Expr::UnOp(op, right) => {
                self.push_state(State::ExitUnaryExpr(op, dest));
                self.push_state(State::EnterExprAnywhere(right));
                Ok(())
            }
            Expr::BinOp(op, left, right) => {
                self.push_state(State::ContinueBinaryExpr(op, right, dest));
                self.push_state(State::EnterExprAnywhere(left));
                Ok(())
            }
        }
    }

    fn exit_return_expression(&mut self, right: MaybeTempRegister, dest: ExprDest) -> Result<()> {
        self.free_temp(right);
        self.push_instruction(Instruction::Return {
            register: right.into(),
        });
        let dest = self.allocate(dest)?;
        self.push_state(State::ExitExpr(dest));
        Ok(())
    }

    fn exit_unary_expression(
        &mut self,
        op: UnOp,
        from: MaybeTempRegister,
        dest: ExprDest,
    ) -> Result<()> {
        self.free_temp(from);
        let dest = self.allocate(dest)?;
        let instruction = match op {
            UnOp::Neg => Instruction::Neg {
                destination: dest.into(),
                right: from.into(),
            },
        };
        self.push_instruction(instruction);
        self.push_state(State::ExitExpr(dest));
        Ok(())
    }

    fn continue_binary_expression(
        &mut self,
        op: BinOp,
        left: MaybeTempRegister,
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
        left: MaybeTempRegister,
        right: MaybeTempRegister,
        dest: ExprDest,
    ) -> Result<()> {
        self.free_temp(left);
        self.free_temp(right);
        let dest = self.allocate(dest)?;
        let instruction = match op {
            BinOp::Add => Instruction::Add {
                destination: dest.into(),
                left: left.into(),
                right: right.into(),
            },
            BinOp::Sub => Instruction::Sub {
                destination: dest.into(),
                left: left.into(),
                right: right.into(),
            },
            BinOp::Mul => Instruction::Mul {
                destination: dest.into(),
                left: left.into(),
                right: right.into(),
            },
            BinOp::Div => Instruction::Div {
                destination: dest.into(),
                left: left.into(),
                right: right.into(),
            },
        };
        self.push_instruction(instruction);
        self.push_state(State::ExitExpr(dest));
        Ok(())
    }
}

#[derive(Copy, Clone, Debug)]
pub enum MaybeTempRegister {
    /// Should not be freed after use
    Protected(Register),
    /// Should be freed after use
    Temporary(Register),
}

impl From<MaybeTempRegister> for Register {
    fn from(value: MaybeTempRegister) -> Self {
        match value {
            MaybeTempRegister::Protected(r) | MaybeTempRegister::Temporary(r) => r,
        }
    }
}

impl From<MaybeTempRegister> for crate::prototype::Register {
    fn from(value: MaybeTempRegister) -> Self {
        match value {
            MaybeTempRegister::Protected(r) | MaybeTempRegister::Temporary(r) => r.into(),
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
