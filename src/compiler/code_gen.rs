use std::collections::hash_map::Entry;
use std::collections::HashMap;

use crate::compiler::ast::{BinOp, Expr, NodeRef, RefLen, UnOp};
use crate::compiler::registers::{Register, Registers};
use crate::prototype::{ConstantIndex, Instruction, Prototype};
use crate::value::Value;
use crate::{PString, StringInterner};

type Ast = crate::compiler::ast::Ast<PString>;
type Node = crate::compiler::ast::Node<PString>;
type Stat = crate::compiler::ast::Stat<PString>;

#[derive(Debug)]
pub enum CodeGenError {
    UnexpectedNode,
    BadStateTransfer,
    NoScopeAvailable,
    NameInScope,
    NoRegistersAvailable,
    ConstantPoolFull,
}

pub type Result<T> = std::result::Result<T, CodeGenError>;

pub fn code_gen<I: StringInterner<String = PString>>(
    ast: Ast,
    mut strings: I,
) -> Result<Prototype> {
    let name = strings.intern(b"test");
    let mut current_function = PrototypeBuilder {
        name,
        scopes: Vec::with_capacity(8),
        num_locals: 0,
        registers: Registers::new(),
        instructions: Vec::with_capacity(64),
        constants_map: HashMap::with_capacity(32),
        constants: Vec::with_capacity(32),
    };

    let mut code_gen = CodeGen {
        nodes: &ast.nodes,
        refs: &ast.refs,
        ref_cursor: ast.refs.len(),

        strings,

        state: Vec::with_capacity(32),

        current_function,
    };

    let root_ref = ast.root();
    let root = code_gen.get_statement(root_ref)?;
    code_gen.push_state(State::ExitStat);
    code_gen.push_state(State::EnterStat(root));

    code_gen.visit()?;
    code_gen.finish()?;

    Ok(code_gen.current_function.build())
}

struct PrototypeBuilder {
    name: PString,
    scopes: Vec<Scope>,
    num_locals: u8,
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
enum State<'ast> {
    // statements
    EnterStat(&'ast Stat),
    ExitStat,

    ExitCompoundStat(u16),
    ExitVarDecl(Local),
    ExitExprStat,

    // expressions
    EnterExpr(&'ast Expr),
    ExitExpr(Register),

    ExitUnaryExpr(UnOp),
    ContinueBinaryExpr(BinOp, &'ast Expr),
    ExitBinaryExpr(BinOp, Register),
}

impl<'ast> State<'ast> {
    fn enter<I: StringInterner<String = PString>>(
        &mut self,
        from: Option<State>,
        code_gen: &mut CodeGen<'ast, I>,
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
                    code_gen.enter_statement(statement)
                }
                _ => fail_transfer!(),
            },
            State::ExitStat => match from {
                Some(State::EnterStat(..))
                | Some(State::ExitStat)
                | Some(State::ExitCompoundStat(..))
                | Some(State::ExitVarDecl(..))
                | Some(State::ExitExprStat) => Ok(()),
                _ => fail_transfer!(),
            },

            State::ExitCompoundStat(stack_top) => match from {
                Some(State::EnterStat(..)) | Some(State::ExitStat) => {
                    code_gen.exit_compound_statement(*stack_top)
                }
                _ => fail_transfer!(),
            },
            State::ExitVarDecl(local) => match from {
                Some(State::ExitExpr(register)) => {
                    code_gen.exit_variable_declaration(*local, register)
                }
                _ => fail_transfer!(),
            },
            State::ExitExprStat => match from {
                Some(State::ExitExpr(register)) => code_gen.exit_expression_statement(register),
                _ => fail_transfer!(),
            },

            // expressions
            State::EnterExpr(expression) => match from {
                Some(State::EnterStat(..))
                | Some(State::EnterExpr(..))
                | Some(State::ContinueBinaryExpr(..)) => code_gen.enter_expression(expression),
                _ => fail_transfer!(),
            },
            State::ExitExpr(..) => match from {
                Some(State::EnterExpr(..))
                | Some(State::ExitUnaryExpr(..))
                | Some(State::ExitBinaryExpr(..)) => Ok(()),
                _ => fail_transfer!(),
            },

            State::ExitUnaryExpr(op) => match from {
                Some(State::ExitExpr(register)) => code_gen.exit_unary_expression(*op, register),
                _ => fail_transfer!(),
            },
            State::ContinueBinaryExpr(op, right) => match from {
                Some(State::ExitExpr(left)) => {
                    code_gen.continue_binary_expression(*op, left, right)
                }
                _ => fail_transfer!(),
            },
            State::ExitBinaryExpr(op, left) => match from {
                Some(State::ExitExpr(right)) => code_gen.exit_binary_expression(*op, *left, right),
                _ => fail_transfer!(),
            },
        }
    }
}

struct CodeGen<'ast, I: StringInterner<String = PString>> {
    nodes: &'ast [Node],
    refs: &'ast [NodeRef],
    ref_cursor: usize,

    strings: I,

    state: Vec<State<'ast>>,

    current_function: PrototypeBuilder,
}

impl<'ast, 'prototype, I: StringInterner<String = PString>> CodeGen<'ast, I> {
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

    fn push_state(&mut self, state: State<'ast>) {
        self.state.push(state);
    }

    fn pop_state(&mut self) -> Option<State<'ast>> {
        self.state.pop()
    }

    fn begin_scope(&mut self) {
        self.current_function.scopes.push(Scope::new());
    }

    fn end_scope(&mut self) -> Result<()> {
        let scope = self
            .current_function
            .scopes
            .pop()
            .ok_or(CodeGenError::NoScopeAvailable)?;
        self.current_function.num_locals -= scope.0.len() as u8;
        Ok(())
    }

    fn declare(&mut self, name: PString) -> Result<Local> {
        let scope = self
            .current_function
            .scopes
            .last_mut()
            .ok_or(CodeGenError::NoScopeAvailable)?;

        match scope.0.entry(name) {
            Entry::Occupied(_) => Err(CodeGenError::NameInScope),
            Entry::Vacant(entry) => {
                let local = Local(self.current_function.num_locals);
                entry.insert(local);

                self.current_function.num_locals += 1;

                Ok(local)
            }
        }
    }

    fn lookup(&self, name: PString) -> Option<Local> {
        for scope in self.current_function.scopes.iter().rev() {
            if let Some(&binding) = scope.0.get(&name) {
                return Some(binding);
            }
        }

        None
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

    fn enter_statement(&mut self, node: &Stat) -> Result<()> {
        match node {
            Stat::Compound(len) => {
                self.begin_scope();
                let stack_top = self.current_function.registers.stack_top();
                self.push_state(State::ExitCompoundStat(stack_top));
                let statements = self.get_refs(*len).iter().rev();
                for statement in statements {
                    let statement = self.get_statement(*statement)?;
                    self.push_state(State::EnterStat(statement));
                }
                Ok(())
            }
            Stat::VarDecl(name, definition) => {
                let local = self.declare(name.clone())?;

                match definition {
                    Some(definition) => {
                        let definition = self.get_expression(*definition)?;
                        self.push_state(State::ExitVarDecl(local));
                        self.push_state(State::EnterExpr(definition));
                    }
                    None => {
                        let register = self
                            .current_function
                            .registers
                            .allocate_any()
                            .ok_or(CodeGenError::NoRegistersAvailable)?;
                        let constant = self.push_constant(Value::Integer(0))?;
                        self.push_instruction(Instruction::LoadConstant {
                            destination: register.into(),
                            constant: constant.into(),
                        });
                        self.exit_variable_declaration(local, register)?;
                    }
                }
                Ok(())
            }
            Stat::Expr(expression) => {
                let expression = self.get_expression(*expression)?;
                self.push_state(State::ExitExprStat);
                self.push_state(State::EnterExpr(expression));
                Ok(())
            }
        }
    }

    fn exit_compound_statement(&mut self, stack_top: u16) -> Result<()> {
        self.end_scope()?;
        self.current_function.registers.pop_to(stack_top);
        self.push_state(State::ExitStat);
        Ok(())
    }

    fn exit_variable_declaration(&mut self, local: Local, register: Register) -> Result<()> {
        self.current_function
            .registers
            .assign_local(local, register);
        self.push_state(State::ExitStat);
        Ok(())
    }

    fn exit_expression_statement(&mut self, register: Register) -> Result<()> {
        self.current_function.registers.free(register);
        self.push_state(State::ExitStat);
        Ok(())
    }

    // expressions

    fn enter_expression(&mut self, node: &Expr) -> Result<()> {
        match *node {
            Expr::Integer(v) => {
                let register = self
                    .current_function
                    .registers
                    .allocate_any()
                    .ok_or(CodeGenError::NoRegistersAvailable)?;
                let constant = self.push_constant(Value::Integer(v))?;
                self.push_instruction(Instruction::LoadConstant {
                    destination: register.into(),
                    constant,
                });
                self.push_state(State::ExitExpr(register));
                Ok(())
            }
            Expr::Float(v) => {
                let register = self
                    .current_function
                    .registers
                    .allocate_any()
                    .ok_or(CodeGenError::NoRegistersAvailable)?;
                let constant = self.push_constant(Value::Float(v))?;
                self.push_instruction(Instruction::LoadConstant {
                    destination: register.into(),
                    constant,
                });
                self.push_state(State::ExitExpr(register));
                Ok(())
            }
            Expr::UnOp(op, right) => {
                self.push_state(State::ExitUnaryExpr(op));
                let right = self.get_expression(right)?;
                self.push_state(State::EnterExpr(right));
                Ok(())
            }
            Expr::BinOp(op, left, right) => {
                let right = self.get_expression(right)?;
                self.push_state(State::ContinueBinaryExpr(op, right));
                let left = self.get_expression(left)?;
                self.push_state(State::EnterExpr(left));
                Ok(())
            }
        }
    }

    fn exit_unary_expression(&mut self, op: UnOp, from: Register) -> Result<()> {
        self.current_function.registers.free(from);
        let destination = self
            .current_function
            .registers
            .allocate_any()
            .ok_or(CodeGenError::NoRegistersAvailable)?;
        let instruction = match op {
            UnOp::Neg => Instruction::Neg {
                destination: destination.into(),
                right: from.into(),
            },
        };
        self.push_instruction(instruction);
        self.push_state(State::ExitExpr(destination));
        Ok(())
    }

    fn continue_binary_expression(
        &mut self,
        op: BinOp,
        left: Register,
        right: &'ast Expr,
    ) -> Result<()> {
        self.push_state(State::ExitBinaryExpr(op, left));
        self.push_state(State::EnterExpr(right));
        Ok(())
    }

    fn exit_binary_expression(&mut self, op: BinOp, left: Register, right: Register) -> Result<()> {
        self.current_function.registers.free(left);
        self.current_function.registers.free(right);
        let destination = self
            .current_function
            .registers
            .allocate_any()
            .ok_or(CodeGenError::NoRegistersAvailable)?;
        let instruction = match op {
            BinOp::Add => Instruction::Add {
                destination: destination.into(),
                left: left.into(),
                right: right.into(),
            },
            BinOp::Sub => Instruction::Sub {
                destination: destination.into(),
                left: left.into(),
                right: right.into(),
            },
            BinOp::Mul => Instruction::Mul {
                destination: destination.into(),
                left: left.into(),
                right: right.into(),
            },
            BinOp::Div => Instruction::Div {
                destination: destination.into(),
                left: left.into(),
                right: right.into(),
            },
        };
        self.push_instruction(instruction);
        self.push_state(State::ExitExpr(destination));
        Ok(())
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Local(pub u8);

struct Scope(HashMap<PString, Local>);

impl Scope {
    fn new() -> Self {
        Self(HashMap::new())
    }
}
