use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::Display;
use std::ops::{Neg, Not};

use gc_arena::Mutation;
use smallvec::{smallvec, SmallVec};
use thiserror::Error;

use crate::compiler::ast::{Ast2, Ast2Iterator, BinOp, NodeError, NodeRef, RefLen, Root, UnOp};
use crate::compiler::callback::Diagnostic;
use crate::compiler::lexer::{LineNumber, Span};
use crate::compiler::registers::{Register, Registers};
use crate::instruction::{CIndex16, CIndex8, Instruction, RIndex};
use crate::prototype::Prototype;
use crate::value::Value;
use crate::{MessageKind, PString, StringInterner};

type Stat<'gc> = crate::compiler::ast::Stat<PString<'gc>>;
type Expr<'gc> = crate::compiler::ast::Expr<PString<'gc>>;

#[derive(Debug, Error)]
pub enum CodeGenError {
    #[error("{}", .0)]
    CodeGenFailed(#[from] CodeGenMessage),
    #[error(transparent)]
    NodeError(#[from] NodeError),
    #[error("Invalid state transition")]
    BadTransition,
    #[error("Missing variable binding")]
    MissingBinding,
    #[error("Local does not have a register assigned")]
    MissingLocalRegister,
    #[error("Invalid jump label")]
    InvalidJumpLabel,
    #[error("Negative conditional jump")]
    NegativeConditionalJump,
}

#[derive(Debug, Error)]
pub enum CodeGenMessage {
    #[error("Jump is too large")]
    JumpTooLarge,
    #[error("No registers are available. Your function is too large.")]
    NoRegistersAvailable,
    #[error("Constant pool is full. Your function is too large.")]
    ConstantPoolFull,
}

impl Diagnostic for CodeGenMessage {
    fn kind(&self) -> MessageKind {
        MessageKind::Error
    }

    fn message(&self) -> &dyn Display {
        self
    }
}

pub type Result<T> = std::result::Result<T, CodeGenError>;

enum LastLineNumber {
    Node(NodeRef),
    LineNumber(LineNumber),
}

impl From<NodeRef> for LastLineNumber {
    fn from(value: NodeRef) -> Self {
        Self::Node(value)
    }
}

impl From<LineNumber> for LastLineNumber {
    fn from(value: LineNumber) -> Self {
        Self::LineNumber(value)
    }
}

pub fn code_gen<'gc, I, L>(
    mc: &Mutation<'gc>,
    ast: Ast2<PString<'gc>>,
    mut strings: I,
    get_line_number: L,
) -> Result<Prototype<'gc>>
where
    I: StringInterner<'gc, String = PString<'gc>>,
    L: Fn(Span) -> LineNumber,
{
    let name = strings.intern(mc, b"test");
    let current_function = PrototypeBuilder {
        name,
        registers: Registers::new(),
        instructions: Vec::with_capacity(64),
        line_numbers: Vec::with_capacity(64),
        constants_map: HashMap::with_capacity(32),
        constants: Vec::with_capacity(32),
    };

    let mut code_gen = CodeGen {
        mc,
        ast: ast.iterator(),

        strings,
        get_line_number,
        last_line_number: LastLineNumber::LineNumber(LineNumber::new(1)),

        state: smallvec![],

        current_function,
    };

    code_gen.push_state(State::EnterRoot);

    code_gen.visit()?;
    code_gen.finish()?;

    Ok(code_gen.current_function.build())
}

struct PrototypeBuilder<'gc> {
    name: PString<'gc>,
    registers: Registers,
    instructions: Vec<Instruction>,
    constants_map: HashMap<Value<'gc>, CIndex>,
    constants: Vec<Value<'gc>>,
    line_numbers: Vec<u32>,
}

impl<'gc> PrototypeBuilder<'gc> {
    fn build(self) -> Prototype<'gc> {
        Prototype {
            name: self.name,
            stack_size: self.registers.stack_size().saturating_sub(1) as u8,
            instructions: self.instructions.into_boxed_slice(),
            line_numbers: self.line_numbers.into_boxed_slice(),
            constants: self.constants.into_boxed_slice(),
        }
    }
}

#[derive(Debug)]
enum State<'gc> {
    // root
    EnterRoot,

    // statements
    EnterStat,
    ExitStat,

    ContinueCompoundStat(RefLen),
    ExitCompoundStat(u16),
    ExitVarDecl,
    ExitIfStatCondition(bool),
    ExitIfStatBody(Label, bool),
    ExitIfStatElseBody(Label),
    ExitIfStat,
    ExitExprStat,

    // expressions
    EnterExprAnywhere,
    EnterExpr(ExprDest),
    ExitExpr(AnyExpr<'gc>),

    ExitVariableExpr(Register, ExprDest),
    ExitReturnExpr(ExprDest),
    ExitUnaryExpr(UnOp, ExprDest),
    ContinueBinaryExpr(BinOp, ExprDest),
    ExitBinaryExpr(BinOp, AnyExpr<'gc>, ExprDest),
    ContinueBlockExpr(RefLen),
    ExitBlockExpr(u16, ExprDest),

    // skip
    SkipStat,
    SkipCompoundStat(RefLen),
    SkipExpr,
    ExitSkip,
}

impl<'gc> State<'gc> {
    fn enter<I, L>(
        &mut self,
        from: Option<State<'gc>>,
        code_gen: &mut CodeGen<'gc, '_, I, L>,
    ) -> Result<()>
    where
        I: StringInterner<'gc, String = PString<'gc>>,
        L: Fn(Span) -> LineNumber,
    {
        macro_rules! fail_transfer {
            () => {{
                dbg!(&from);
                Err(CodeGenError::BadTransition)
            }};
        }

        match self {
            // root
            State::EnterRoot => match from {
                None => code_gen.enter_root(),
                _ => fail_transfer!(),
            },

            // statements
            State::EnterStat => match from {
                Some(State::EnterRoot)
                | Some(State::ExitStat)
                | Some(State::EnterStat)
                | Some(State::ContinueCompoundStat(..))
                | Some(State::ExitIfStatCondition(..))
                | Some(State::ExitIfStatBody(..))
                | Some(State::ContinueBlockExpr(..)) => code_gen.enter_statement(),
                _ => fail_transfer!(),
            },
            State::ExitStat => match from {
                Some(State::EnterStat)
                | Some(State::ExitStat)
                | Some(State::ExitCompoundStat(..))
                | Some(State::ExitVarDecl)
                | Some(State::ExitIfStat)
                | Some(State::ExitExprStat) => Ok(()),
                _ => fail_transfer!(),
            },

            State::ContinueCompoundStat(len) => match from {
                Some(State::EnterStat) | Some(State::ExitStat) | Some(State::ExitSkip) => {
                    code_gen.continue_compound_statement(*len)
                }
                _ => fail_transfer!(),
            },
            State::ExitCompoundStat(stack_top) => match from {
                Some(State::EnterStat)
                | Some(State::ExitStat)
                | Some(State::ContinueCompoundStat(..)) => {
                    code_gen.exit_compound_statement(*stack_top)
                }
                _ => fail_transfer!(),
            },
            State::ExitVarDecl => match from {
                Some(State::ExitExpr(register)) => code_gen.exit_variable_declaration(register),
                _ => fail_transfer!(),
            },
            State::ExitIfStatCondition(has_else) => match from {
                Some(State::ExitExpr(condition)) => {
                    code_gen.exit_if_stat_condition(*has_else, condition)
                }
                _ => fail_transfer!(),
            },
            State::ExitIfStatBody(if_label, has_else) => match from {
                Some(State::ExitStat) => code_gen.exit_if_stat_body(*if_label, *has_else),
                _ => fail_transfer!(),
            },
            State::ExitIfStatElseBody(else_label) => match from {
                Some(State::ExitStat) => code_gen.exit_if_stat_else_body(*else_label),
                _ => fail_transfer!(),
            },
            State::ExitIfStat => match from {
                Some(State::ExitIfStatBody(..)) | Some(State::ExitIfStatElseBody(..)) => {
                    code_gen.exit_if_stat()
                }
                _ => fail_transfer!(),
            },
            State::ExitExprStat => match from {
                Some(State::ExitExpr(register)) => code_gen.exit_expression_statement(register),
                _ => fail_transfer!(),
            },

            // expressions
            State::EnterExprAnywhere => match from {
                Some(State::EnterStat)
                | Some(State::ExitStat)
                | Some(State::EnterExprAnywhere)
                | Some(State::EnterExpr(..))
                | Some(State::ContinueBinaryExpr(..))
                | Some(State::ContinueBlockExpr(..)) => code_gen.enter_expression_anywhere(),
                _ => fail_transfer!(),
            },
            State::EnterExpr(dest) => match from {
                Some(State::EnterStat)
                | Some(State::EnterExprAnywhere)
                | Some(State::EnterExpr(..))
                | Some(State::ContinueBinaryExpr(..)) => code_gen.enter_expression(*dest),
                _ => fail_transfer!(),
            },
            State::ExitExpr(..) => match from {
                Some(State::EnterExprAnywhere)
                | Some(State::EnterExpr(..))
                | Some(State::ExitReturnExpr(..))
                | Some(State::ExitUnaryExpr(..))
                | Some(State::ExitBinaryExpr(..))
                | Some(State::ExitBlockExpr(..)) => Ok(()),
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
            State::ContinueBinaryExpr(op, dest) => match from {
                Some(State::ExitExpr(left)) => {
                    code_gen.continue_binary_expression(*op, left, *dest)
                }
                _ => fail_transfer!(),
            },
            State::ExitBinaryExpr(op, left, dest) => match from {
                Some(State::ExitExpr(right)) => {
                    code_gen.exit_binary_expression(*op, *left, right, *dest)
                }
                _ => fail_transfer!(),
            },
            State::ContinueBlockExpr(len) => match from {
                Some(State::ExitStat)
                | Some(State::EnterExprAnywhere)
                | Some(State::EnterExpr(..)) => code_gen.continue_block_expr(*len),
                _ => fail_transfer!(),
            },
            State::ExitBlockExpr(stack_top, dest) => match from {
                Some(State::ExitExpr(tail_expr)) => {
                    code_gen.exit_block_expr(*stack_top, tail_expr, *dest)
                }
                _ => fail_transfer!(),
            },

            // skip
            State::SkipStat => match from {
                Some(State::ExitStat)
                | Some(State::ExitIfStatCondition(..))
                | Some(State::SkipCompoundStat(..)) => code_gen.skip_statement(),
                _ => fail_transfer!(),
            },
            State::SkipCompoundStat(len) => match from {
                Some(State::SkipStat) | Some(State::ExitSkip) => code_gen.skip_compound_stat(*len),
                _ => fail_transfer!(),
            },
            State::SkipExpr => match from {
                Some(State::SkipStat) => code_gen.skip_expr(),
                _ => fail_transfer!(),
            },
            State::ExitSkip => match from {
                Some(State::SkipStat)
                | Some(State::SkipCompoundStat(..))
                | Some(State::SkipExpr)
                | Some(State::ExitSkip) => Ok(()),
                _ => fail_transfer!(),
            },
        }
    }
}

struct CodeGen<'gc, 'ast, I, L> {
    mc: &'ast Mutation<'gc>,
    ast: Ast2Iterator<'ast, PString<'gc>>,

    // TODO: remove allow when needed
    #[allow(dead_code)]
    strings: I,
    get_line_number: L,
    last_line_number: LastLineNumber,

    state: SmallVec<[State<'gc>; 32]>,

    current_function: PrototypeBuilder<'gc>,
}

impl<'gc, 'ast, I, L> CodeGen<'gc, 'ast, I, L>
where
    I: StringInterner<'gc, String = PString<'gc>>,
    L: Fn(Span) -> LineNumber,
{
    fn set_last_line_number<S: Into<LastLineNumber>>(&mut self, line_number: S) {
        self.last_line_number = line_number.into();
    }

    fn last_line_number(&mut self) -> LineNumber {
        match self.last_line_number {
            LastLineNumber::Node(n) => {
                let line_number = self
                    .ast
                    .location_of(n)
                    .map(|span| (self.get_line_number)(span))
                    .unwrap_or(LineNumber::new(1));
                self.set_last_line_number(line_number);
                line_number
            }
            LastLineNumber::LineNumber(l) => l,
        }
    }

    fn push_state(&mut self, state: State<'gc>) {
        self.state.push(state);
    }

    fn pop_state(&mut self) -> Option<State<'gc>> {
        self.state.pop()
    }

    fn allocate(&mut self, dest: ExprDest) -> Result<MaybeTempRegister> {
        Ok(match dest {
            ExprDest::Register(register) => MaybeTempRegister::Protected(register),
            ExprDest::Anywhere => MaybeTempRegister::Temporary(
                self.current_function
                    .registers
                    .allocate_any()
                    .ok_or(CodeGenMessage::NoRegistersAvailable)?,
            ),
        })
    }

    fn free_temp(&mut self, register: impl IntoTempRegister) {
        if let Some(register) = register.into_temp_register() {
            self.current_function.registers.free(register);
        }
    }

    fn flatten_any_expr(&mut self, value: AnyExpr<'gc>) -> Result<RegisterOrConstant16> {
        Ok(match value {
            AnyExpr::Protected(r) => RegisterOrConstant16::Protected(r),
            AnyExpr::Temporary(r) => RegisterOrConstant16::Temporary(r),
            AnyExpr::Constant8(c) => RegisterOrConstant16::Constant8(c),
            AnyExpr::Constant16(c) => RegisterOrConstant16::Constant16(c),
            AnyExpr::Value(v) => {
                let constant = self.push_constant(v)?;
                match constant {
                    CIndex::Constant8(c) => RegisterOrConstant16::Constant8(c),
                    CIndex::Constant16(c) => RegisterOrConstant16::Constant16(c),
                }
            }
        })
    }

    fn try_inline_value<F, R, const N: usize>(
        &mut self,
        values: [AnyExpr<'gc>; N],
        f: F,
        dest: ExprDest,
    ) -> Result<Option<AnyExpr<'gc>>>
    where
        F: FnOnce(&'ast Mutation<'gc>, [Value<'gc>; N]) -> std::result::Result<Value<'gc>, R>,
    {
        let mut mapped: [Value<'gc>; N] = [Value::Null; N];
        for (i, v) in values.into_iter().enumerate() {
            match self.get_value(v) {
                Some(v) => mapped[i] = v,
                _ => return Ok(None),
            }
        }
        Ok(match f(self.mc, mapped) {
            Ok(value) => Some(match dest {
                ExprDest::Register(_) => self.push_constant_to_register(value, dest)?.into(),
                ExprDest::Anywhere => AnyExpr::Value(value),
            }),
            Err(_) => None,
        })
    }

    fn get_value(&mut self, value: AnyExpr<'gc>) -> Option<Value<'gc>> {
        match value {
            AnyExpr::Protected(_) | AnyExpr::Temporary(_) => None,
            AnyExpr::Constant8(c) => self.current_function.constants.get(c as usize).copied(),
            AnyExpr::Constant16(c) => self.current_function.constants.get(c as usize).copied(),
            AnyExpr::Value(v) => Some(v),
        }
    }

    fn flatten_constant(
        &mut self,
        value: AnyExpr<'gc>,
        dest: ExprDest,
    ) -> Result<RegisterOrConstant8> {
        Ok(match value {
            AnyExpr::Protected(r) => RegisterOrConstant8::Protected(r),
            AnyExpr::Temporary(r) => RegisterOrConstant8::Temporary(r),
            AnyExpr::Constant8(c) => RegisterOrConstant8::Constant8(c),
            AnyExpr::Constant16(c) => {
                let register = self.allocate(dest)?;
                self.push_instruction(Instruction::LoadC {
                    destination: register.into(),
                    constant: c,
                });
                register.into()
            }
            AnyExpr::Value(v) => {
                let constant = self.push_constant(v)?;
                match constant {
                    CIndex::Constant8(c) => RegisterOrConstant8::Constant8(c),
                    CIndex::Constant16(c) => {
                        let register = self.allocate(dest)?;
                        self.push_instruction(Instruction::LoadC {
                            destination: register.into(),
                            constant: c,
                        });
                        register.into()
                    }
                }
            }
        })
    }

    fn push_constant_to_register(
        &mut self,
        constant: Value<'gc>,
        dest: ExprDest,
    ) -> Result<MaybeTempRegister> {
        let dest = self.allocate(dest)?;

        match constant {
            Value::Null => {
                self.push_instruction(Instruction::LoadN {
                    destination: dest.into(),
                });
                return Ok(dest);
            }
            Value::Boolean(v) => {
                self.push_instruction(Instruction::LoadB {
                    destination: dest.into(),
                    boolean: v,
                });
                return Ok(dest);
            }
            Value::Integer(v) => {
                if let Ok(v) = v.try_into() {
                    self.push_instruction(Instruction::LoadI {
                        destination: dest.into(),
                        integer: v,
                    });
                    return Ok(dest);
                }
            }
            Value::Float(v) => {
                let v_as_i64 = v as i64;
                if v_as_i64 as f64 == v {
                    if let Ok(v) = v_as_i64.try_into() {
                        self.push_instruction(Instruction::LoadI {
                            destination: dest.into(),
                            integer: v,
                        });
                        return Ok(dest);
                    }
                }
            }
            _ => {}
        };

        let constant = self.push_constant(constant)?;
        self.push_instruction(Instruction::LoadC {
            destination: dest.into(),
            constant: constant.into(),
        });
        Ok(dest)
    }

    fn push_instruction(&mut self, instruction: Instruction) -> Label {
        let label = self.current_function.instructions.len();
        self.current_function.instructions.push(instruction);
        let line_number = self.last_line_number().get();
        self.current_function.line_numbers.push(line_number);
        label as Label
    }

    fn patch_jump(&mut self, jump: Label, dest: Label) -> Result<()> {
        let Some(instr) = self.current_function.instructions.get_mut(jump as usize) else {
            return Err(CodeGenError::InvalidJumpLabel);
        };

        match instr {
            Instruction::CJumpR { jump: jmp, .. } | Instruction::CJumpC { jump: jmp, .. } => {
                const MAX_JUMP: i64 = u16::MAX as i64;
                match (dest as i64).checked_sub(jump as i64 + 1) {
                    Some(jump @ 0..=MAX_JUMP) => {
                        *jmp = jump as u16;
                        Ok(())
                    }
                    Some(jump) if jump < 0 => Err(CodeGenError::NegativeConditionalJump),
                    _ => Err(CodeGenMessage::JumpTooLarge.into()),
                }
            }
            Instruction::Jump { jump: jmp } => {
                const MIN_JUMP: i64 = i16::MIN as i64;
                const MAX_JUMP: i64 = i16::MAX as i64;
                match (dest as i64).checked_sub(jump as i64 + 1) {
                    Some(jump @ MIN_JUMP..=MAX_JUMP) => {
                        *jmp = jump as i16;
                        Ok(())
                    }
                    _ => Err(CodeGenMessage::JumpTooLarge.into()),
                }
            }
            _ => Err(CodeGenError::InvalidJumpLabel),
        }
    }

    /// Returns a label to the next instruction (does not yet exist!)
    fn next_instruction(&self) -> Label {
        self.current_function.instructions.len() as Label
    }

    fn push_constant(&mut self, constant: Value<'gc>) -> Result<CIndex> {
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
                    _ => return Err(CodeGenMessage::ConstantPoolFull.into()),
                };
                self.current_function.constants.push(constant);
                entry.insert(index);
                index
            }
        })
    }

    fn finish(&mut self) -> Result<()> {
        // implicit return
        let constant = self.push_constant(Value::Null)?;
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

    // root

    fn enter_root(&mut self) -> Result<()> {
        match self.ast.next_root()? {
            Root::Statements => {
                self.push_state(State::EnterStat);
            }
        }

        Ok(())
    }

    // statements

    fn enter_statement(&mut self) -> Result<()> {
        let statement = self.ast.next_stat()?;
        self.set_last_line_number(self.ast.previous_node());
        match *statement {
            Stat::Compound { len, .. } => {
                let stack_top = self.current_function.registers.stack_top();
                self.push_state(State::ExitCompoundStat(stack_top));
                self.push_state(State::ContinueCompoundStat(len));

                Ok(())
            }
            Stat::VarDecl { def, .. } => {
                let binding = self
                    .ast
                    .get_binding_at(self.ast.previous_node())
                    .ok_or(CodeGenError::MissingBinding)?;
                let local = binding.index;
                let register = self
                    .current_function
                    .registers
                    .allocate_any()
                    .ok_or(CodeGenMessage::NoRegistersAvailable)?;
                self.current_function
                    .registers
                    .assign_local(local, register);

                match def {
                    true => {
                        self.push_state(State::ExitVarDecl);
                        self.push_state(State::EnterExpr(ExprDest::Register(register)));
                    }
                    false => {
                        let constant = self.push_constant(Value::Null)?;
                        self.push_instruction(Instruction::LoadC {
                            destination: register.into(),
                            constant: constant.into(),
                        });
                        self.exit_variable_declaration(AnyExpr::Protected(register))?;
                    }
                }

                Ok(())
            }
            Stat::If { has_else } => {
                self.push_state(State::ExitIfStatCondition(has_else));
                self.push_state(State::EnterExprAnywhere);

                Ok(())
            }
            Stat::Expr => {
                self.push_state(State::ExitExprStat);
                self.push_state(State::EnterExprAnywhere);

                Ok(())
            }
        }
    }

    fn continue_compound_statement(&mut self, len: RefLen) -> Result<()> {
        if len > 0 {
            self.push_state(State::ContinueCompoundStat(len - 1));
            self.push_state(State::EnterStat);
        }

        Ok(())
    }

    fn exit_compound_statement(&mut self, stack_top: u16) -> Result<()> {
        self.current_function.registers.pop_to(stack_top);
        self.push_state(State::ExitStat);
        Ok(())
    }

    fn exit_variable_declaration(&mut self, register: AnyExpr<'gc>) -> Result<()> {
        assert!(
            matches!(register, AnyExpr::Protected(_)),
            "Register must be protected"
        );
        self.push_state(State::ExitStat);
        Ok(())
    }

    fn exit_if_stat_condition(&mut self, has_else: bool, condition: AnyExpr<'gc>) -> Result<()> {
        // if the condition can be known at compile-time,
        // then the entire if statement can be optimized away
        if let Some(v) = self.get_value(condition) {
            match (v.to_bool(), has_else) {
                (true, true) => {
                    self.push_state(State::SkipStat);
                    self.push_state(State::EnterStat);
                }
                (true, false) => {
                    self.push_state(State::EnterStat);
                }
                (false, true) => {
                    self.push_state(State::EnterStat);
                    self.push_state(State::SkipStat);
                }
                (false, false) => {
                    self.push_state(State::SkipStat);
                }
            }

            return Ok(());
        }

        let dest = self.flatten_constant(condition, ExprDest::Anywhere)?;
        self.free_temp(dest);

        let if_label = self.push_instruction(Instruction::cond_jump(dest, 0));
        self.push_state(State::ExitIfStatBody(if_label, has_else));
        self.push_state(State::EnterStat);
        Ok(())
    }

    fn exit_if_stat_body(&mut self, if_label: Label, has_else: bool) -> Result<()> {
        match has_else {
            true => {
                let else_label = self.push_instruction(Instruction::Jump { jump: 0 });
                self.patch_jump(if_label, else_label + 1)?;
                self.push_state(State::ExitIfStatElseBody(else_label));
                self.push_state(State::EnterStat);
            }
            false => {
                self.patch_jump(if_label, self.next_instruction())?;
                self.push_state(State::ExitIfStat);
            }
        }

        Ok(())
    }

    fn exit_if_stat_else_body(&mut self, else_label: Label) -> Result<()> {
        self.patch_jump(else_label, self.next_instruction())?;
        self.push_state(State::ExitIfStat);

        Ok(())
    }

    fn exit_if_stat(&mut self) -> Result<()> {
        self.push_state(State::ExitStat);

        Ok(())
    }

    fn exit_expression_statement(&mut self, register: AnyExpr<'gc>) -> Result<()> {
        self.free_temp(register);
        self.push_state(State::ExitStat);
        Ok(())
    }

    // expressions

    fn enter_expression_anywhere(&mut self) -> Result<()> {
        let expr = self.ast.next_expr()?;
        let node = self.ast.previous_node();
        self.set_last_line_number(node);
        self.consume_expr_anywhere(node, expr)
    }

    fn consume_expr_anywhere(&mut self, node: NodeRef, expr: &'ast Expr<'gc>) -> Result<()> {
        match *expr {
            Expr::Null => {
                self.push_state(State::ExitExpr(Value::Null.into()));

                Ok(())
            }
            Expr::Bool(v) => {
                self.push_state(State::ExitExpr(Value::Boolean(v).into()));

                Ok(())
            }
            Expr::Integer(v) => {
                self.push_state(State::ExitExpr(Value::Integer(v).into()));

                Ok(())
            }
            Expr::Float(v) => {
                self.push_state(State::ExitExpr(Value::Float(v).into()));

                Ok(())
            }
            Expr::String(v) => {
                self.push_state(State::ExitExpr(Value::String(v).into()));

                Ok(())
            }
            Expr::Var { assignment, .. } => {
                let binding = self
                    .ast
                    .get_binding_at(node)
                    .ok_or(CodeGenError::MissingBinding)?;
                let local = binding.index;
                let local = self
                    .current_function
                    .registers
                    .address_of_local(local)
                    .ok_or(CodeGenError::MissingLocalRegister)?;
                match assignment {
                    true => {
                        self.push_state(State::EnterExpr(ExprDest::Register(local)));
                    }
                    false => {
                        self.push_state(State::ExitExpr(AnyExpr::Protected(local)));
                    }
                }

                Ok(())
            }
            Expr::Return { .. } => {
                // Execution can't continue after this,
                // no need to allocate an actual new register
                let register = Register::default();
                self.consume_expr(node, expr, ExprDest::Register(register))?;

                Ok(())
            }
            _ => {
                self.consume_expr(node, expr, ExprDest::Anywhere)?;

                Ok(())
            }
        }
    }

    fn enter_expression(&mut self, dest: ExprDest) -> Result<()> {
        let expr = self.ast.next_expr()?;
        let node = self.ast.previous_node();
        self.set_last_line_number(node);
        self.consume_expr(node, expr, dest)
    }

    fn consume_expr(&mut self, node: NodeRef, expr: &'ast Expr<'gc>, dest: ExprDest) -> Result<()> {
        match *expr {
            Expr::Null => {
                let dest = self.push_constant_to_register(Value::Null, dest)?;
                self.push_state(State::ExitExpr(dest.into()));

                Ok(())
            }
            Expr::Bool(v) => {
                let dest = self.push_constant_to_register(Value::Boolean(v), dest)?;
                self.push_state(State::ExitExpr(dest.into()));

                Ok(())
            }
            Expr::Integer(v) => {
                let dest = self.push_constant_to_register(Value::Integer(v), dest)?;
                self.push_state(State::ExitExpr(dest.into()));

                Ok(())
            }
            Expr::Float(v) => {
                let dest = self.push_constant_to_register(Value::Float(v), dest)?;
                self.push_state(State::ExitExpr(dest.into()));

                Ok(())
            }
            Expr::String(v) => {
                let dest = self.push_constant_to_register(Value::String(v), dest)?;
                self.push_state(State::ExitExpr(dest.into()));

                Ok(())
            }
            Expr::Var { assignment, .. } => {
                let binding = self
                    .ast
                    .get_binding_at(node)
                    .ok_or(CodeGenError::MissingBinding)?;
                let local = binding.index;
                let local = self
                    .current_function
                    .registers
                    .address_of_local(local)
                    .ok_or(CodeGenError::MissingLocalRegister)?;
                match assignment {
                    true => {
                        self.push_state(State::ExitVariableExpr(local, dest));
                        self.push_state(State::EnterExpr(ExprDest::Register(local)));
                    }
                    false => {
                        self.exit_variable_expression(local, dest)?;
                    }
                }

                Ok(())
            }
            Expr::Return { right } => match right {
                true => {
                    self.push_state(State::ExitReturnExpr(dest));
                    self.push_state(State::EnterExprAnywhere);

                    Ok(())
                }
                false => {
                    self.exit_return_expression(Value::Null.into(), dest)?;

                    Ok(())
                }
            },
            Expr::UnOp { op } => {
                self.push_state(State::ExitUnaryExpr(op, dest));
                self.push_state(State::EnterExprAnywhere);

                Ok(())
            }
            Expr::BinOp { op, len } => {
                self.push_state(State::ContinueBinaryExpr(op, dest));
                for _ in 1..len {
                    self.push_state(State::ContinueBinaryExpr(op, ExprDest::Anywhere));
                }
                self.push_state(State::EnterExprAnywhere);

                Ok(())
            }
            Expr::Block { len, .. } => {
                let stack_top = self.current_function.registers.stack_top();
                self.push_state(State::ExitBlockExpr(stack_top, dest));
                self.push_state(State::EnterExprAnywhere);
                self.push_state(State::ContinueBlockExpr(len));

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

    fn exit_return_expression(&mut self, right: AnyExpr<'gc>, dest: ExprDest) -> Result<()> {
        let right = self.flatten_any_expr(right)?;
        self.free_temp(right);
        self.push_instruction(Instruction::ret(right));
        let dest = self.allocate(dest)?;
        self.push_state(State::ExitExpr(dest.into()));

        Ok(())
    }

    fn exit_unary_expression(
        &mut self,
        op: UnOp,
        right: AnyExpr<'gc>,
        dest: ExprDest,
    ) -> Result<()> {
        if let Some(dest) = self.try_inline_value(
            [right],
            |_, [right]| match op {
                UnOp::Neg => right.neg(),
                UnOp::Not => right.not(),
            },
            dest,
        )? {
            self.push_state(State::ExitExpr(dest));
            return Ok(());
        }

        let right = self.flatten_any_expr(right)?;
        self.free_temp(right);
        let dest = self.allocate(dest)?;
        let instruction = match op {
            UnOp::Neg => Instruction::neg(dest.into(), right),
            UnOp::Not => Instruction::not(dest.into(), right),
        };
        self.push_instruction(instruction);
        self.push_state(State::ExitExpr(dest.into()));

        Ok(())
    }

    fn continue_binary_expression(
        &mut self,
        op: BinOp,
        left: AnyExpr<'gc>,
        dest: ExprDest,
    ) -> Result<()> {
        self.push_state(State::ExitBinaryExpr(op, left, dest));
        self.push_state(State::EnterExprAnywhere);

        Ok(())
    }

    fn exit_binary_expression(
        &mut self,
        op: BinOp,
        left: AnyExpr<'gc>,
        right: AnyExpr<'gc>,
        dest: ExprDest,
    ) -> Result<()> {
        if let Some(dest) = self.try_inline_value(
            [left, right],
            |mc, [a, b]| match op {
                BinOp::Eq => Ok(a.eq(&b).into()),
                BinOp::NotEq => Ok((!a.eq(&b)).into()),
                BinOp::Add => a.add(mc, b),
                BinOp::Sub => a - b,
                BinOp::Mul => a * b,
                BinOp::Div => a / b,
            },
            dest,
        )? {
            self.push_state(State::ExitExpr(dest));
            return Ok(());
        }

        let left = self.flatten_constant(left, ExprDest::Anywhere)?;
        let right = self.flatten_constant(right, ExprDest::Anywhere)?;
        self.free_temp(left);
        self.free_temp(right);
        let dest = self.allocate(dest)?;
        let instruction = match op {
            BinOp::Eq => Instruction::eq(dest.into(), left, right),
            BinOp::NotEq => Instruction::neq(dest.into(), left, right),
            BinOp::Add => Instruction::add(dest.into(), left, right),
            BinOp::Sub => Instruction::sub(dest.into(), left, right),
            BinOp::Mul => Instruction::mul(dest.into(), left, right),
            BinOp::Div => Instruction::div(dest.into(), left, right),
        };
        self.push_instruction(instruction);
        self.push_state(State::ExitExpr(dest.into()));

        Ok(())
    }

    fn continue_block_expr(&mut self, len: RefLen) -> Result<()> {
        if len > 0 {
            self.push_state(State::ContinueBlockExpr(len - 1));
            self.push_state(State::EnterStat);
        }

        Ok(())
    }

    fn exit_block_expr(
        &mut self,
        stack_top: u16,
        expr: AnyExpr<'gc>,
        dest: ExprDest,
    ) -> Result<()> {
        let expr = self.flatten_any_expr(expr)?;
        self.current_function.registers.pop_to(stack_top);

        let dest = self.allocate(dest)?;
        match expr {
            RegisterOrConstant16::Protected(tail_r) | RegisterOrConstant16::Temporary(tail_r) => {
                let dest_register: Register = dest.into();
                if dest_register != tail_r {
                    self.push_instruction(Instruction::LoadR {
                        destination: dest.into(),
                        from: tail_r.into(),
                    });
                }
            }
            _ => {
                self.push_instruction(Instruction::load(dest.into(), expr));
            }
        }

        self.push_state(State::ExitExpr(dest.into()));

        Ok(())
    }

    // skip

    fn skip_statement(&mut self) -> Result<()> {
        self.push_state(State::ExitSkip);

        let statement = self.ast.next_stat()?;
        match *statement {
            Stat::Compound { len, .. } => {
                self.push_state(State::SkipCompoundStat(len));
            }
            Stat::VarDecl { def, .. } => {
                if def {
                    self.push_state(State::SkipExpr);
                }
            }
            Stat::If { has_else } => {
                if has_else {
                    self.push_state(State::SkipStat);
                }
                self.push_state(State::SkipStat);
                self.push_state(State::SkipExpr);
            }
            Stat::Expr => {
                self.push_state(State::SkipExpr);
            }
        }

        Ok(())
    }

    fn skip_compound_stat(&mut self, len: RefLen) -> Result<()> {
        if len > 0 {
            self.push_state(State::SkipCompoundStat(len - 1));
            self.push_state(State::SkipStat);
        }

        Ok(())
    }

    fn skip_expr(&mut self) -> Result<()> {
        self.push_state(State::ExitSkip);

        let expr = self.ast.next_expr()?;
        match *expr {
            Expr::Null => {}
            Expr::Bool(_) => {}
            Expr::Integer(_) => {}
            Expr::Float(_) => {}
            Expr::String(_) => {}
            Expr::Var { assignment, .. } => {
                if assignment {
                    self.push_state(State::SkipExpr);
                }
            }
            Expr::Return { .. } => {
                self.push_state(State::SkipExpr);
            }
            Expr::UnOp { .. } => {
                self.push_state(State::SkipExpr);
            }
            Expr::BinOp { .. } => {
                self.push_state(State::SkipExpr);
                self.push_state(State::SkipExpr);
            }
            Expr::Block { len, .. } => {
                self.push_state(State::SkipExpr);
                self.push_state(State::SkipCompoundStat(len));
            }
        }

        Ok(())
    }
}

trait IntoTempRegister {
    fn into_temp_register(self) -> Option<Register>;
}

#[derive(Copy, Clone, Debug)]
enum AnyExpr<'gc> {
    Protected(Register),
    Temporary(Register),
    Constant8(CIndex8),
    Constant16(CIndex16),
    Value(Value<'gc>),
}

impl IntoTempRegister for AnyExpr<'_> {
    fn into_temp_register(self) -> Option<Register> {
        match self {
            AnyExpr::Temporary(r) => Some(r),
            _ => None,
        }
    }
}

impl From<RegisterOrConstant16> for AnyExpr<'_> {
    fn from(value: RegisterOrConstant16) -> Self {
        match value {
            RegisterOrConstant16::Protected(r) => AnyExpr::Protected(r),
            RegisterOrConstant16::Temporary(r) => AnyExpr::Temporary(r),
            RegisterOrConstant16::Constant8(c) => AnyExpr::Constant8(c),
            RegisterOrConstant16::Constant16(c) => AnyExpr::Constant16(c),
        }
    }
}

impl<'gc> From<Value<'gc>> for AnyExpr<'gc> {
    fn from(value: Value<'gc>) -> Self {
        Self::Value(value)
    }
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

impl From<MaybeTempRegister> for AnyExpr<'_> {
    fn from(value: MaybeTempRegister) -> Self {
        match value {
            MaybeTempRegister::Protected(r) => AnyExpr::Protected(r),
            MaybeTempRegister::Temporary(r) => AnyExpr::Temporary(r),
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

    fn load(dest: Register, from: RegisterOrConstant16) -> Instruction {
        match from {
            RegisterOrConstant16::Protected(r) | RegisterOrConstant16::Temporary(r) => {
                Instruction::LoadR {
                    destination: dest.into(),
                    from: r.into(),
                }
            }
            RegisterOrConstant16::Constant8(c) => Instruction::LoadC {
                destination: dest.into(),
                constant: c.into(),
            },
            RegisterOrConstant16::Constant16(c) => Instruction::LoadC {
                destination: dest.into(),
                constant: c,
            },
        }
    }

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

    fn not(dest: Register, right: RegisterOrConstant16) -> Instruction {
        match right {
            RegisterOrConstant16::Protected(r) | RegisterOrConstant16::Temporary(r) => {
                Instruction::NotR {
                    destination: dest.into(),
                    right: r.into(),
                }
            }
            RegisterOrConstant16::Constant8(c) => Instruction::NotC {
                destination: dest.into(),
                right: c.into(),
            },
            RegisterOrConstant16::Constant16(c) => Instruction::NotC {
                destination: dest.into(),
                right: c,
            },
        }
    }

    fn eq(dest: Register, left: RegisterOrConstant8, right: RegisterOrConstant8) -> Instruction {
        match (left, right) {
            (
                RegisterOrConstant8::Protected(l) | RegisterOrConstant8::Temporary(l),
                RegisterOrConstant8::Protected(r) | RegisterOrConstant8::Temporary(r),
            ) => Instruction::EqRR {
                destination: dest.into(),
                left: l.into(),
                right: r.into(),
            },
            (
                RegisterOrConstant8::Protected(l) | RegisterOrConstant8::Temporary(l),
                RegisterOrConstant8::Constant8(r),
            ) => Instruction::EqRC {
                destination: dest.into(),
                left: l.into(),
                right: r,
            },
            (
                RegisterOrConstant8::Constant8(l),
                RegisterOrConstant8::Protected(r) | RegisterOrConstant8::Temporary(r),
            ) => Instruction::EqCR {
                destination: dest.into(),
                left: l,
                right: r.into(),
            },
            (RegisterOrConstant8::Constant8(l), RegisterOrConstant8::Constant8(r)) => {
                Instruction::EqCC {
                    destination: dest.into(),
                    left: l,
                    right: r,
                }
            }
        }
    }

    fn neq(dest: Register, left: RegisterOrConstant8, right: RegisterOrConstant8) -> Instruction {
        match (left, right) {
            (
                RegisterOrConstant8::Protected(l) | RegisterOrConstant8::Temporary(l),
                RegisterOrConstant8::Protected(r) | RegisterOrConstant8::Temporary(r),
            ) => Instruction::NeqRR {
                destination: dest.into(),
                left: l.into(),
                right: r.into(),
            },
            (
                RegisterOrConstant8::Protected(l) | RegisterOrConstant8::Temporary(l),
                RegisterOrConstant8::Constant8(r),
            ) => Instruction::NeqRC {
                destination: dest.into(),
                left: l.into(),
                right: r,
            },
            (
                RegisterOrConstant8::Constant8(l),
                RegisterOrConstant8::Protected(r) | RegisterOrConstant8::Temporary(r),
            ) => Instruction::NeqCR {
                destination: dest.into(),
                left: l,
                right: r.into(),
            },
            (RegisterOrConstant8::Constant8(l), RegisterOrConstant8::Constant8(r)) => {
                Instruction::NeqCC {
                    destination: dest.into(),
                    left: l,
                    right: r,
                }
            }
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

    fn cond_jump(condition: RegisterOrConstant8, jump: u16) -> Instruction {
        match condition {
            RegisterOrConstant8::Protected(r) | RegisterOrConstant8::Temporary(r) => {
                Instruction::CJumpR {
                    register: r.into(),
                    jump,
                }
            }
            RegisterOrConstant8::Constant8(c) => Instruction::CJumpC { constant: c, jump },
        }
    }
}

type Label = u32;
