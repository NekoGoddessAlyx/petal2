use std::fmt::{Display, Formatter};
use std::iter::zip;

use gc_arena::Collect;

use crate::instruction::Instruction;
use crate::value::Value;
use crate::{NumDigits, PString};

#[derive(Debug, Collect)]
#[collect(no_drop)]
pub struct Prototype<'gc> {
    pub name: PString<'gc>,
    pub stack_size: u8,
    pub instructions: Box<[Instruction]>,
    pub line_numbers: Box<[u32]>,
    pub constants: Box<[Value<'gc>]>,
}

impl Display for Prototype<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "fun {}, {} params, {} slots, {} instructions",
            self.name,
            0,
            self.stack_size + 1,
            self.instructions.len()
        )?;

        match self.instructions.is_empty() {
            true => {
                writeln!(f, "Function has no instructions.")?;
            }
            false => {
                #[inline]
                fn fmt_c(
                    f: &mut Formatter<'_>,
                    constants: &[Value],
                    c: impl Into<usize>,
                ) -> std::fmt::Result {
                    match constants.get(c.into()) {
                        None => write!(f, " <invalid>"),
                        Some(constant) => write!(f, " [{:?}]", constant),
                    }
                }

                #[inline]
                fn fmt_jump(
                    f: &mut Formatter<'_>,
                    i: usize,
                    j: impl Into<i64>,
                ) -> std::fmt::Result {
                    match i
                        .checked_add_signed(j.into() as isize)
                        .and_then(|dest| dest.checked_add(1))
                    {
                        None => write!(f, " <invalid>"),
                        Some(dest) => write!(f, " ({} -> {})", i, dest),
                    }
                }

                let line_num_width = self.line_numbers.iter().max().map_or(1, |l| l.num_digits());
                let mut last_line_num = u32::MAX;

                let constants = self.constants.as_ref();
                let iter = zip(
                    self.line_numbers.iter().copied(),
                    self.instructions.iter().copied(),
                )
                .enumerate();
                for (index, (line_num, instruction)) in iter {
                    write!(f, "{:4?}", index)?;

                    match line_num == last_line_num {
                        true => {
                            let line_num_width = line_num_width + 7;
                            write!(f, " {:>line_num_width$}", "|")?;
                        }
                        false => {
                            write!(f, " [line {:>line_num_width$}]", line_num)?;
                        }
                    }
                    last_line_num = line_num;

                    write!(f, " {}", instruction)?;

                    match instruction {
                        Instruction::ReturnR { .. } => Ok(()),
                        Instruction::ReturnC { constant } => fmt_c(f, constants, constant),
                        Instruction::LoadR { .. }
                        | Instruction::NegR { .. }
                        | Instruction::NotR { .. } => Ok(()),
                        Instruction::LoadC {
                            constant: right, ..
                        }
                        | Instruction::NegC { right, .. }
                        | Instruction::NotC { right, .. } => fmt_c(f, constants, right),
                        Instruction::LoadN { .. }
                        | Instruction::LoadB { .. }
                        | Instruction::LoadI { .. } => Ok(()),
                        Instruction::AddRR { .. }
                        | Instruction::SubRR { .. }
                        | Instruction::MulRR { .. }
                        | Instruction::DivRR { .. } => Ok(()),
                        Instruction::AddRC { right, .. }
                        | Instruction::SubRC { right, .. }
                        | Instruction::MulRC { right, .. }
                        | Instruction::DivRC { right, .. } => fmt_c(f, constants, right),
                        Instruction::AddCC { left, right, .. }
                        | Instruction::SubCC { left, right, .. }
                        | Instruction::MulCC { left, right, .. }
                        | Instruction::DivCC { left, right, .. } => {
                            fmt_c(f, constants, right)?;
                            fmt_c(f, constants, left)
                        }
                        Instruction::AddCR { left, .. }
                        | Instruction::SubCR { left, .. }
                        | Instruction::MulCR { left, .. }
                        | Instruction::DivCR { left, .. } => fmt_c(f, constants, left),
                        Instruction::CJumpR { jump, .. } => fmt_jump(f, index, jump),
                        Instruction::CJumpC { constant, jump } => {
                            fmt_c(f, constants, constant)?;
                            fmt_jump(f, index, jump)
                        }
                        Instruction::Jump { jump } => fmt_jump(f, index, jump),
                    }?;

                    writeln!(f)?;
                }
            }
        };

        if !self.constants.is_empty() {
            writeln!(f, "constants ({})", self.constants.len())?;
            for (i, c) in self.constants.iter().enumerate() {
                writeln!(f, "{:4} {:?}", i, c)?;
            }
        }

        Ok(())
    }
}
