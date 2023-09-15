use std::fmt::{Display, Formatter};

use crate::instruction::Instruction;
use crate::value::Value;
use crate::PString;

#[derive(Debug)]
pub struct Prototype<'gc> {
    pub name: PString<'gc>,
    pub stack_size: u8,
    pub instructions: Box<[Instruction]>,
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
                        Some(constant) => write!(f, " [{:?}]", constant),
                        None => write!(f, " <invalid>"),
                    }
                }

                let constants = self.constants.as_ref();
                let iter = self.instructions.iter().copied().enumerate();
                for (index, instruction) in iter {
                    write!(f, "{:4?}", index)?;
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
                        Instruction::LoadI { .. } => Ok(()),
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
