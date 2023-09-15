use std::fmt::{Display, Formatter};

use gc_arena::Collect;

pub type RIndex = u8;
pub type CIndex8 = u8;
pub type CIndex16 = u16;

#[derive(Copy, Clone, Debug, Collect)]
#[collect(require_static)]
pub enum Instruction {
    ReturnR {
        register: RIndex,
    },
    ReturnC {
        constant: CIndex16,
    },

    LoadR {
        destination: RIndex,
        from: RIndex,
    },
    LoadC {
        destination: RIndex,
        constant: CIndex16,
    },
    LoadI {
        destination: RIndex,
        integer: i16,
    },

    NegR {
        destination: RIndex,
        right: RIndex,
    },
    NegC {
        destination: RIndex,
        right: CIndex16,
    },

    NotR {
        destination: RIndex,
        right: RIndex,
    },
    NotC {
        destination: RIndex,
        right: CIndex16,
    },

    AddRR {
        destination: RIndex,
        left: RIndex,
        right: RIndex,
    },
    AddRC {
        destination: RIndex,
        left: RIndex,
        right: CIndex8,
    },
    AddCC {
        destination: RIndex,
        left: CIndex8,
        right: CIndex8,
    },
    AddCR {
        destination: RIndex,
        left: CIndex8,
        right: RIndex,
    },

    SubRR {
        destination: RIndex,
        left: RIndex,
        right: RIndex,
    },
    SubRC {
        destination: RIndex,
        left: RIndex,
        right: CIndex8,
    },
    SubCC {
        destination: RIndex,
        left: CIndex8,
        right: CIndex8,
    },
    SubCR {
        destination: RIndex,
        left: CIndex8,
        right: RIndex,
    },

    MulRR {
        destination: RIndex,
        left: RIndex,
        right: RIndex,
    },
    MulRC {
        destination: RIndex,
        left: RIndex,
        right: CIndex8,
    },
    MulCC {
        destination: RIndex,
        left: CIndex8,
        right: CIndex8,
    },
    MulCR {
        destination: RIndex,
        left: CIndex8,
        right: RIndex,
    },

    DivRR {
        destination: RIndex,
        left: RIndex,
        right: RIndex,
    },
    DivRC {
        destination: RIndex,
        left: RIndex,
        right: CIndex8,
    },
    DivCC {
        destination: RIndex,
        left: CIndex8,
        right: CIndex8,
    },
    DivCR {
        destination: RIndex,
        left: CIndex8,
        right: RIndex,
    },
}

static_assert_size!(Instruction, 4);

impl Instruction {
    pub const fn name(self) -> &'static str {
        match self {
            Instruction::ReturnR { .. } => "RETURN_R",
            Instruction::ReturnC { .. } => "RETURN_C",
            Instruction::LoadR { .. } => "LOAD_R",
            Instruction::LoadC { .. } => "LOAD_C",
            Instruction::LoadI { .. } => "LOAD_I",
            Instruction::NegR { .. } => "NEG_R",
            Instruction::NegC { .. } => "NEG_C",
            Instruction::NotR { .. } => "NOT_R",
            Instruction::NotC { .. } => "NOT_C",
            Instruction::AddRR { .. } => "ADD_RR",
            Instruction::AddRC { .. } => "ADD_RC",
            Instruction::AddCC { .. } => "ADD_CC",
            Instruction::AddCR { .. } => "ADD_CR",
            Instruction::SubRR { .. } => "SUB_RR",
            Instruction::SubRC { .. } => "SUB_RC",
            Instruction::SubCC { .. } => "SUB_CC",
            Instruction::SubCR { .. } => "SUB_CR",
            Instruction::MulRR { .. } => "MUL_RR",
            Instruction::MulRC { .. } => "MUL_RC",
            Instruction::MulCC { .. } => "MUL_CC",
            Instruction::MulCR { .. } => "MUL_CR",
            Instruction::DivRR { .. } => "DIV_RR",
            Instruction::DivRC { .. } => "DIV_RC",
            Instruction::DivCC { .. } => "DIV_CC",
            Instruction::DivCR { .. } => "DIV_CR",
        }
    }
}

impl Display for Instruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:<8} ", self.name())?;
        match self {
            Instruction::ReturnR { register } => {
                write!(f, "{:4}          ", register)
            }

            Instruction::ReturnC { constant } => {
                write!(f, "{:4}          ", constant)
            }

            Instruction::LoadR {
                destination,
                from: right,
            }
            | Instruction::NegR { destination, right }
            | Instruction::NotR { destination, right } => {
                write!(f, "{:4} {:4}     ", destination, right)
            }

            Instruction::LoadC {
                destination,
                constant: right,
            }
            | Instruction::NegC { destination, right }
            | Instruction::NotC { destination, right } => {
                write!(f, "{:4} {:4}     ", destination, right)
            }

            Instruction::LoadI {
                destination,
                integer,
            } => {
                write!(f, "{:4} {:9}", destination, integer)
            }

            Instruction::AddRR {
                destination,
                left,
                right,
            }
            | Instruction::SubRR {
                destination,
                left,
                right,
            }
            | Instruction::MulRR {
                destination,
                left,
                right,
            }
            | Instruction::DivRR {
                destination,
                left,
                right,
            } => {
                write!(f, "{:4} {:4} {:4}", destination, left, right)
            }

            Instruction::AddRC {
                destination,
                left,
                right,
            }
            | Instruction::SubRC {
                destination,
                left,
                right,
            }
            | Instruction::MulRC {
                destination,
                left,
                right,
            }
            | Instruction::DivRC {
                destination,
                left,
                right,
            } => {
                write!(f, "{:4} {:4} {:4}", destination, left, right)
            }

            Instruction::AddCC {
                destination,
                left,
                right,
            }
            | Instruction::SubCC {
                destination,
                left,
                right,
            }
            | Instruction::MulCC {
                destination,
                left,
                right,
            }
            | Instruction::DivCC {
                destination,
                left,
                right,
            } => {
                write!(f, "{:4} {:4} {:4}", destination, left, right)
            }

            Instruction::AddCR {
                destination,
                left,
                right,
            }
            | Instruction::SubCR {
                destination,
                left,
                right,
            }
            | Instruction::MulCR {
                destination,
                left,
                right,
            }
            | Instruction::DivCR {
                destination,
                left,
                right,
            } => {
                write!(f, "{:4} {:4} {:4}", destination, left, right)
            }
        }
    }
}
