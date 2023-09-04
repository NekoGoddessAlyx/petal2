use crate::value::Value;
use crate::PString;

#[derive(Debug)]
pub struct Prototype {
    pub name: PString,
    pub stack_size: u8,
    pub instructions: Box<[Instruction]>,
    pub constants: Box<[Value]>,
}

pub type RIndex = u8;
pub type CIndex8 = u8;
pub type CIndex16 = u16;

#[derive(Copy, Clone, Debug)]
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

    NegR {
        destination: RIndex,
        right: RIndex,
    },
    NegC {
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
