use crate::value::Value;
use crate::PString;

#[derive(Debug)]
pub struct Prototype {
    pub name: PString,
    pub stack_size: u8,
    pub instructions: Box<[Instruction]>,
    pub constants: Box<[Value]>,
}

pub type Register = u8;
pub type ConstantIndex = u16;

#[derive(Copy, Clone, Debug)]
pub enum Instruction {
    Return {
        register: Register,
    },
    Move {
        destination: Register,
        from: Register,
    },
    LoadConstant {
        destination: Register,
        constant: ConstantIndex,
    },
    Neg {
        destination: Register,
        right: Register,
    },
    Add {
        destination: Register,
        left: Register,
        right: Register,
    },
    Sub {
        destination: Register,
        left: Register,
        right: Register,
    },
    Mul {
        destination: Register,
        left: Register,
        right: Register,
    },
    Div {
        destination: Register,
        left: Register,
        right: Register,
    },
}

static_assert_size!(Instruction, 4);
