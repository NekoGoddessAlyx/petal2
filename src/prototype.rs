use crate::{PString, Value};

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
    LoadConstant {
        register: Register,
        constant: ConstantIndex,
    },
}

static_assert_size!(Instruction, 4);
