use crate::instruction::Instruction;
use crate::value::Value;
use crate::PString;

#[derive(Debug)]
pub struct Prototype<'gc> {
    pub name: PString<'gc>,
    pub stack_size: u8,
    pub instructions: Box<[Instruction]>,
    pub constants: Box<[Value]>,
}
