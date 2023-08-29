use crate::prototype::{ConstantIndex, Instruction, Prototype, Register};
use crate::value::Value;

pub enum InterpretResult {}

pub fn interpret(function: Prototype) -> Result<Value, InterpretResult> {
    let instructions = function.instructions.as_ref();
    let constants = function.constants.as_ref();

    // Ensure instructions is not empty and ends with a return
    // or else bad things might happen.
    // There is currently no checking that register accesses are valid
    // OR that constant accesses are valid ¯\_(ツ)_/¯
    // Could do it.
    // Could.
    assert!(instructions
        .last()
        .is_some_and(|i| matches!(i, Instruction::Return { .. })));

    let mut stack = Vec::with_capacity(256);
    let mut pc: usize = 0;
    stack.resize(function.stack_size as usize + 1, Value::Integer(0));

    // 🦶🔫
    let stack = &mut stack as *mut Vec<Value>;
    let peek = |register: Register| unsafe { *(&*stack).get_unchecked(register as usize) };
    let mov = |register: Register, value: Value| unsafe {
        *(&mut *stack).get_unchecked_mut(register as usize) = value;
    };
    let get_constant =
        |constant: ConstantIndex| unsafe { *constants.get_unchecked(constant as usize) };

    unsafe {
        loop {
            let instruction = *instructions.get_unchecked(pc);
            pc += 1;
            match instruction {
                Instruction::Return { register } => {
                    return Ok(peek(register));
                }
                Instruction::LoadConstant {
                    destination,
                    constant,
                } => mov(destination, get_constant(constant)),
                Instruction::Neg { destination, right } => mov(destination, -peek(right)),
                Instruction::Add {
                    destination,
                    left,
                    right,
                } => mov(destination, peek(left) + peek(right)),
                Instruction::Sub {
                    destination,
                    left,
                    right,
                } => mov(destination, peek(left) - peek(right)),
                Instruction::Mul {
                    destination,
                    left,
                    right,
                } => mov(destination, peek(left) * peek(right)),
                Instruction::Div {
                    destination,
                    left,
                    right,
                } => mov(destination, peek(left) / peek(right)),
            };
        }
    }
}
