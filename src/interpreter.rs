use crate::prototype::{Instruction, Prototype};
use crate::value::Value;

pub enum InterpretResult {}

pub fn interpret(function: Prototype) -> Result<Value, InterpretResult> {
    let instructions = function.instructions.as_ref();
    #[allow(unused)]
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

    macro_rules! peek {
        ($r: expr) => {
            *stack.get_unchecked($r as usize)
        };
    }
    macro_rules! mov {
        ($r: expr, $v: expr) => {
            *stack.get_unchecked_mut($r as usize) = $v
        };
    }
    macro_rules! constant {
        ($c: expr) => {
            *constants.get_unchecked($c as usize)
        };
    }

    unsafe {
        loop {
            let instruction = *instructions.get_unchecked(pc);
            pc += 1;
            match instruction {
                Instruction::Return { register } => {
                    return Ok(peek!(register));
                }
                Instruction::Move { destination, from } => mov!(destination, peek!(from)),
                Instruction::LoadConstant {
                    destination,
                    constant,
                } => mov!(destination, constant!(constant)),
                Instruction::Neg { destination, right } => mov!(destination, -peek!(right)),
                Instruction::Add {
                    destination,
                    left,
                    right,
                } => mov!(destination, peek!(left) + peek!(right)),
                Instruction::Sub {
                    destination,
                    left,
                    right,
                } => mov!(destination, peek!(left) - peek!(right)),
                Instruction::Mul {
                    destination,
                    left,
                    right,
                } => mov!(destination, peek!(left) * peek!(right)),
                Instruction::Div {
                    destination,
                    left,
                    right,
                } => mov!(destination, peek!(left) / peek!(right)),
            };
        }
    }
}
