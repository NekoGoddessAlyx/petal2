use gc_arena::Mutation;
use thiserror::Error;

use crate::instruction::Instruction;
use crate::prototype::Prototype;
use crate::value::{TypeError, Value};

#[derive(Debug, Error)]
pub enum InterpretResult {
    #[error(transparent)]
    TypeError(#[from] TypeError),
}

pub fn interpret<'gc>(
    mc: &Mutation<'gc>,
    function: Prototype<'gc>,
) -> Result<Value<'gc>, InterpretResult> {
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
        .is_some_and(|i| matches!(i, Instruction::ReturnR { .. } | Instruction::ReturnC { .. })));

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

    // SAFETY:
    // aside from jumps (which are not yet implemented),
    // the validity of instructions has already been checked above
    // That is: instructions len is non-zero AND it ends in a return statement
    // (thus execution cannot continue past the end of the instructions)
    unsafe {
        loop {
            let instruction = *instructions.get_unchecked(pc);
            pc += 1;
            match instruction {
                Instruction::ReturnR { register } => return Ok(peek!(register)),
                Instruction::ReturnC { constant } => return Ok(constant!(constant)),

                Instruction::LoadR { destination, from } => mov!(destination, peek!(from)),
                Instruction::LoadC {
                    destination,
                    constant,
                } => mov!(destination, constant!(constant)),

                Instruction::LoadI {
                    destination,
                    integer,
                } => mov!(destination, Value::Integer(integer as i64)),

                Instruction::NegR { destination, right } => mov!(destination, (-peek!(right))?),
                Instruction::NegC { destination, right } => mov!(destination, (-constant!(right))?),

                Instruction::NotR { destination, right } => mov!(destination, (!peek!(right))?),
                Instruction::NotC { destination, right } => mov!(destination, (!constant!(right))?),

                Instruction::AddRR {
                    destination,
                    left,
                    right,
                } => mov!(destination, peek!(left).add(mc, peek!(right))?),
                Instruction::AddRC {
                    destination,
                    left,
                    right,
                } => mov!(destination, peek!(left).add(mc, constant!(right))?),
                Instruction::AddCR {
                    destination,
                    left,
                    right,
                } => mov!(destination, constant!(left).add(mc, peek!(right))?),
                Instruction::AddCC {
                    destination,
                    left,
                    right,
                } => mov!(destination, constant!(left).add(mc, constant!(right))?),

                Instruction::SubRR {
                    destination,
                    left,
                    right,
                } => mov!(destination, (peek!(left) - peek!(right))?),
                Instruction::SubRC {
                    destination,
                    left,
                    right,
                } => mov!(destination, (peek!(left) - constant!(right))?),
                Instruction::SubCR {
                    destination,
                    left,
                    right,
                } => mov!(destination, (constant!(left) - peek!(right))?),
                Instruction::SubCC {
                    destination,
                    left,
                    right,
                } => mov!(destination, (constant!(left) - constant!(right))?),

                Instruction::MulRR {
                    destination,
                    left,
                    right,
                } => mov!(destination, (peek!(left) * peek!(right))?),
                Instruction::MulRC {
                    destination,
                    left,
                    right,
                } => mov!(destination, (peek!(left) * constant!(right))?),
                Instruction::MulCR {
                    destination,
                    left,
                    right,
                } => mov!(destination, (constant!(left) * peek!(right))?),
                Instruction::MulCC {
                    destination,
                    left,
                    right,
                } => mov!(destination, (constant!(left) * constant!(right))?),

                Instruction::DivRR {
                    destination,
                    left,
                    right,
                } => mov!(destination, (peek!(left) / peek!(right))?),
                Instruction::DivRC {
                    destination,
                    left,
                    right,
                } => mov!(destination, (peek!(left) / constant!(right))?),
                Instruction::DivCR {
                    destination,
                    left,
                    right,
                } => mov!(destination, (constant!(left) / peek!(right))?),
                Instruction::DivCC {
                    destination,
                    left,
                    right,
                } => mov!(destination, (constant!(left) / constant!(right))?),
            };
        }
    }
}
