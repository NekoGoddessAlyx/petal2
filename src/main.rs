use std::fs::read_to_string;

use gc_arena::rootless_arena;

use petal2::{compile, interpret, CompilerMessage, PString};

fn callback(message: CompilerMessage<PString>) {
    println!("{}", message);
}

fn main() {
    let source = read_to_string("scripts/test.pt").expect("Could not read test script");

    rootless_arena(|mc| {
        let compile_result = compile(mc, callback, source);
        let function = match compile_result {
            Ok(prototype) => {
                println!("Prototype: {}", prototype);
                prototype
            }
            Err(_) => {
                println!("Compilation failed");
                return;
            }
        };

        match interpret(mc, function) {
            Ok(result) => {
                println!("Result: {:?}", result);
            }
            Err(error) => {
                println!("Error occurred while interpreting: {:?}", error);
            }
        }
    })
}
