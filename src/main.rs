use std::fs::read_to_string;

use petal2::{compile, interpret, CompilerMessage};

fn callback(message: CompilerMessage) {
    println!("{}", message);
}

fn main() {
    let source = read_to_string("scripts/test.pt").expect("Could not read test script");

    let compile_result = compile(callback, source);
    let function = match compile_result {
        Ok(prototype) => {
            println!("Prototype: {:#?}", prototype);
            prototype
        }
        Err(_) => {
            println!("Compilation failed");
            return;
        }
    };

    match interpret(function) {
        Ok(result) => {
            println!("Result: {:?}", result);
        }
        Err(_error) => {
            println!("Error occurred while interpreting");
        }
    }
}
