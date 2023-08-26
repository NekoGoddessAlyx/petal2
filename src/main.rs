use std::fs::read_to_string;

use crate::compiler::compile;
use crate::interpreter::interpret;

mod interpreter;
mod compiler;

fn callback(message: &str) {
    println!("{}", message);
}

fn main() {
    let source = read_to_string("scripts/test.pt").expect("Could not read test script");

    let parse_result = compile(callback, source);
    let ast = match parse_result {
        Ok(ast) => {
            println!("Ast: {:?}", ast);
            ast
        }
        Err(error) => {
            println!("Parse failed: {:?}", error);
            return;
        }
    };

    let result = interpret(ast);
    println!("Result: {}", result);
}
