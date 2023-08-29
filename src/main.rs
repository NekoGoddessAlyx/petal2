use std::fs::read_to_string;

use petal2::code_gen::code_gen;
use petal2::{compile, CompilerMessage};

fn callback(message: CompilerMessage) {
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

    println!("{}", ast);

    let code_gen_result = code_gen(ast);

    let _prototype = match code_gen_result {
        Ok(prototype) => {
            println!("Prototype: {:#?}", prototype);
            prototype
        }
        Err(error) => {
            println!("Compilation failed: {:?}", error);
            return;
        }
    };

    // let result = interpret(ast);
    // println!("Result: {}", result);
}
