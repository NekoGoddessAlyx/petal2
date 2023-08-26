use std::fs::read_to_string;

use crate::interpreter::interpret;
use crate::lexer::lex;
use crate::parser::parse;

mod lexer;
mod parser;
mod interpreter;

fn callback(message: &'static str) {
    println!("{}", message);
}

fn main() {
    let source = read_to_string("scripts/test.pt").expect("Could not read test script");

    let (tokens, sources) = lex(callback, source);
    println!("Tokens: {:?}", tokens);
    println!("Sources: {:?}", sources);

    let parse_result = parse(callback, &tokens);
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
