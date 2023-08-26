use crate::lexer::lex;
use crate::parser::parse;

mod lexer;
mod parser;

fn callback(message: &'static str) {
    println!("{}", message);
}

fn main() {
    let source = "1 + 2 * 3\n * 4";

    let (tokens, sources) = lex(callback, source);
    println!("Tokens: {:?}", tokens);
    println!("Sources: {:?}", sources);

    let parse_result = parse(callback, &tokens);
    match parse_result {
        Ok(ast) => {
            println!("Ast: {:?}", ast);
        }
        Err(error) => {
            println!("Parse failed: {:?}", error);
        }
    }
}
