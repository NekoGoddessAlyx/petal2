use crate::lexer::lex;

mod lexer;

fn callback(message: &'static str) {
    println!("{}", message);
}

fn main() {
    let source = "1 + 2 * 3\n* 2";

    let (tokens, sources) = lex(callback, source);
    println!("Tokens: {:?}", tokens);
    println!("Sources: {:?}", sources);
}
