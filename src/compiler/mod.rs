use crate::compiler::ast::Ast;
use crate::compiler::lexer::lex;
use crate::compiler::parser::{parse, ParserError};

mod lexer;
mod parser;
pub mod ast;

pub trait Callback: FnMut(&'static str) {}

impl<T: FnMut(&'static str)> Callback for T {}

pub fn compile<C, S>(mut callback: C, source: S) -> Result<Ast, ParserError>
    where C: Callback,
          S: AsRef<[u8]> {
    let source = source.as_ref();

    let (tokens, sources) = lex(&mut callback, source);
    println!("Tokens: {:?}", tokens);
    println!("Sources: {:?}", sources);

    parse(&mut callback, &tokens)
}