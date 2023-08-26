use std::borrow::Cow;

use crate::compiler::ast::Ast;
use crate::compiler::lexer::{lex, Source};
use crate::compiler::parser::{parse, ParserError};

mod lexer;
mod parser;
pub mod ast;

pub trait Callback: FnMut(&str) {}

impl<T: FnMut(&str)> Callback for T {}

pub trait CompilerCallback: FnMut(Cow<str>, Option<Source>) {}

impl<T: FnMut(Cow<str>, Option<Source>)> CompilerCallback for T {}

pub fn compile<C, S>(mut callback: C, source: S) -> Result<Ast, ParserError>
    where C: Callback,
          S: AsRef<[u8]> {
    let source = source.as_ref();

    let mut callback = |message: Cow<str>, at: Option<Source>| {
        let full_message = match at {
            Some(at) => format!(
                "Error: {}\n[line {}] {}",
                message,
                at.line_number.0,
                String::from_utf8_lossy(&source[at.span.start as usize..at.span.end as usize]),
            ),
            None => format!(
                "Error: {}",
                message,
            ),
        };

        callback(&full_message);
    };

    let (tokens, sources) = lex(&mut callback, source);
    println!("Tokens: {:?}", tokens);
    println!("Sources: {:?}", sources);

    parse(&mut callback, &tokens, &sources)
}