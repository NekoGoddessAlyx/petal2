use std::fmt::{Display, Formatter};
use std::str::from_utf8;

use crate::compiler::ast::Ast;
use crate::compiler::lexer::{lex, Source};
use crate::compiler::parser::{parse, ParserError};

mod lexer;
mod parser;
pub mod ast;

// callback

pub trait Callback: FnMut(CompilerMessage) {}

impl<T: FnMut(CompilerMessage)> Callback for T {}

pub struct CompilerMessage<'compiler>(&'compiler dyn Display, Option<(&'compiler [u8], Source)>);

impl Display for CompilerMessage<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Error: {}", self.0)?;
        if let Some(source) = self.1 {
            writeln!(f)?;
            write!(f, "[line {}]", source.1.line_number.0)?;

            let start = source.1.span.start as usize;
            let end = source.1.span.end as usize;
            let source = source.0
                .get(start..end)
                .and_then(|source| from_utf8(source).ok());
            match source {
                Some(source) => write!(f, " {}", source)?,
                None => write!(f, " (could not display source)")?,
            }
        }

        Ok(())
    }
}

// compile

pub fn compile<C, S>(mut callback: C, source: S) -> Result<Ast, ParserError>
    where C: Callback,
          S: AsRef<[u8]> {
    let source = source.as_ref();

    let mut callback = |message: &dyn Display, at: Option<Source>| {
        let message = CompilerMessage(
            message,
            at.map(|at| (source, at)),
        );
        callback(message);
    };

    let (tokens, sources) = lex(&mut callback, source);
    println!("Tokens: {:?}", tokens);
    println!("Sources: {:?}", sources);

    parse(&mut callback, &tokens, &sources)
}

mod callback {
    use std::fmt::Display;

    use crate::compiler::lexer::Source;

    pub trait CompilerCallback: FnMut(&dyn Display, Option<Source>) {}

    impl<T: FnMut(&dyn Display, Option<Source>)> CompilerCallback for T {}
}