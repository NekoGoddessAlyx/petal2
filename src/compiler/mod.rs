use std::fmt::{Display, Formatter};
use std::str::from_utf8;

use crate::compiler::ast::Ast;
use crate::compiler::lexer::{lex, Source, Span};
use crate::compiler::parser::{parse, ParserError};

mod lexer;
mod parser;
pub mod ast;

// callback

pub trait Callback: FnMut(CompilerMessage) {}

impl<T: FnMut(CompilerMessage)> Callback for T {}

pub struct CompilerMessage<'compiler>(&'compiler dyn Display, Option<(&'compiler [u8], Source, Span)>);

impl Display for CompilerMessage<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Error: {}", self.0)?;
        if let Some(source) = self.1 {
            writeln!(f)?;
            write!(f, "[line {}]", source.1.line_number.0)?;

            let start = source.2.start as usize;
            let end = source.2.end as usize;
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

    let lexer_callback = |message: &dyn Display, at: Option<Source>| {
        let message = CompilerMessage(
            message,
            at.map(|at| (source, at, at.span)),
        );
        callback(message);
    };

    let (tokens, sources, line_spans) = lex(lexer_callback, source);
    println!("Tokens: {:?}", tokens);
    println!("Sources: {:?}", sources);
    println!("Lines: {:?}", line_spans);

    let parser_callback = |message: &dyn Display, at: Option<Source>| {
        let message = CompilerMessage(
            message,
            at.map(|at| {
                let line_span = line_spans
                    .get((at.line_number.0 as usize).saturating_sub(1))
                    .copied()
                    .unwrap_or(at.span);
                (source, at, line_span)
            }),
        );
        callback(message);
    };

    parse(parser_callback, &tokens, &sources)
}

mod callback {
    use std::fmt::Display;

    use crate::compiler::lexer::Source;

    pub trait CompilerCallback: FnMut(&dyn Display, Option<Source>) {}

    impl<T: FnMut(&dyn Display, Option<Source>)> CompilerCallback for T {}
}