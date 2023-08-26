use std::fmt::{Display, Formatter};
use std::str::from_utf8;

use crate::compiler::ast::Ast;
use crate::compiler::lexer::{lex, Source, Span};
use crate::compiler::parser::{parse, ParserError};

mod lexer;
mod parser;
pub mod ast;

// callback

/// Some message generated during compilation.
///
/// Right meow this is only for errors but in the future it could
/// contain warnings or other diagnostic information.
///
/// May have information about where this message was generated from.
pub struct CompilerMessage<'compiler> {
    message: &'compiler dyn Display,
    source_information: Option<SourceInformation<'compiler>>,
}

struct SourceInformation<'source> {
    source: &'source [u8],
    at: Source,
    line_span: Span,
}

#[allow(unused)]
impl CompilerMessage<'_> {
    pub fn message(&self) -> &dyn Display {
        self.message
    }

    pub fn line_number(&self) -> Option<u32> {
        self.source_information
            .as_ref()
            .map(|source| source.at.line_number.0)
    }
}

impl Display for CompilerMessage<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Error: {}", self.message)?;
        if let Some(source) = &self.source_information {
            writeln!(f)?;
            write!(f, "[line {}]", source.at.line_number.0)?;

            let start = source.line_span.start as usize;
            let end = source.line_span.end as usize;
            let source = source.source
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
    where C: FnMut(CompilerMessage),
          S: AsRef<[u8]> {
    let source = source.as_ref();

    let lexer_callback = |message: &dyn Display, at: Option<Source>| {
        let message = CompilerMessage {
            message,
            source_information: at.map(|at| {
                SourceInformation {
                    source,
                    at,
                    line_span: at.span,
                }
            }),
        };
        callback(message);
    };

    let (tokens, sources, line_spans) = lex(lexer_callback, source);
    println!("Tokens: {:?}", tokens);
    println!("Sources: {:?}", sources);
    println!("Lines: {:?}", line_spans);

    let parser_callback = |message: &dyn Display, at: Option<Source>| {
        let message = CompilerMessage {
            message,
            source_information: at.map(|at| {
                SourceInformation {
                    source,
                    at,
                    line_span: line_spans
                        .get((at.line_number.0 as usize).saturating_sub(1))
                        .copied()
                        .unwrap_or(at.span),
                }
            }),
        };
        callback(message);
    };

    parse(parser_callback, &tokens, &sources)
}

mod callback {
    use std::fmt::Display;

    use crate::compiler::lexer::Source;

    pub trait Callback: FnMut(&dyn Display, Option<Source>) {}

    impl<T: FnMut(&dyn Display, Option<Source>)> Callback for T {}
}