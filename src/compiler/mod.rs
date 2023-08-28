use std::fmt::{Display, Formatter};
use std::str::from_utf8;

use crate::compiler::ast::Ast;
use crate::compiler::lexer::{lex, SourceLocation, Span};
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
    at: SourceLocation,
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

            let line_number = source.at.line_number.0;
            let source_line = source.source
                .get(source.line_span.start as usize..source.line_span.end as usize)
                .and_then(|line| from_utf8(line).ok())
                .map(|source_line| source_line.trim());

            match source_line {
                Some(source_line) => {
                    let caret_len = source.source
                        .get(source.at.span.start as usize..source.at.span.end as usize)
                        .and_then(|source| from_utf8(source).ok())
                        .map(|source| source.trim_end().len().max(1))
                        .unwrap_or(1);
                    let padding = source.source
                        .get(source.line_span.start as usize..source.at.span.start as usize)
                        .and_then(|preceding| from_utf8(preceding).ok())
                        .map(|preceding| preceding.trim_start().len())
                        .unwrap_or(0);

                    // magic number 6: the number of characters in "[line ]",
                    // excluding the ']' because that's getting replaced with '|'
                    let indent = match line_number {
                        0 => 1,
                        v => v.ilog10() as usize + 1
                    } + 6;

                    write!(f, "[line {}] {}", line_number, source_line)?;

                    // render a line of carets under the offending token/span
                    // this only works correctly if the line is ascii
                    if source_line.is_ascii() {
                        const EMPTY: &str = "";

                        writeln!(f)?;
                        write!(f, "{EMPTY:-indent$}| {EMPTY:padding$}{EMPTY:^>caret_len$}")?
                    }
                }
                None => {
                    write!(f, "[line {}] (could not display source)", line_number)?;
                }
            };
        }

        Ok(())
    }
}

// compile

pub fn compile<C, S>(mut callback: C, source: S) -> Result<Ast, ParserError>
    where C: FnMut(CompilerMessage),
          S: AsRef<[u8]> {
    let source = lex(source.as_ref());
    println!("Tokens: {:?}", source.tokens);
    println!("Locations: {:?}", source.locations);
    println!("Line Starts: {:?}", source.line_starts);

    let parser_callback = |message: &dyn Display, at: Option<SourceLocation>| {
        let message = CompilerMessage {
            message,
            source_information: at.map(|at| {
                SourceInformation {
                    source: source.bytes,
                    at,
                    line_span: {
                        let line_number = at.line_number.0 as usize;
                        let start = source.line_starts
                            .get(line_number.saturating_sub(1))
                            .copied()
                            .unwrap_or(at.span.start);
                        let end = source.line_starts
                            .get(line_number)
                            .copied()
                            .unwrap_or(source.bytes.len() as u32);
                        Span {
                            start,
                            end,
                        }
                    },
                }
            }),
        };
        callback(message);
    };

    parse(parser_callback, &source.tokens, &source.locations)
}

mod callback {
    use std::fmt::Display;

    use crate::compiler::lexer::SourceLocation;

    pub trait ParserCallback: FnMut(&dyn Display, Option<SourceLocation>) {}

    impl<T: FnMut(&dyn Display, Option<SourceLocation>)> ParserCallback for T {}
}