use std::fmt::{Display, Formatter};
use std::str::from_utf8;

use crate::compiler::ast::Ast;
use crate::compiler::lexer::{lex, Source, Span};
use crate::compiler::parser::{parse, ParserError};

pub mod ast;
pub mod code_gen;
mod lexer;
mod parser;
mod registers;

// callback

/// Some message generated during compilation.
///
/// Right meow this is only for errors but in the future it could
/// contain warnings or other diagnostic information.
///
/// May have information about where this message was generated from.
pub struct CompilerMessage<'compiler> {
    message: &'compiler dyn Display,
    source: &'compiler Source<'compiler>,
    at: Option<Span>,
}

#[allow(unused)]
impl CompilerMessage<'_> {
    pub fn message(&self) -> &dyn Display {
        self.message
    }

    pub fn line_number(&self) -> Option<u32> {
        self.at
            .map(|at| (*self.source.get_line_numbers(at).start()).into())
    }
}

impl Display for CompilerMessage<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Error: {}", self.message)?;

        let source = self.source;
        if let Some(at) = self.at {
            writeln!(f)?;

            let line_number: usize = (*source.get_line_numbers(at).start()).into();

            let line_span = source.get_line_span(line_number).unwrap_or(at);
            let source_line = source
                .get_bytes_at(line_span)
                .and_then(|line| from_utf8(line).ok())
                .map(|source_line| source_line.trim());

            match source_line {
                Some(source_line) => {
                    let caret_len = source
                        .get_bytes_at(at)
                        .and_then(|source| from_utf8(source).ok())
                        .map(|source| source.trim_end().len().max(1))
                        .unwrap_or(1);
                    let padding = source
                        .get_bytes_at(Span {
                            start: line_span.start,
                            end: at.start,
                        })
                        .and_then(|preceding| from_utf8(preceding).ok())
                        .map(|preceding| preceding.trim_start().len())
                        .unwrap_or(0);

                    // ilog10() + 1 is the number of digits in line_number
                    // magic number 6: the number of characters in "[line ]",
                    // excluding the ']' because that's getting replaced with '|'
                    let indent = line_number.ilog10() as usize + 1 + 6;

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
where
    C: FnMut(CompilerMessage),
    S: AsRef<[u8]>,
{
    let source = lex(source.as_ref());
    println!("Tokens: {:?}", source.tokens);
    println!("Locations: {:?}", source.locations);
    println!("Line Starts: {:?}", source.line_starts);

    let parser_callback = |message: &dyn Display, at: Option<Span>| {
        callback(CompilerMessage {
            message,
            source: &source,
            at,
        });
    };

    parse(parser_callback, &source.tokens, &source.locations)
}

mod callback {
    use std::fmt::Display;

    use crate::compiler::lexer::Span;

    pub trait ParserCallback: FnMut(&dyn Display, Option<Span>) {}

    impl<T: FnMut(&dyn Display, Option<Span>)> ParserCallback for T {}
}
