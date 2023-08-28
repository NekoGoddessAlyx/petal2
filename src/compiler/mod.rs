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
    let source = source.as_ref();
    let tokens = lex(source);
    println!("Tokens: {:?}", tokens.tokens);
    println!("Sources: {:?}", tokens.sources);
    println!("Line Starts: {:?}", tokens.line_starts);

    let parser_callback = |message: &dyn Display, at: Option<Source>| {
        let message = CompilerMessage {
            message,
            source_information: at.map(|at| {
                SourceInformation {
                    source,
                    at,
                    line_span: {
                        let line_number = at.line_number.0 as usize;
                        let start = tokens.line_starts
                            .get(line_number.saturating_sub(1))
                            .copied()
                            .unwrap_or(at.span.start);
                        let end = tokens.line_starts
                            .get(line_number)
                            .copied()
                            .unwrap_or(source.len() as u32);
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

    parse(parser_callback, &tokens.tokens, &tokens.sources)
}

mod callback {
    use std::fmt::Display;

    use crate::compiler::lexer::Source;

    pub trait ParserCallback: FnMut(&dyn Display, Option<Source>) {}

    impl<T: FnMut(&dyn Display, Option<Source>)> ParserCallback for T {}
}