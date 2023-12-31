use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::str::from_utf8;
use std::time::Duration;

use gc_arena::Mutation;
use thiserror::Error;

use crate::compiler::ast::{Ast1, Ast2};
use crate::compiler::callback::Diagnostic;
use crate::compiler::code_gen::{code_gen, CodeGenError};
use crate::compiler::lexer::{lex, Source, Span};
use crate::compiler::parser::{parse, ParserError};
use crate::compiler::sem_check::{sem_check, SemCheckError};
use crate::prototype::Prototype;
use crate::{timed, NumDigits, PString, PStringInterner, StringInterner};

mod ast;
mod code_gen;
mod lexer;
mod parser;
mod registers;
mod sem_check;

// callback

/// Some message generated during compilation.
///
/// The message may be an error or some other diagnostic information.
/// May have information about where this message was generated from.
///
/// # Examples
/// Simply logging all messages:
/// ```rust
/// # use petal2::{CompilerMessage, PString};
/// fn callback(message: CompilerMessage<PString>) {
///     // print the message with all available information
///     // or write to some other logging mechanism
///     print!("{}", message);
/// }
/// ```
///
/// Filtering messages to only errors:
/// ```rust
/// # use petal2::{CompilerMessage, MessageKind, PString};
/// fn callback(message: CompilerMessage<PString>) {
///     // ignores all messages that aren't errors
///     if message.kind() == MessageKind::Error {
///         print!("{}", message);
///     }
/// }
/// ```
pub struct CompilerMessage<'compiler, S> {
    message: &'compiler dyn Diagnostic,
    source: &'compiler Source<'compiler, S>,
    at: Option<Span>,
}

impl<S> CompilerMessage<'_, S> {
    pub fn kind(&self) -> MessageKind {
        self.message.kind()
    }

    pub fn message(&self) -> &dyn Display {
        self.message.message()
    }

    /// A line number if this message was generated from some point in the source code
    pub fn line_number(&self) -> Option<u32> {
        self.at
            .map(|at| (*self.source.get_line_numbers(at).start()).into())
    }
}

impl<S> Display for CompilerMessage<'_, S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.kind() {
            MessageKind::Info => write!(f, "Info: "),
            MessageKind::Error => write!(f, "Error: "),
        }?;
        write!(f, "{}", self.message())?;

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

                    // magic number 6: the number of characters in "[line ]",
                    // excluding the ']' because that's getting replaced with '|'
                    let indent = line_number.num_digits() + 6;

                    write!(f, "[line {}] {}", line_number, source_line)?;

                    // render a line of carets under the offending token/span
                    // this only works correctly if the line is ascii
                    if source_line.is_ascii() {
                        const EMPTY: &str = "";

                        writeln!(f)?;
                        write!(f, "{EMPTY:-indent$}| {EMPTY:padding$}{EMPTY:^>caret_len$}")?;
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

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum MessageKind {
    Info,
    Error,
}

// compile

#[derive(Debug, Error)]
pub enum CompileError {
    /// The source input failed to compile
    CompileFailed(u32),
    /// An internal compiler error occurred, this is a bug and should be reported
    CompilerError(#[from] Box<dyn Error>),
}

impl Display for CompileError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::CompileFailed(0) => {
                write!(
                    f,
                    "Compilation failed despite no reported errors ¯\\_(ツ)_/¯"
                )
            }
            CompileError::CompileFailed(1) => {
                write!(f, "Compilation failed (1 error)")
            }
            CompileError::CompileFailed(num_errors) => {
                write!(f, "Compilation failed ({} errors)", num_errors)
            }
            CompileError::CompilerError(error) => write!(f, "Compiler error occurred: {}", error),
        }
    }
}

pub fn compile<'gc, C, S>(
    mc: &Mutation<'gc>,
    mut callback: C,
    source: S,
) -> Result<Prototype<'gc>, CompileError>
where
    C: FnMut(CompilerMessage<PString>),
    S: AsRef<[u8]>,
{
    let mut num_errors = 0;
    let mut strings = PStringInterner::default();
    let (source, lex_time) = timed(|| lex(source.as_ref(), |bytes| strings.intern(mc, bytes)));

    let mut callback = |message: &dyn Diagnostic, at: Option<Span>| {
        match message.kind() {
            MessageKind::Info => {}
            MessageKind::Error => num_errors += 1,
        }
        callback(CompilerMessage {
            message,
            source: &source,
            at,
        });
    };

    struct SourceDisplay<'source, 'gc>(&'source Source<'source, PString<'gc>>, Duration);
    impl Display for SourceDisplay<'_, '_> {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            writeln!(f, "Lexing({:?})", self.1)?;
            writeln!(f, "Tokens: {:?}", self.0.tokens)?;
            writeln!(f, "Locations: {:?}", self.0.locations)?;
            write!(f, "Line Starts: {:?}", self.0.line_starts)?;
            Ok(())
        }
    }
    callback(&(SourceDisplay(&source, lex_time), MessageKind::Info), None);

    let (parse_result, parse_time) = timed(|| {
        parse(
            |message, at| callback(message, at),
            &source.tokens,
            &source.locations,
            |bytes| strings.intern(mc, bytes),
        )
    });
    let ast = match parse_result {
        Ok(ast) => ast,
        Err(ParserError::FailedParse) => return Err(CompileError::CompileFailed(num_errors)),
        Err(error) => return Err((Box::new(error) as Box<dyn Error>).into()),
    };

    struct AstDisplay<'compiler, 'gc>(&'compiler Ast1<PString<'gc>>, Duration);
    impl Display for AstDisplay<'_, '_> {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            writeln!(f, "Parsing({:?})", self.1)?;
            writeln!(f, "Nodes: {:?}", self.0.nodes())?;
            writeln!(f, "Locations: {:?}", self.0.locations())?;
            writeln!(f, "Nodes (mem): {}", std::mem::size_of_val(self.0.nodes()))?;

            writeln!(f, "✨✨✨✨✨✨✨ Nodes (pretty) ✨✨✨✨✨✨✨")?;
            for (i, n) in self.0.nodes().iter().enumerate() {
                writeln!(f, "{}: {:?}", i, n)?;
            }
            writeln!(f, "✨✨✨✨✨✨✨ -------------- ✨✨✨✨✨✨✨")?;

            write!(f, "Ast (pretty): {}", self.0)?;
            Ok(())
        }
    }
    callback(&(AstDisplay(&ast, parse_time), MessageKind::Info), None);

    let (sem_check_result, sem_check_time) =
        timed(|| sem_check(|message, at| callback(message, at), ast));
    let ast = match sem_check_result {
        Ok(ast) => ast,
        Err(error) => {
            return match error {
                SemCheckError::FailedSemCheck => Err(CompileError::CompileFailed(num_errors)),
                error => Err((Box::new(error) as Box<dyn Error>).into()),
            };
        }
    };

    struct Ast2Display<'compiler, 'gc>(&'compiler Ast2<PString<'gc>>, Duration);
    impl Display for Ast2Display<'_, '_> {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            writeln!(f, "Semantics Check({:?})", self.1)?;
            writeln!(f, "Bindings: {:?}", self.0.bindings())?;
            write!(f, "Ast (pretty): {}", self.0)?;
            Ok(())
        }
    }
    callback(
        &(Ast2Display(&ast, sem_check_time), MessageKind::Info),
        None,
    );

    let (code_gen_result, code_gen_time) = timed(|| {
        code_gen(mc, ast, strings, |span| {
            *source.get_line_numbers(span).start()
        })
    });
    let function = match code_gen_result {
        Ok(function) => function,
        Err(error) => {
            return match error {
                CodeGenError::CodeGenFailed(message) => {
                    callback(&message, None);
                    Err(CompileError::CompileFailed(num_errors))
                }
                error => Err(CompileError::CompilerError(error.into())),
            }
        }
    };

    struct CodeGenDisplay<'compiler, 'gc>(&'compiler Prototype<'gc>, Duration);
    impl Display for CodeGenDisplay<'_, '_> {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            writeln!(f, "CodeGen({:?})", self.1)?;
            write!(f, "{}", self.0)?;
            Ok(())
        }
    }
    callback(
        &(CodeGenDisplay(&function, code_gen_time), MessageKind::Info),
        None,
    );

    Ok(function)
}

#[rustfmt::skip]
mod string {
    use std::borrow::Borrow;
    use std::fmt::{Debug, Display};
    use std::hash::Hash;
    use std::ops::Deref;

    pub trait NewString<S>: FnMut(&[u8]) -> S {}
    impl<T: FnMut(&[u8]) -> S, S: CompileString> NewString<S> for T {}

    pub trait CompileString: Clone + Eq + Hash + Borrow<[u8]> + AsRef<[u8]> + Deref<Target = [u8]> + Display + Debug {}
    impl<T: Clone + Eq + Hash + Borrow<[u8]> + AsRef<[u8]> + Deref<Target = [u8]> + Display + Debug> CompileString for T {}
}

#[rustfmt::skip]
mod callback {
    use std::fmt::Display;

    use crate::compiler::lexer::Span;
    use crate::MessageKind;

    pub trait Diagnostic {
        fn kind(&self) -> MessageKind;
        fn message(&self) -> &dyn Display;
    }

    impl<T: Display> Diagnostic for (T, MessageKind) {
        fn kind(&self) -> MessageKind {
            self.1
        }

        fn message(&self) -> &dyn Display {
            &self.0
        }
    }

    impl Diagnostic for &'static str {
        fn kind(&self) -> MessageKind {
            MessageKind::Error
        }

        fn message(&self) -> &dyn Display {
            self
        }
    }

    pub trait Callback: FnMut(&dyn Diagnostic, Option<Span>) {}
    impl<T: FnMut(&dyn Diagnostic, Option<Span>)> Callback for T {}
}
