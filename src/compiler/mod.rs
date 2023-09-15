use std::error::Error;
use std::fmt::{Display, Formatter};
use std::str::from_utf8;

use gc_arena::Mutation;

use crate::compiler::ast::Node;
use crate::compiler::code_gen::{code_gen, CodeGenError};
use crate::compiler::lexer::{lex, Source, Span};
use crate::compiler::parser::{parse, ParserError};
use crate::compiler::sem_check::{sem_check, SemCheckError};
use crate::prototype::Prototype;
use crate::{PString, PStringInterner, StringInterner};

mod ast;
mod code_gen;
mod lexer;
mod parser;
mod registers;
mod sem_check;

// callback

/// Some message generated during compilation.
///
/// Right meow this is only for errors but in the future it could
/// contain warnings or other diagnostic information.
///
/// May have information about where this message was generated from.
pub struct CompilerMessage<'compiler, S> {
    message: &'compiler dyn Display,
    source: &'compiler Source<'compiler, S>,
    at: Option<Span>,
}

#[allow(unused)]
impl<S> CompilerMessage<'_, S> {
    pub fn message(&self) -> &dyn Display {
        self.message
    }

    pub fn line_number(&self) -> Option<u32> {
        self.at
            .map(|at| (*self.source.get_line_numbers(at).start()).into())
    }
}

impl<S> Display for CompilerMessage<'_, S> {
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

#[derive(Debug)]
pub enum CompileError {
    /// The source input failed to compile
    CompileFailed(u32),
    /// An internal compiler error occurred, this is a bug and should be reported
    CompilerError(Box<dyn Error>),
}

impl CompileError {
    fn from<T: Error + 'static>(value: T) -> Self {
        Self::CompilerError(Box::new(value) as Box<dyn Error>)
    }
}

impl Error for CompileError {}

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
    let source = lex(source.as_ref(), |bytes| strings.intern(mc, bytes));
    println!("Tokens: {:?}", source.tokens);
    println!("Locations: {:?}", source.locations);
    println!("Line Starts: {:?}", source.line_starts);

    let mut callback = |message: &dyn Display, at: Option<Span>| {
        num_errors += 1;
        callback(CompilerMessage {
            message,
            source: &source,
            at,
        });
    };

    let ast = match parse(
        |message, at| callback(message, at),
        &source.tokens,
        &source.locations,
        |bytes| strings.intern(mc, bytes),
    ) {
        Ok(ast) => ast,
        Err(ParserError::FailedParse) => return Err(CompileError::CompileFailed(num_errors)),
        Err(error) => return Err(CompileError::from(error)),
    };
    println!("Nodes: {:?}", ast.nodes);
    println!("Locations: {:?}", ast.locations);
    println!(
        "Nodes (mem): {}",
        std::mem::size_of::<Node<PString>>() * ast.nodes.len()
    );

    println!("✨✨✨✨✨✨✨ Nodes (pretty) ✨✨✨✨✨✨✨");
    for (i, n) in ast.nodes.iter().enumerate() {
        println!("{}: {:?}", i, n);
    }
    println!("✨✨✨✨✨✨✨ -------------- ✨✨✨✨✨✨✨");

    println!("Ast (pretty): {}", ast);

    let ast = match sem_check(|message, at| callback(message, at), ast) {
        Ok(ast) => ast,
        Err(error) => {
            return match error {
                SemCheckError::FailedSemCheck => Err(CompileError::CompileFailed(num_errors)),
                error => Err(CompileError::from(error)),
            };
        }
    };
    println!("Bindings: {:?}", ast.bindings);

    match code_gen(mc, ast, strings) {
        Ok(prototype) => Ok(prototype),
        Err(error) => match error {
            CodeGenError::ExpectedNode
            | CodeGenError::ExpectedRoot
            | CodeGenError::ExpectedStat
            | CodeGenError::ExpectedExpr
            | CodeGenError::BadTransition
            | CodeGenError::MissingBinding
            | CodeGenError::MissingLocalRegister => Err(CompileError::CompilerError(error.into())),
            CodeGenError::NoRegistersAvailable | CodeGenError::ConstantPoolFull => {
                callback(&error, None);
                Err(CompileError::CompileFailed(num_errors))
            }
        },
    }
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

    pub trait Callback: FnMut(&dyn Display, Option<Span>) {}
    impl<T: FnMut(&dyn Display, Option<Span>)> Callback for T {}
}
