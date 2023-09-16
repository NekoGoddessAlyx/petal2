use std::fmt::{Display, Formatter};
use std::num::NonZeroU32;
use std::ops::RangeInclusive;
use std::str::{from_utf8, FromStr};

use smallvec::{smallvec, SmallVec};

use crate::compiler::string::{CompileString, NewString};
use crate::compiler::Diagnostic;
use crate::MessageKind;

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum LexerErr {
    UnexpectedCharacter(u8),
    FailedNumberParse,
    UnknownCharacterEscape(u8),
    UnterminatedString,
}

impl Display for LexerErr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let to_char = |c: u8| char::from_u32(c as u32).unwrap_or(char::REPLACEMENT_CHARACTER);
        match *self {
            LexerErr::UnexpectedCharacter(c) => {
                write!(f, "Unexpected character '{}'", to_char(c))
            }
            LexerErr::FailedNumberParse => {
                write!(f, "Failed to parse number")
            }
            LexerErr::UnknownCharacterEscape(c) => {
                write!(f, "Unknown character escape '\\{}'", to_char(c))
            }
            LexerErr::UnterminatedString => {
                write!(f, "Unterminated string")
            }
        }
    }
}

impl Diagnostic for LexerErr {
    fn kind(&self) -> MessageKind {
        MessageKind::Error
    }

    fn message(&self) -> &dyn Display {
        self
    }
}

#[derive(Clone, Debug)]
pub enum Token<S> {
    Val,
    Var,
    If,
    Else,
    Null,
    True,
    False,
    Return,

    BraceOpen,
    BraceClose,
    ParenOpen,
    ParenClose,

    Eq,

    Bang,

    Add,
    Sub,
    Mul,
    Div,

    Integer(i64),
    Float(f64),
    String(S),

    Identifier(S),

    Nl,
    Eof,

    Err(LexerErr),
}

impl<S> PartialEq for Token<S> {
    fn eq(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }
}

// Assuming the string type is no larger than a u64
static_assert_size!(Token<u64>, 16);

#[derive(Copy, Clone, Default, Debug)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

impl Span {
    pub fn merge(self, other: Span) -> Self {
        Self {
            start: self.start.min(other.start),
            end: self.start.max(other.end),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct LineNumber(NonZeroU32);

impl LineNumber {
    fn new(value: u32) -> Self {
        Self(NonZeroU32::new(value).expect("Line numbers are 1-indexed"))
    }
    #[inline]
    fn get(self) -> u32 {
        self.0.get()
    }
}

impl From<LineNumber> for u32 {
    fn from(value: LineNumber) -> Self {
        value.get()
    }
}

impl From<LineNumber> for usize {
    fn from(value: LineNumber) -> Self {
        value.get() as usize
    }
}

#[derive(Debug)]
pub struct Source<'source, S> {
    pub bytes: &'source [u8],
    pub tokens: Box<[Token<S>]>,
    pub locations: Box<[Span]>,
    pub line_starts: Box<[u32]>,
}

impl<'source, S> Source<'source, S> {
    pub fn get_bytes_at(&self, span: Span) -> Option<&'source [u8]> {
        assert!(span.start <= span.end);
        self.bytes.get(span.start as usize..span.end as usize)
    }

    pub fn get_line_span(&self, line_number: impl Into<usize>) -> Option<Span> {
        let line_number: usize = line_number.into();
        let index = line_number.saturating_sub(1);
        let start = self.line_starts.get(index).copied()?;
        let end = self
            .line_starts
            .get(index + 1)
            .copied()
            .unwrap_or(self.bytes.len() as u32);
        Some(Span { start, end })
    }

    pub fn get_line_numbers(&self, span: Span) -> RangeInclusive<LineNumber> {
        assert!(span.start <= span.end);

        let start = self.index_to_line_number(span.start);
        let end = self.index_to_line_number(span.end);
        start..=end
    }

    fn index_to_line_number(&self, index: u32) -> LineNumber {
        match self.line_starts.binary_search(&index) {
            Ok(line_number) => LineNumber::new(line_number.saturating_add(1) as u32),
            Err(line_number) => LineNumber::new(line_number as u32),
        }
    }
}

pub fn lex<NS, S>(source: &[u8], new_string: NS) -> Source<S>
where
    NS: NewString<S>,
    S: CompileString,
{
    let mut lexer = Lexer {
        source,
        new_string,
        buffer: smallvec![],
        cursor: Span { start: 0, end: 0 },
        line_cursor: Span { start: 0, end: 0 },
        line_number: 1,
        tokens: Vec::with_capacity(64),
        locations: Vec::with_capacity(64),
        line_starts: Vec::with_capacity(32),
    };

    lexer.line_starts.push(0);
    while lexer.peek(0).is_some() {
        lexer.read_next();
    }

    lexer.start_cursor();
    lexer.tokens.push(Token::Eof);
    lexer.locations.push(lexer.cursor);

    let tokens = lexer.tokens.into_boxed_slice();
    let locations = lexer.locations.into_boxed_slice();
    let line_starts = lexer.line_starts.into_boxed_slice();
    assert!(!tokens.is_empty(), "Tokens is empty");
    assert!(!locations.is_empty(), "Locations is empty");
    assert!(!line_starts.is_empty(), "Line starts is empty");
    assert_eq!(
        tokens.len(),
        locations.len(),
        "Mismatch between tokens and locations"
    );
    assert_eq!(
        line_starts.len(),
        lexer.line_number as usize,
        "Mismatch between line sources and line number"
    );
    Source {
        bytes: source,
        tokens,
        locations,
        line_starts,
    }
}

struct Lexer<'source, NS, S> {
    source: &'source [u8],
    new_string: NS,
    buffer: SmallVec<[u8; 16]>,
    cursor: Span,
    line_cursor: Span,
    line_number: u32,
    tokens: Vec<Token<S>>,
    locations: Vec<Span>,
    line_starts: Vec<u32>,
}

impl<NS, S> Lexer<'_, NS, S>
where
    NS: NewString<S>,
    S: CompileString,
{
    fn on_error(&mut self, err: LexerErr) {
        self.push_token(Token::Err(err));
    }

    fn on_error_at_last_char(&mut self, err: LexerErr) {
        self.tokens.push(Token::Err(err));
        self.locations.push(Span {
            start: self.cursor.end.saturating_sub(1),
            end: self.cursor.end,
        });
    }

    fn take_string(&mut self) -> S {
        let s = (self.new_string)(&self.buffer);
        self.buffer.clear();
        s
    }

    fn peek(&self, n: usize) -> Option<u8> {
        self.source.get(self.cursor.end as usize + n).copied()
    }

    fn advance(&mut self, n: usize) {
        assert!(
            self.cursor.end as usize + n <= self.source.len(),
            "Cannot advance past end of source"
        );
        self.cursor.end += n as u32;
        self.line_cursor.end += n as u32;
    }

    fn parse<T: FromStr>(&self) -> Option<T> {
        let s = from_utf8(&self.buffer).ok()?;
        str::parse(s).ok()
    }

    fn start_cursor(&mut self) {
        self.cursor.start = self.cursor.end;
    }

    fn push_token(&mut self, token: Token<S>) {
        debug_assert_ne!(token, Token::Eof);

        self.tokens.push(token);
        self.locations.push(self.cursor);
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek(0) {
            match c {
                b'\r' | b'\n' => self.read_line_end(),
                b' ' | b'\t' => self.advance(1),
                _ => break,
            }
        }
    }

    fn read_line_end(&mut self) {
        assert!(self.peek(0).is_some_and(|c| matches!(c, b'\r' | b'\n')));

        let c1 = self.peek(0).unwrap();
        self.advance(1);

        if let Some(c2) = self.peek(0) {
            if c1 != c2 && matches!(c2, b'\r' | b'\n') {
                self.advance(1);
            }
        }

        self.push_token(Token::Nl);
        self.line_number += 1;
        self.line_cursor.start = self.line_cursor.end;
        self.line_starts.push(self.line_cursor.start);
    }

    fn read_next(&mut self) {
        self.skip_whitespace();

        self.start_cursor();

        match self.peek(0) {
            Some(b'{') => {
                self.advance(1);
                self.push_token(Token::BraceOpen);
            }
            Some(b'}') => {
                self.advance(1);
                self.push_token(Token::BraceClose);
            }
            Some(b'(') => {
                self.advance(1);
                self.push_token(Token::ParenOpen);
            }
            Some(b')') => {
                self.advance(1);
                self.push_token(Token::ParenClose);
            }
            Some(b'=') => {
                self.advance(1);
                self.push_token(Token::Eq);
            }
            Some(b'!') => {
                self.advance(1);
                self.push_token(Token::Bang);
            }
            Some(b'+') => {
                self.advance(1);
                self.push_token(Token::Add);
            }
            Some(b'-') => {
                self.advance(1);
                self.push_token(Token::Sub);
            }
            Some(b'*') => {
                self.advance(1);
                self.push_token(Token::Mul);
            }
            Some(b'/') => {
                self.advance(1);
                self.push_token(Token::Div);
            }
            Some(b'"') => {
                self.advance(1);
                self.read_string();
            }
            Some(c) if c.is_ascii_digit() => self.read_number(),
            Some(c) if c.is_ascii_alphabetic() => self.read_identifier(),
            Some(c) => {
                self.advance(1);
                self.on_error(LexerErr::UnexpectedCharacter(c));
            }
            None => {}
        }
    }

    fn read_string(&mut self) {
        fn char_escape(c: u8) -> Option<u8> {
            match c {
                b'0' => Some(b'\0'),
                b'b' => Some(0x08),
                b't' => Some(b'\t'),
                b'r' => Some(b'\r'),
                b'n' => Some(b'\n'),
                b'\\' | b'"' => Some(c),
                _ => None,
            }
        }

        struct UnterminatedString;

        let mut try_read_string = || {
            self.buffer.clear();
            loop {
                let c1 = self.peek(0).ok_or(UnterminatedString)?;
                if matches!(c1, b'\r' | b'\n') {
                    return Err(UnterminatedString);
                }

                self.advance(1);
                match c1 {
                    // escape
                    b'\\' => {
                        let c2 = self.peek(0).ok_or(UnterminatedString)?;
                        match char_escape(c2) {
                            None => {
                                self.advance(1);
                                self.buffer.push(c1);
                                self.buffer.push(c2);
                                self.on_error_at_last_char(LexerErr::UnknownCharacterEscape(c2));
                            }
                            Some(ce) => {
                                self.advance(1);
                                self.buffer.push(ce);
                            }
                        }
                    }

                    // string end
                    b'"' => {
                        let string = self.take_string();
                        self.push_token(Token::String(string));
                        return Ok(());
                    }

                    // any other char
                    c1 => self.buffer.push(c1),
                }
            }
        };

        match try_read_string() {
            Ok(_) => {}
            Err(_) => {
                self.on_error(LexerErr::UnterminatedString);
                let string = self.take_string();
                self.push_token(Token::String(string));
            }
        }
    }

    fn read_number(&mut self) {
        assert!(self.peek(0).is_some_and(|c| c.is_ascii_digit()));

        self.buffer.clear();

        let mut has_radix = false;
        let mut underscore_allowed = false;
        while let Some(c) = self.peek(0) {
            if c == b'.' && !has_radix && matches!(self.peek(1), Some(c) if c.is_ascii_digit()) {
                self.advance(1);
                self.buffer.push(b'.');
                has_radix = true;
                underscore_allowed = false;
            } else if c.is_ascii_digit() {
                self.advance(1);
                self.buffer.push(c);
                underscore_allowed = true;
            } else if underscore_allowed && c == b'_' {
                self.advance(1);
            } else {
                break;
            }
        }

        if !has_radix {
            if let Some(i) = self.parse() {
                self.push_token(Token::Integer(i));
                return;
            }
        }

        match self.parse() {
            Some(f) => {
                self.push_token(Token::Float(f));
            }
            None => {
                self.on_error(LexerErr::FailedNumberParse);
                self.push_token(Token::Float(f64::NAN));
            }
        }
    }

    fn read_identifier(&mut self) {
        assert!(self
            .peek(0)
            .is_some_and(|c| c.is_ascii_alphabetic() || c == b'_'));

        self.buffer.clear();

        while let Some(c) = self.peek(0) {
            if c.is_ascii_alphanumeric() || c == b'_' {
                self.buffer.push(c);
                self.advance(1);
            } else {
                break;
            }
        }

        let string = self.take_string();
        match string.as_ref() {
            b"val" => {
                self.push_token(Token::Val);
            }
            b"var" => {
                self.push_token(Token::Var);
            }
            b"if" => {
                self.push_token(Token::If);
            }
            b"else" => {
                self.push_token(Token::Else);
            }
            b"null" => {
                self.push_token(Token::Null);
            }
            b"true" => {
                self.push_token(Token::True);
            }
            b"false" => {
                self.push_token(Token::False);
            }
            b"return" => {
                self.push_token(Token::Return);
            }
            _ => {
                self.push_token(Token::Identifier(string));
            }
        }
    }
}
