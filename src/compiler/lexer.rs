use std::fmt::{Display, Formatter};
use std::ops::RangeInclusive;
use std::str::{from_utf8, FromStr};

use crate::static_assert_size;

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum LexerErr {
    UnexpectedCharacter(u8),
    FailedNumberParse,
}

impl Display for LexerErr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            LexerErr::UnexpectedCharacter(c) => {
                let char = char::from_u32(*c as u32).unwrap_or(char::REPLACEMENT_CHARACTER);
                write!(f, "Unexpected character '{}'", char)
            }
            LexerErr::FailedNumberParse => {
                write!(f, "Failed to parse number")
            }
        }
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Token {
    Add,
    Sub,
    Mul,
    Div,

    Integer(i64),
    Float(f64),

    Nl,
    Eof,

    Err(LexerErr),
}

static_assert_size!(Token, 16);

#[derive(Copy, Clone, Debug)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

#[derive(Copy, Clone, Debug)]
pub struct LineNumber(pub(super) u32);

#[derive(Debug)]
pub struct Source<'source> {
    pub bytes: &'source [u8],
    pub tokens: Box<[Token]>,
    pub locations: Box<[Span]>,
    pub line_starts: Box<[u32]>,
}

impl<'source> Source<'source> {
    pub fn get_bytes_at(&self, span: Span) -> Option<&'source [u8]> {
        assert!(span.start <= span.end);
        self.bytes.get(span.start as usize..span.end as usize)
    }

    pub fn get_line_span(&self, line_number: LineNumber) -> Option<Span> {
        // TODO: this shouldn't happen
        // maybe make LineNumber store a NonZeroU32?
        if line_number.0 == 0 {
            return None;
        }

        let index = line_number.0.saturating_sub(1) as usize;
        let start = self.line_starts.get(index).copied()?;
        let end = self.line_starts
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
            Ok(line_number) => LineNumber(line_number.saturating_add(1) as u32),
            Err(line_number) => LineNumber(line_number as u32),
        }
    }
}

pub fn lex(source: &[u8]) -> Source {
    let mut lexer = Lexer {
        source,
        buffer: Vec::with_capacity(16),
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
    assert_eq!(tokens.len(), locations.len(), "Mismatch between tokens and locations");
    assert_eq!(line_starts.len(), lexer.line_number as usize, "Mismatch between line sources and line number");
    Source {
        bytes: source,
        tokens,
        locations,
        line_starts,
    }
}

struct Lexer<'source> {
    source: &'source [u8],
    buffer: Vec<u8>,
    cursor: Span,
    line_cursor: Span,
    line_number: u32,
    tokens: Vec<Token>,
    locations: Vec<Span>,
    line_starts: Vec<u32>,
}

impl Lexer<'_> {
    fn on_error(&mut self, err: LexerErr) {
        self.push_token(Token::Err(err));
    }

    fn peek(&self, n: usize) -> Option<u8> {
        self.source.get(self.cursor.end as usize + n).copied()
    }

    fn advance(&mut self, n: usize) {
        assert!(self.cursor.end as usize + n <= self.source.len(), "Cannot advance past end of source");
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

    fn push_token(&mut self, token: Token) {
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
            Some(c) if c.is_ascii_digit() => self.read_number(),
            Some(c) => {
                self.advance(1);
                self.on_error(LexerErr::UnexpectedCharacter(c));
            }
            None => {}
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
}