use std::borrow::Cow;
use std::str::{from_utf8, FromStr};

use crate::compiler::CompilerCallback;

#[derive(Copy, Clone, Debug)]
pub enum Token {
    Add,
    Sub,
    Mul,
    Div,

    Integer(i64),
    Float(f64),

    EOF,
}

#[derive(Copy, Clone, Debug)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

#[derive(Copy, Clone, Debug)]
pub struct LineNumber(pub(super) u32);

#[derive(Copy, Clone, Debug)]
pub struct Source {
    pub span: Span,
    pub line_number: LineNumber,
}

pub fn lex<C: CompilerCallback, S: AsRef<[u8]>>(callback: &mut C, source: S) -> (Box<[Token]>, Box<[Source]>) {
    let source = source.as_ref();
    let mut lexer = Lexer {
        callback,
        source,
        buffer: Vec::with_capacity(16),
        cursor: Span { start: 0, end: 0 },
        line_number: 1,
        tokens: Vec::with_capacity(64),
        sources: Vec::with_capacity(64),
    };

    while lexer.peek(0).is_some() {
        lexer.read_next();
    }

    let tokens = lexer.tokens.into_boxed_slice();
    let sources = lexer.sources.into_boxed_slice();
    assert_eq!(tokens.len(), sources.len(), "Mismatch between tokens and sources");
    (tokens, sources)
}

struct Lexer<'callback, 'source, C> {
    callback: &'callback mut C,
    source: &'source [u8],
    buffer: Vec<u8>,
    cursor: Span,
    line_number: u32,
    tokens: Vec<Token>,
    sources: Vec<Source>,
}

impl<C: CompilerCallback> Lexer<'_, '_, C> {
    fn on_error(&mut self, message: impl Into<Cow<'static, str>>, source: Option<Source>) {
        (self.callback)(message.into(), source)
    }

    fn peek(&self, n: usize) -> Option<u8> {
        self.source.get(self.cursor.end as usize + n).copied()
    }

    fn advance(&mut self, n: usize) {
        assert!(self.cursor.end as usize + n <= self.source.len(), "Cannot advance past end of source");
        self.cursor.end += n as u32;
    }

    fn parse<T: FromStr>(&self) -> Option<T> {
        let s = from_utf8(&self.buffer).ok()?;
        str::parse(s).ok()
    }

    fn start_cursor(&mut self) {
        self.cursor.start = self.cursor.end;
    }

    fn token_source(&self) -> Source {
        Source {
            span: self.cursor,
            line_number: LineNumber(self.line_number),
        }
    }

    fn push_token(&mut self, token: Token) {
        self.tokens.push(token);
        self.sources.push(self.token_source());
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

        self.line_number += 1;
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
                self.on_error(
                    format!(
                        "Unexpected character '{}'",
                        char::from_u32(c as u32)
                            .unwrap_or(char::REPLACEMENT_CHARACTER)
                    ),
                    Some(self.token_source()),
                );
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
                self.on_error("Failed to parse number", Some(self.token_source()));
                self.push_token(Token::Float(f64::NAN));
            }
        }
    }
}