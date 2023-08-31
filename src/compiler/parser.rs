use std::fmt::Display;

use crate::compiler::ast::RefLen;
use crate::compiler::ast::{Ast, BinOp, Expr, NodeRef, Stat, UnOp};
use crate::compiler::callback::ParserCallback;
use crate::compiler::lexer::{Span, Token};
use crate::compiler::string::{StringRef, Strings};
use crate::StringInterner;

#[derive(Debug)]
pub enum ParserError {
    FailedParse,
    StateError(StateError),
}

impl From<StateError> for ParserError {
    fn from(value: StateError) -> Self {
        Self::StateError(value)
    }
}

#[derive(Debug)]
pub enum StateError {
    CannotTransfer,
}

pub fn parse<C: ParserCallback, I: StringInterner>(
    callback: C,
    tokens: &[Token],
    locations: &[Span],
    strings: &mut Strings<I>,
) -> Result<Ast<I::String>, ParserError> {
    let len = tokens.len();
    let mut parser = Parser {
        callback,
        had_error: false,
        panic_mode: false,
        tokens,
        locations,
        cursor: 0,
        strings,
        state: Vec::with_capacity(32),
        ast: Ast::new(len),
    };

    parser.push_state(State::BeginCompoundStatement);
    parser.parse()
}

#[derive(Debug)]
enum State {
    // statements
    BeginStatement,
    EndStatement(NodeRef),

    BeginCompoundStatement,
    ContinueCompoundStatement(RefLen),
    EndCompoundStatement(RefLen),

    BeginVariableDeclaration,
    EndVariableDeclaration(StringRef),

    BeginExpressionStatement,
    EndExpressionStatement,

    // expressions
    BeginExpression(Precedence),
    EndExpression(NodeRef),
    BeginExpressionInfix(Precedence, NodeRef),

    EndPrefixExpression(Precedence, UnOp),
    EndBinaryExpression(Precedence, BinOp, NodeRef),
}

impl State {
    fn enter<C: ParserCallback, I: StringInterner>(
        &mut self,
        from: Option<State>,
        parser: &mut Parser<'_, C, I>,
    ) -> Result<(), StateError> {
        macro_rules! fail_transfer {
            () => {{
                dbg!(&from);
                Err(StateError::CannotTransfer)
            }};
        }

        match *self {
            // statements
            State::BeginStatement => match from {
                Some(State::BeginCompoundStatement)
                | Some(State::ContinueCompoundStatement(..)) => {
                    parser.begin_statement();
                    Ok(())
                }
                _ => fail_transfer!(),
            },
            State::EndStatement(..) => match from {
                Some(State::EndVariableDeclaration(..)) | Some(State::EndExpressionStatement) => {
                    Ok(())
                }
                _ => fail_transfer!(),
            },

            State::BeginCompoundStatement => match from {
                // only valid while compound statement is the root
                None => {
                    parser.begin_compound_statement();
                    Ok(())
                }
                _ => fail_transfer!(),
            },
            State::ContinueCompoundStatement(len) => match from {
                Some(State::EndStatement(statement)) => {
                    parser.continue_compound_statement(len, statement);
                    Ok(())
                }
                _ => fail_transfer!(),
            },
            State::EndCompoundStatement(len) => match from {
                Some(State::ContinueCompoundStatement(..)) => {
                    parser.end_compound_statement(len);
                    Ok(())
                }
                _ => fail_transfer!(),
            },

            State::BeginVariableDeclaration => match from {
                Some(State::BeginStatement) => {
                    parser.begin_variable_declaration();
                    Ok(())
                }
                _ => fail_transfer!(),
            },
            State::EndVariableDeclaration(name) => match from {
                Some(State::BeginVariableDeclaration) => {
                    parser.end_variable_declaration(name, None);
                    Ok(())
                }
                Some(State::EndExpression(expression)) => {
                    parser.end_variable_declaration(name, Some(expression));
                    Ok(())
                }
                _ => fail_transfer!(),
            },

            State::BeginExpressionStatement => match from {
                Some(State::BeginStatement) => {
                    parser.begin_expression_statement();
                    Ok(())
                }
                _ => fail_transfer!(),
            },
            State::EndExpressionStatement => match from {
                Some(State::EndExpression(expression)) => {
                    parser.end_expression_statement(expression);
                    Ok(())
                }
                _ => fail_transfer!(),
            },

            // expressions
            State::BeginExpression(precedence) => match from {
                Some(State::BeginVariableDeclaration)
                | Some(State::BeginExpressionStatement)
                | Some(State::BeginExpression(..))
                | Some(State::BeginExpressionInfix(..)) => {
                    parser.begin_expression(precedence);
                    Ok(())
                }
                _ => fail_transfer!(),
            },
            State::EndExpression(..) => match from {
                Some(State::BeginExpressionInfix(..)) => Ok(()),
                _ => fail_transfer!(),
            },
            State::BeginExpressionInfix(precedence, left) => match from {
                Some(State::BeginExpression(..))
                | Some(State::EndPrefixExpression(..))
                | Some(State::EndBinaryExpression(..)) => {
                    parser.begin_expression_infix(precedence, left);
                    Ok(())
                }
                _ => fail_transfer!(),
            },

            State::EndPrefixExpression(precedence, op) => match from {
                Some(State::EndExpression(right)) => {
                    parser.end_prefix_expression(precedence, op, right);
                    Ok(())
                }
                _ => fail_transfer!(),
            },
            State::EndBinaryExpression(precedence, op, left) => match from {
                Some(State::EndExpression(right)) => {
                    parser.end_binary_expression(precedence, op, left, right);
                    Ok(())
                }
                _ => fail_transfer!(),
            },
        }
    }
}

struct Parser<'tokens, C, I: StringInterner> {
    callback: C,
    had_error: bool,
    panic_mode: bool,
    tokens: &'tokens [Token],
    locations: &'tokens [Span],
    cursor: usize,
    strings: &'tokens mut Strings<I>,
    state: Vec<State>,
    ast: Ast<I::String>,
}

impl<C: ParserCallback, I: StringInterner> Parser<'_, C, I> {
    fn on_error(&mut self, message: &dyn Display, source: Option<Span>) {
        self.had_error = true;
        if !self.panic_mode {
            self.panic_mode = true;
            (self.callback)(message, source);
        }
    }

    fn peek(&mut self) -> Token {
        let mut token = self.tokens.get(self.cursor);

        // Report error tokens, advance and continue
        while let Some(Token::Err(err)) = token {
            let source = self.locations.get(self.cursor).copied();
            self.on_error(&err, source);
            self.advance();
            token = self.tokens.get(self.cursor);
        }

        token.copied().unwrap_or(Token::Eof)
    }

    fn skip_nl(&mut self) {
        while let Token::Nl = self.peek() {
            self.advance();
        }
    }

    fn peek_location(&self) -> Option<Span> {
        self.locations
            .get(self.cursor)
            .or(self.locations.last())
            .copied()
    }

    fn advance(&mut self) -> Token {
        let token = self.tokens.get(self.cursor).copied().unwrap_or(Token::Eof);
        self.cursor += 1;
        token
    }

    fn end_of_statement(&mut self) {
        match self.peek() {
            Token::Nl | Token::Eof => {}
            _ => {
                self.on_error(&"Expected end of statement", self.peek_location());
            }
        }
    }

    fn push_state(&mut self, state: State) {
        self.state.push(state);
    }

    fn pop_state(&mut self) -> Option<State> {
        self.state.pop()
    }

    // parse

    fn parse(mut self) -> Result<Ast, ParserError> {
        let mut previous = None;

        while let Some(mut state) = self.pop_state() {
            state.enter(previous, &mut self)?;
            previous = Some(state);
        }

        self.skip_nl();
        if self.peek() != Token::Eof {
            self.on_error(&"Could not read all tokens", self.peek_location());
        }

        match self.had_error {
            true => Err(ParserError::FailedParse),
            false => Ok(self.ast),
        }
    }

    // statements

    fn begin_statement(&mut self) {
        self.skip_nl();
        match self.peek() {
            _ => {
                self.push_state(State::BeginExpressionStatement);
            }
        }
    }

    fn begin_compound_statement(&mut self) {
        self.push_state(State::ContinueCompoundStatement(RefLen(0)));
        self.push_state(State::BeginStatement);
    }

    fn continue_compound_statement(&mut self, len: RefLen, statement: NodeRef) {
        self.ast.push_ref(statement);

        // panic recovery
        if self.panic_mode {
            self.panic_mode = false;

            loop {
                if is_statement(self.peek()) {
                    break;
                }

                self.advance();
            }
        }

        self.skip_nl();
        match self.peek() {
            Token::Eof => {
                self.push_state(State::EndCompoundStatement(RefLen(len.0 + 1)));
            }
            _ => {
                self.push_state(State::ContinueCompoundStatement(RefLen(len.0 + 1)));
                self.push_state(State::BeginStatement);
            }
        }
    }

    fn end_compound_statement(&mut self, len: RefLen) {
        let _statement = self.ast.push_node(Stat::Compound(len));
    }

    fn begin_expression_statement(&mut self) {
        self.push_state(State::EndExpressionStatement);
        self.push_state(State::BeginExpression(Precedence::root()));
    }

    fn end_expression_statement(&mut self, expression: NodeRef) {
        let statement = self.ast.push_node(Stat::Expr(expression));
        self.push_state(State::EndStatement(statement));
        self.end_of_statement();
    }

    // expressions

    fn begin_expression(&mut self, precedence: Precedence) {
        self.skip_nl();
        match self.peek() {
            Token::Sub => {
                self.advance();
                self.push_state(State::EndPrefixExpression(precedence, UnOp::Neg));
                self.push_state(State::BeginExpression(Precedence::Prefix));
            }
            Token::Integer(v) => {
                self.advance();
                let left = self.ast.push_node(Expr::Integer(v));
                self.push_state(State::BeginExpressionInfix(precedence, left));
            }
            Token::Float(v) => {
                self.advance();
                let left = self.ast.push_node(Expr::Float(v));
                self.push_state(State::BeginExpressionInfix(precedence, left));
            }
            _ => {
                self.on_error(&"Expected expression", self.peek_location());

                // keep the state consistent
                let left = self.ast.push_node(Expr::Integer(0));
                self.push_state(State::BeginExpressionInfix(precedence, left));
            }
        }
    }

    fn begin_expression_infix(&mut self, precedence: Precedence, left: NodeRef) {
        // do *not* skip newlines
        let next_token = self.peek();
        let next_precedence = get_precedence(next_token);

        if precedence > next_precedence {
            self.push_state(State::EndExpression(left));
            return;
        }

        match next_token {
            Token::Add => {
                self.advance();
                self.push_state(State::EndBinaryExpression(precedence, BinOp::Add, left));
                self.push_state(State::BeginExpression(next_precedence.next_precedence()));
            }
            Token::Sub => {
                self.advance();
                self.push_state(State::EndBinaryExpression(precedence, BinOp::Sub, left));
                self.push_state(State::BeginExpression(next_precedence.next_precedence()));
            }
            Token::Mul => {
                self.advance();
                self.push_state(State::EndBinaryExpression(precedence, BinOp::Mul, left));
                self.push_state(State::BeginExpression(next_precedence.next_precedence()));
            }
            Token::Div => {
                self.advance();
                self.push_state(State::EndBinaryExpression(precedence, BinOp::Div, left));
                self.push_state(State::BeginExpression(next_precedence.next_precedence()));
            }
            _ => {
                self.push_state(State::EndExpression(left));
            }
        }
    }

    fn end_prefix_expression(&mut self, precedence: Precedence, op: UnOp, right: NodeRef) {
        let left = self.ast.push_node(Expr::UnOp(op, right));
        self.push_state(State::BeginExpressionInfix(precedence, left));
    }

    fn end_binary_expression(
        &mut self,
        precedence: Precedence,
        op: BinOp,
        left: NodeRef,
        right: NodeRef,
    ) {
        let left = self.ast.push_node(Expr::BinOp(op, left, right));
        self.push_state(State::BeginExpressionInfix(precedence, left));
    }
}

#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
enum Precedence {
    None,
    Additive,
    Multiplicative,
    Prefix,
    Primary,
}

impl Precedence {
    fn root() -> Self {
        Precedence::Additive
    }

    fn next_precedence(self) -> Self {
        match self {
            Precedence::None => Precedence::Additive,
            Precedence::Additive => Precedence::Multiplicative,
            Precedence::Multiplicative => Precedence::Prefix,
            Precedence::Prefix | Precedence::Primary => Precedence::Primary,
        }
    }
}

fn get_precedence(token: Token) -> Precedence {
    match token {
        Token::Add | Token::Sub => Precedence::Additive,
        Token::Mul | Token::Div => Precedence::Multiplicative,
        Token::Var
        | Token::Integer(_)
        | Token::Float(_)
        | Token::Identifier(_)
        | Token::Nl
        | Token::Eof
        | Token::Err(_) => Precedence::None,
    }
}

fn is_statement(token: Token) -> bool {
    match token {
        token if is_expression(token) => true,
        _ => false,
    }
}

fn is_expression(token: Token) -> bool {
    match token {
        Token::Integer(_) | Token::Float(_) => true,
        _ => false,
    }
}
