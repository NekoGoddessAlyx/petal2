use crate::compiler::ast::{Ast, BinOp, Node, NodeRef, UnOp};
use crate::compiler::Callback;
use crate::compiler::lexer::Token;

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

pub fn parse<C: Callback>(callback: &mut C, tokens: &[Token]) -> Result<Ast, ParserError> {
    let len = tokens.len();
    let mut parser = Parser {
        callback,
        had_error: false,
        tokens,
        cursor: 0,
        state: Vec::with_capacity(32),
        ast: Ast::new(len),
    };

    parser.push_state(State::BeginExpression(Precedence::root()));
    parser.parse()
}

#[derive(Debug)]
enum State {
    BeginExpression(Precedence),
    EndExpression(NodeRef),
    BeginExpressionInfix(Precedence, NodeRef),

    EndPrefixExpression(Precedence, UnOp),
    EndBinaryExpression(Precedence, BinOp, NodeRef),
}

impl State {
    fn enter<C: Callback>(&mut self, from: Option<State>, parser: &mut Parser<'_, '_, C>) -> Result<(), StateError> {
        macro_rules! fail_transfer {
            () => {
                {
                    dbg!(&from);
                    Err(StateError::CannotTransfer)
                }
            };
        }

        match *self {
            State::BeginExpression(precedence) => match from {
                None | // <- only valid while expression is the root
                Some(State::BeginExpression(..)) |
                Some(State::BeginExpressionInfix(..)) => {
                    parser.begin_expression(precedence);
                    Ok(())
                }
                _ => fail_transfer!(),
            }
            State::EndExpression(..) => match from {
                Some(State::BeginExpressionInfix(..)) => Ok(()),
                _ => fail_transfer!(),
            }
            State::BeginExpressionInfix(precedence, left) => match from {
                Some(State::BeginExpression(..)) |
                Some(State::EndPrefixExpression(..)) |
                Some(State::EndBinaryExpression(..)) => {
                    parser.begin_expression_infix(precedence, left);
                    Ok(())
                }
                _ => fail_transfer!(),
            }

            State::EndPrefixExpression(precedence, op) => match from {
                Some(State::EndExpression(right)) => {
                    parser.end_prefix_expression(precedence, op, right);
                    Ok(())
                }
                _ => fail_transfer!(),
            }
            State::EndBinaryExpression(precedence, op, left) => match from {
                Some(State::EndExpression(right)) => {
                    parser.end_binary_expression(precedence, op, left, right);
                    Ok(())
                }
                _ => fail_transfer!(),
            }
        }
    }
}

struct Parser<'callback, 'tokens, C> {
    callback: &'callback mut C,
    had_error: bool,
    tokens: &'tokens [Token],
    cursor: usize,
    state: Vec<State>,
    ast: Ast,
}

impl<C: Callback> Parser<'_, '_, C> {
    fn on_error(&mut self, message: &'static str) {
        self.had_error = true;
        (self.callback)(message);
    }

    fn peek(&self) -> Token {
        self.tokens.get(self.cursor).copied().unwrap_or(Token::EOF)
    }

    fn advance(&mut self) -> Token {
        let token = self.tokens[self.cursor];
        self.cursor += 1;
        token
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

        match self.had_error {
            true => Err(ParserError::FailedParse),
            false => Ok(self.ast),
        }
    }

    // expressions

    fn begin_expression(&mut self, precedence: Precedence) {
        match self.peek() {
            Token::Sub => {
                self.advance();
                self.push_state(State::EndPrefixExpression(precedence, UnOp::Neg));
                self.push_state(State::BeginExpression(Precedence::Prefix));
            }
            Token::Integer(v) => {
                self.advance();
                let left = self.ast.push(Node::Integer(v));
                self.push_state(State::BeginExpressionInfix(precedence, left));
            }
            Token::Float(v) => {
                self.advance();
                let left = self.ast.push(Node::Float(v));
                self.push_state(State::BeginExpressionInfix(precedence, left));
            }
            _ => {
                self.on_error("Expected expression");

                // keep the state consistent
                let left = self.ast.push(Node::Integer(0));
                self.push_state(State::BeginExpressionInfix(precedence, left));
            }
        }
    }

    fn begin_expression_infix(&mut self, precedence: Precedence, left: NodeRef) {
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
        let left = self.ast.push(Node::UnOp(op, right));
        self.push_state(State::BeginExpressionInfix(precedence, left));
    }

    fn end_binary_expression(&mut self, precedence: Precedence, op: BinOp, left: NodeRef, right: NodeRef) {
        let left = self.ast.push(Node::BinOp(op, left, right));
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
        Token::Add |
        Token::Sub => Precedence::Additive,
        Token::Mul |
        Token::Div => Precedence::Multiplicative,
        Token::Integer(_) |
        Token::Float(_) |
        Token::EOF => Precedence::None,
    }
}