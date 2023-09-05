use std::fmt::Display;

use crate::compiler::ast::{Ast, BinOp, Expr, Mutability, NodeRef, Stat, UnOp};
use crate::compiler::ast::{Node, RefLen};
use crate::compiler::callback::Callback;
use crate::compiler::lexer::{Span, Token};
use crate::compiler::string::{CompileString, NewString};

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

pub fn parse<C: Callback, NS: NewString<S>, S: CompileString>(
    callback: C,
    tokens: &[Token<S>],
    locations: &[Span],
    new_string: NS,
) -> Result<Ast<S>, ParserError> {
    let mut parser = Parser {
        callback,
        new_string,
        had_error: false,
        panic_mode: false,
        tokens,
        locations,
        cursor: 0,
        state: Vec::with_capacity(32),
        nodes: Vec::with_capacity(tokens.len()),
        refs: Vec::with_capacity(tokens.len() / 2),
        ast_locations: Vec::with_capacity(tokens.len()),
    };

    parser.push_state(State::BeginCompoundStatement);
    parser.parse()?;

    parser.skip_nl();
    if parser.peek() != &Token::Eof {
        parser.on_error(&"Could not read all tokens", Some(parser.peek_location()));
    }

    assert!(!parser.nodes.is_empty(), "Nodes is empty");
    assert!(!parser.locations.is_empty(), "Locations is empty");
    assert_eq!(
        tokens.len(),
        locations.len(),
        "Mismatch between nodes and locations"
    );

    match parser.had_error {
        true => Err(ParserError::FailedParse),
        false => Ok(Ast {
            nodes: parser.nodes.into_boxed_slice(),
            refs: parser.refs.into_boxed_slice(),
            locations: parser.ast_locations.into_boxed_slice(),
        }),
    }
}

#[derive(Debug)]
enum State<S> {
    // statements
    BeginStatement,
    EndStatement(NodeRef),

    BeginCompoundStatement,
    ContinueCompoundStatement(NodeRef),
    EndCompoundStatement(NodeRef),

    BeginVariableDeclaration(Span, Mutability),
    EndVariableDeclaration(Span, Mutability, S),

    BeginExpressionStatement,
    EndExpressionStatement,

    // expressions
    BeginExpression(Precedence),
    EndExpression(NodeRef),
    BeginExpressionInfix(Precedence, NodeRef),

    EndParenExpression(Precedence),
    EndVarExpression(Precedence, Span, S),
    EndReturnExpression(Precedence, Span),
    EndPrefixExpression(Precedence, Span, UnOp),
    EndBinaryExpression(Precedence, Span, BinOp, NodeRef),
}

impl<S: CompileString> State<S> {
    fn enter<C: Callback, NS: NewString<S>>(
        &mut self,
        from: Option<State<S>>,
        parser: &mut Parser<'_, C, NS, S>,
    ) -> Result<(), StateError> {
        macro_rules! fail_transfer {
            () => {{
                dbg!(&from);
                Err(StateError::CannotTransfer)
            }};
        }

        match self {
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
                Some(State::EndCompoundStatement(..))
                | Some(State::EndVariableDeclaration(..))
                | Some(State::EndExpressionStatement) => Ok(()),
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
                    parser.continue_compound_statement(*len, statement);
                    Ok(())
                }
                _ => fail_transfer!(),
            },
            State::EndCompoundStatement(statement) => match from {
                Some(State::ContinueCompoundStatement(..)) => {
                    parser.end_compound_statement(*statement);
                    Ok(())
                }
                _ => fail_transfer!(),
            },

            State::BeginVariableDeclaration(var_span, mutability) => match from {
                Some(State::BeginStatement) => {
                    parser.begin_variable_declaration(*var_span, *mutability);
                    Ok(())
                }
                _ => fail_transfer!(),
            },
            State::EndVariableDeclaration(var_span, mutability, name) => match from {
                Some(State::BeginVariableDeclaration(..)) => {
                    parser.end_variable_declaration(*var_span, *mutability, name.clone(), None);
                    Ok(())
                }
                Some(State::EndExpression(expression)) => {
                    parser.end_variable_declaration(
                        *var_span,
                        *mutability,
                        name.clone(),
                        Some(expression),
                    );
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
                Some(State::BeginVariableDeclaration(..))
                | Some(State::BeginExpressionStatement)
                | Some(State::BeginExpression(..))
                | Some(State::BeginExpressionInfix(..)) => {
                    parser.begin_expression(*precedence);
                    Ok(())
                }
                _ => fail_transfer!(),
            },
            State::EndExpression(..) => match from {
                Some(State::BeginExpressionInfix(..)) | Some(State::EndReturnExpression(..)) => {
                    Ok(())
                }
                _ => fail_transfer!(),
            },
            State::BeginExpressionInfix(precedence, left) => match from {
                Some(State::BeginExpression(..))
                | Some(State::EndParenExpression(..))
                | Some(State::EndVarExpression(..))
                | Some(State::EndReturnExpression(..))
                | Some(State::EndPrefixExpression(..))
                | Some(State::EndBinaryExpression(..)) => {
                    parser.begin_expression_infix(*precedence, *left);
                    Ok(())
                }
                _ => fail_transfer!(),
            },

            State::EndParenExpression(precedence) => match from {
                Some(State::EndExpression(expression)) => {
                    parser.end_paren_expression(*precedence, expression);
                    Ok(())
                }
                _ => fail_transfer!(),
            },
            State::EndVarExpression(precedence, var_span, name) => match from {
                Some(State::BeginExpression(..)) => {
                    parser.end_variable_expression(*precedence, *var_span, name.clone(), None);
                    Ok(())
                }
                Some(State::EndExpression(assignment)) => {
                    parser.end_variable_expression(
                        *precedence,
                        *var_span,
                        name.clone(),
                        Some(assignment),
                    );
                    Ok(())
                }
                _ => fail_transfer!(),
            },
            State::EndReturnExpression(precedence, return_span) => match from {
                Some(State::BeginExpression(..)) => {
                    parser.end_return_expression(*precedence, *return_span, None);
                    Ok(())
                }
                Some(State::EndExpression(right)) => {
                    parser.end_return_expression(*precedence, *return_span, Some(right));
                    Ok(())
                }
                _ => fail_transfer!(),
            },
            State::EndPrefixExpression(precedence, op_span, op) => match from {
                Some(State::EndExpression(right)) => {
                    parser.end_prefix_expression(*precedence, *op_span, *op, right);
                    Ok(())
                }
                _ => fail_transfer!(),
            },
            State::EndBinaryExpression(precedence, op_span, op, left) => match from {
                Some(State::EndExpression(right)) => {
                    parser.end_binary_expression(*precedence, *op_span, *op, *left, right);
                    Ok(())
                }
                _ => fail_transfer!(),
            },
        }
    }
}

struct Parser<'tokens, C, NS, S> {
    callback: C,
    new_string: NS,
    had_error: bool,
    panic_mode: bool,
    tokens: &'tokens [Token<S>],
    locations: &'tokens [Span],
    cursor: usize,
    state: Vec<State<S>>,
    nodes: Vec<Node<S>>,
    refs: Vec<NodeRef>,
    ast_locations: Vec<Span>,
}

impl<C: Callback, NS: NewString<S>, S: CompileString> Parser<'_, C, NS, S> {
    fn on_error(&mut self, message: &dyn Display, source: Option<Span>) {
        self.had_error = true;
        if !self.panic_mode {
            self.panic_mode = true;
            (self.callback)(message, source);
        }
    }

    fn new_string(&mut self, bytes: &[u8]) -> S {
        (self.new_string)(bytes)
    }

    fn peek(&mut self) -> &Token<S> {
        let mut token = self.tokens.get(self.cursor);

        // Report error tokens, advance and continue
        while let Some(Token::Err(err)) = token {
            let source = self.locations.get(self.cursor).copied();
            self.on_error(&err, source);
            self.advance();
            token = self.tokens.get(self.cursor);
        }

        token.unwrap_or(&Token::Eof)
    }

    fn skip_nl(&mut self) {
        while let Token::Nl = self.peek() {
            self.advance();
        }
    }

    fn peek_location(&self) -> Span {
        self.locations
            .get(self.cursor)
            .or(self.locations.last())
            .copied()
            .unwrap()
    }

    fn advance(&mut self) -> Token<S> {
        let token = self.tokens.get(self.cursor).cloned().unwrap_or(Token::Eof);
        self.cursor += 1;
        token
    }

    fn end_of_statement(&mut self) {
        match self.peek() {
            Token::Nl | Token::Eof => {}
            _ => {
                self.on_error(&"Expected end of statement", Some(self.peek_location()));
            }
        }
    }

    fn push_state(&mut self, state: State<S>) {
        self.state.push(state);
    }

    fn pop_state(&mut self) -> Option<State<S>> {
        self.state.pop()
    }

    fn push_node<N: Into<Node<S>>>(&mut self, node: N, location: Span) -> NodeRef {
        let index = self.nodes.len();
        self.nodes.push(node.into());
        self.ast_locations.push(location);
        NodeRef(index as u32)
    }

    fn push_ref(&mut self, root: NodeRef, index: NodeRef) {
        match self.nodes.get_mut(root.0 as usize) {
            Some(Node::Stat(Stat::Compound { len })) => {
                len.0 += 1;
            }
            _ => todo!("Unexpected node"),
        }
        self.refs.push(index);
    }

    fn get_node_location(&self, index: NodeRef) -> Span {
        self.ast_locations[index.0 as usize]
    }

    // parse

    fn parse(&mut self) -> Result<(), ParserError> {
        let mut previous = None;

        while let Some(mut state) = self.pop_state() {
            state.enter(previous, self)?;
            previous = Some(state);
        }

        Ok(())
    }

    // statements

    fn begin_statement(&mut self) {
        self.skip_nl();
        match self.peek() {
            Token::Val => {
                let val_span = self.peek_location();
                self.advance();
                self.push_state(State::BeginVariableDeclaration(
                    val_span,
                    Mutability::Immutable,
                ));
            }
            Token::Var => {
                let var_span = self.peek_location();
                self.advance();
                self.push_state(State::BeginVariableDeclaration(
                    var_span,
                    Mutability::Mutable,
                ));
            }
            _ => {
                self.push_state(State::BeginExpressionStatement);
            }
        }
    }

    fn begin_compound_statement(&mut self) {
        let statement = self.push_node(Stat::Compound { len: RefLen(0) }, self.peek_location());
        self.push_state(State::ContinueCompoundStatement(statement));
        self.push_state(State::BeginStatement);
    }

    fn continue_compound_statement(&mut self, root: NodeRef, child: NodeRef) {
        self.push_ref(root, child);

        // panic recovery
        if self.panic_mode {
            self.panic_mode = false;

            loop {
                match self.peek() {
                    Token::Eof => break,
                    token if token.is_statement() => break,
                    _ => self.advance(),
                };
            }
        }

        self.skip_nl();
        match self.peek() {
            Token::Eof => {
                self.push_state(State::EndCompoundStatement(root));
            }
            _ => {
                self.push_state(State::ContinueCompoundStatement(root));
                self.push_state(State::BeginStatement);
            }
        }
    }

    fn end_compound_statement(&mut self, node: NodeRef) {
        self.push_state(State::EndStatement(node));
    }

    fn begin_variable_declaration(&mut self, var_span: Span, mutability: Mutability) {
        self.skip_nl();
        let name = match self.peek() {
            Token::Identifier(name) => {
                let name = name.clone();
                self.advance();
                name
            }
            _ => {
                self.on_error(&"Expected variable name", Some(self.peek_location()));
                self.new_string(b"")
            }
        };

        self.push_state(State::EndVariableDeclaration(var_span, mutability, name));

        // do not skip NL
        if let Token::Eq = self.peek() {
            self.advance();
            self.push_state(State::BeginExpression(Precedence::root()));
        }
    }

    fn end_variable_declaration(
        &mut self,
        var_span: Span,
        mutability: Mutability,
        name: S,
        def: Option<NodeRef>,
    ) {
        let statement = self.push_node(
            Stat::VarDecl {
                mutability,
                name,
                def,
            },
            var_span,
        );
        self.push_state(State::EndStatement(statement));
        self.end_of_statement();
    }

    fn begin_expression_statement(&mut self) {
        self.push_state(State::EndExpressionStatement);
        self.push_state(State::BeginExpression(Precedence::root()));
    }

    fn end_expression_statement(&mut self, expression: NodeRef) {
        let statement = self.push_node(
            Stat::Expr { expr: expression },
            self.get_node_location(expression),
        );
        self.push_state(State::EndStatement(statement));
        self.end_of_statement();
    }

    // expressions

    fn begin_expression(&mut self, precedence: Precedence) {
        self.skip_nl();
        match *self.peek() {
            Token::Return => {
                let return_span = self.peek_location();
                self.advance();
                self.push_state(State::EndReturnExpression(precedence, return_span));

                // do not peek
                if self.peek().is_expression() {
                    self.push_state(State::BeginExpression(Precedence::root()));
                }
            }
            Token::ParenOpen => {
                self.advance();
                self.push_state(State::EndParenExpression(precedence));
                self.push_state(State::BeginExpression(Precedence::root()));
            }
            Token::Sub => {
                let op_span = self.peek_location();
                self.advance();
                self.push_state(State::EndPrefixExpression(precedence, op_span, UnOp::Neg));
                self.push_state(State::BeginExpression(Precedence::Prefix));
            }
            Token::Integer(v) => {
                let expr_span = self.peek_location();
                self.advance();
                let left = self.push_node(Expr::Integer(v), expr_span);
                self.push_state(State::BeginExpressionInfix(precedence, left));
            }
            Token::Float(v) => {
                let expr_span = self.peek_location();
                self.advance();
                let left = self.push_node(Expr::Float(v), expr_span);
                self.push_state(State::BeginExpressionInfix(precedence, left));
            }
            Token::Identifier(ref name) => {
                let name = name.clone();
                let var_span = self.peek_location();
                self.advance();

                self.push_state(State::EndVarExpression(precedence, var_span, name));

                if let Token::Eq = self.peek() {
                    self.advance();
                    self.push_state(State::BeginExpression(Precedence::root()));
                }
            }
            _ => {
                let location = self.peek_location();
                self.on_error(&"Expected expression", Some(location));

                // keep the state consistent
                let left = self.push_node(Expr::Integer(0), location);
                self.push_state(State::BeginExpressionInfix(precedence, left));
            }
        }
    }

    fn begin_expression_infix(&mut self, precedence: Precedence, left: NodeRef) {
        // do *not* skip newlines
        let next_token = self.peek();
        let next_precedence = next_token.precedence();

        if precedence > next_precedence {
            self.push_state(State::EndExpression(left));
            return;
        }

        match next_token {
            Token::Add => {
                let op_span = self.peek_location();
                self.advance();
                self.push_state(State::EndBinaryExpression(
                    precedence,
                    op_span,
                    BinOp::Add,
                    left,
                ));
                self.push_state(State::BeginExpression(next_precedence.next_precedence()));
            }
            Token::Sub => {
                let op_span = self.peek_location();
                self.advance();
                self.push_state(State::EndBinaryExpression(
                    precedence,
                    op_span,
                    BinOp::Sub,
                    left,
                ));
                self.push_state(State::BeginExpression(next_precedence.next_precedence()));
            }
            Token::Mul => {
                let op_span = self.peek_location();
                self.advance();
                self.push_state(State::EndBinaryExpression(
                    precedence,
                    op_span,
                    BinOp::Mul,
                    left,
                ));
                self.push_state(State::BeginExpression(next_precedence.next_precedence()));
            }
            Token::Div => {
                let op_span = self.peek_location();
                self.advance();
                self.push_state(State::EndBinaryExpression(
                    precedence,
                    op_span,
                    BinOp::Div,
                    left,
                ));
                self.push_state(State::BeginExpression(next_precedence.next_precedence()));
            }
            _ => {
                self.push_state(State::EndExpression(left));
            }
        }
    }

    fn end_paren_expression(&mut self, precedence: Precedence, expression: NodeRef) {
        match self.peek() {
            Token::ParenClose => {
                self.advance();
            }
            _ => {
                self.on_error(&"Expected ')'", Some(self.peek_location()));
            }
        };
        self.push_state(State::BeginExpressionInfix(precedence, expression));
    }

    fn end_variable_expression(
        &mut self,
        precedence: Precedence,
        var_span: Span,
        name: S,
        right: Option<NodeRef>,
    ) {
        let left = self.push_node(
            Expr::Var {
                name,
                assignment: right,
            },
            var_span,
        );
        self.push_state(State::BeginExpressionInfix(precedence, left));
    }

    fn end_return_expression(
        &mut self,
        precedence: Precedence,
        return_span: Span,
        right: Option<NodeRef>,
    ) {
        let left = self.push_node(Expr::Return { right }, return_span);
        self.push_state(State::BeginExpressionInfix(precedence, left));
    }

    fn end_prefix_expression(
        &mut self,
        precedence: Precedence,
        op_span: Span,
        op: UnOp,
        right: NodeRef,
    ) {
        let left = self.push_node(Expr::UnOp { op, right }, op_span);
        self.push_state(State::BeginExpressionInfix(precedence, left));
    }

    fn end_binary_expression(
        &mut self,
        precedence: Precedence,
        op_span: Span,
        op: BinOp,
        left: NodeRef,
        right: NodeRef,
    ) {
        let left = self.push_node(Expr::BinOp { op, left, right }, op_span);
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

impl<S> Token<S> {
    fn precedence(&self) -> Precedence {
        match self {
            Token::Add | Token::Sub => Precedence::Additive,
            Token::Mul | Token::Div => Precedence::Multiplicative,
            _ => Precedence::None,
        }
    }

    fn is_statement(&self) -> bool {
        match self {
            Token::Val | Token::Var => true,
            _ if self.is_expression() => true,
            _ => false,
        }
    }

    fn is_expression(&self) -> bool {
        // don't care
        #[allow(clippy::match_like_matches_macro)]
        match self {
            Token::Return
            | Token::ParenOpen
            | Token::Integer(_)
            | Token::Float(_)
            | Token::Identifier(_) => true,
            _ => false,
        }
    }
}
