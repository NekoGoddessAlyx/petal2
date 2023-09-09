use std::fmt::Display;

use crate::compiler::ast::{Ast, AstBuilder, BinOp, Mutability, NodeRef, Stat, UnOp};
use crate::compiler::ast::{Expr, RefLen};
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
        ast: AstBuilder::new(tokens.len()),
    };

    parser.push_state(State::BeginStatementsRoot);
    parser.parse()?;

    parser.skip_nl();
    if parser.peek() != &Token::Eof {
        parser.on_error(&"Could not read all tokens", Some(parser.peek_location()));
    }

    match parser.had_error {
        true => Err(ParserError::FailedParse),
        false => Ok(parser.ast.build()),
    }
}

// enum dispatch? I think that's a crate
#[derive(Copy, Clone, Debug)]
enum PushStat {
    StatementRoot(NodeRef),
    BlockStatement(NodeRef),
    BlockExpression(NodeRef),
}

impl PushStat {
    fn push<S>(self, ast: &mut AstBuilder<S>, stat: Stat<S>, location: Span) -> NodeRef {
        match self {
            PushStat::StatementRoot(root) => ast.patch_compound_stat(root, stat, location),
            PushStat::BlockStatement(block) => ast.patch_compound_stat(block, stat, location),
            PushStat::BlockExpression(block) => ast.patch_block_expr_stat(block, stat, location),
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum PushExpr {
    VarDeclDef(NodeRef),
    ExprStat(NodeRef),

    VarExprAssignment(NodeRef),
    ReturnExprRight(NodeRef),
    UnOpExprRight(NodeRef),
    BinExprRight(NodeRef),
}

impl PushExpr {
    fn push<S>(self, ast: &mut AstBuilder<S>, expr: Expr<S>, location: Span) -> NodeRef {
        match self {
            PushExpr::VarDeclDef(var_decl) => ast.patch_var_decl_def(var_decl, expr, location),
            PushExpr::ExprStat(root) => ast.patch_expr_stat(root, expr, location),

            PushExpr::VarExprAssignment(var_expr) => ast.patch_var_expr(var_expr, expr, location),
            PushExpr::ReturnExprRight(ret_expr) => ast.patch_return_expr(ret_expr, expr, location),
            PushExpr::UnOpExprRight(un_op_expr) => ast.patch_un_op_expr(un_op_expr, expr, location),
            PushExpr::BinExprRight(bin_op_expr) => {
                ast.patch_bin_op_expr_right(bin_op_expr, expr, location)
            }
        }
    }
}

#[derive(Debug)]
enum State {
    // root
    BeginStatementsRoot,
    ContinueStatementsRoot {
        root: NodeRef,
    },

    // statements
    BeginStatement {
        push_stat: PushStat,
    },
    EndStatement,

    BeginBlockStatement,
    ContinueBlockStatement {
        block: NodeRef,
    },
    EndBlockStatement,

    BeginVariableDeclaration,
    EndVariableDeclaration,

    BeginExpressionStatement,
    EndExpressionStatement,

    // expressions
    BeginExpression {
        push_expr: PushExpr,
        precedence: Precedence,
    },
    BeginExpressionInfix {
        precedence: Precedence,
        left: NodeRef,
    },
    EndExpression {
        expr: NodeRef,
    },

    EndParenExpression {
        precedence: Precedence,
    },

    ContinueBlockExpression {
        precedence: Precedence,
        block: NodeRef,
    },
    EndBlockExpression,
}

impl State {
    fn enter<C: Callback, NS: NewString<S>, S: CompileString>(
        &mut self,
        from: Option<State>,
        parser: &mut Parser<'_, C, NS, S>,
    ) -> Result<(), StateError> {
        macro_rules! fail_transfer {
            () => {{
                dbg!(&from);
                Err(StateError::CannotTransfer)
            }};
        }

        match *self {
            // root
            State::BeginStatementsRoot => match from {
                None => {
                    parser.begin_statements_root();
                    Ok(())
                }
                _ => fail_transfer!(),
            },
            State::ContinueStatementsRoot { root } => match from {
                Some(State::BeginStatementsRoot) | Some(State::EndStatement) => {
                    parser.continue_statements_root(root);
                    Ok(())
                }
                _ => fail_transfer!(),
            },

            // statements
            State::BeginStatement { .. } => match from {
                Some(State::ContinueStatementsRoot { .. })
                | Some(State::ContinueBlockStatement { .. })
                | Some(State::ContinueBlockExpression { .. }) => {
                    parser.begin_statement();
                    Ok(())
                }
                _ => fail_transfer!(),
            },
            State::EndStatement => match from {
                Some(State::EndVariableDeclaration)
                | Some(State::EndExpressionStatement)
                | Some(State::EndBlockStatement) => Ok(()),
                _ => fail_transfer!(),
            },

            State::BeginBlockStatement => match from {
                Some(State::BeginStatement { push_stat }) => {
                    parser.begin_block_statement(push_stat);
                    Ok(())
                }
                _ => fail_transfer!(),
            },
            State::ContinueBlockStatement { block } => match from {
                Some(State::BeginStatement { .. })
                | Some(State::EndStatement)
                | Some(State::BeginBlockStatement { .. }) => {
                    parser.continue_block_statement(block);
                    Ok(())
                }
                _ => fail_transfer!(),
            },
            State::EndBlockStatement => match from {
                Some(State::ContinueBlockStatement { .. }) => {
                    parser.end_block_statement();
                    Ok(())
                }
                _ => fail_transfer!(),
            },

            State::BeginVariableDeclaration => match from {
                Some(State::BeginStatement { push_stat }) => {
                    parser.begin_variable_declaration(push_stat);
                    Ok(())
                }
                _ => fail_transfer!(),
            },
            State::EndVariableDeclaration => match from {
                Some(State::BeginVariableDeclaration { .. })
                | Some(State::EndExpression { .. }) => {
                    parser.end_variable_declaration();
                    Ok(())
                }
                _ => fail_transfer!(),
            },

            State::BeginExpressionStatement => match from {
                Some(State::BeginStatement { push_stat }) => {
                    parser.begin_expression_statement(push_stat);
                    Ok(())
                }
                _ => fail_transfer!(),
            },
            State::EndExpressionStatement => match from {
                Some(State::EndExpression { .. }) => {
                    parser.end_expression_statement();
                    Ok(())
                }
                _ => fail_transfer!(),
            },

            // expressions
            State::BeginExpression {
                push_expr,
                precedence,
            } => match from {
                Some(State::BeginVariableDeclaration { .. })
                | Some(State::BeginExpressionStatement { .. })
                | Some(State::BeginExpression { .. })
                | Some(State::BeginExpressionInfix { .. }) => {
                    parser.begin_expression(push_expr, precedence);
                    Ok(())
                }
                _ => fail_transfer!(),
            },
            State::BeginExpressionInfix { precedence, left } => match from {
                Some(State::BeginExpression { .. })
                | Some(State::EndExpression { .. })
                | Some(State::EndParenExpression { .. })
                | Some(State::EndBlockExpression) => {
                    parser.begin_expression_infix(precedence, left);
                    Ok(())
                }
                _ => fail_transfer!(),
            },
            State::EndExpression { .. } => match from {
                Some(State::BeginExpressionInfix { .. }) => Ok(()),
                _ => fail_transfer!(),
            },

            State::EndParenExpression { precedence } => match from {
                Some(State::EndExpression { expr }) => {
                    parser.end_paren_expression(precedence, expr);
                    Ok(())
                }
                _ => fail_transfer!(),
            },

            State::ContinueBlockExpression { precedence, block } => match from {
                Some(State::BeginExpression { .. })
                | Some(State::EndStatement)
                | Some(State::ContinueBlockExpression { .. }) => {
                    parser.continue_block_expression(precedence, block);
                    Ok(())
                }
                _ => fail_transfer!(),
            },
            State::EndBlockExpression => match from {
                Some(State::ContinueBlockExpression { precedence, block }) => {
                    parser.end_block_expression(precedence, block);
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
    state: Vec<State>,
    ast: AstBuilder<S>,
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
            Token::BraceClose => {}
            _ => {
                self.on_error(&"Expected end of statement", Some(self.peek_location()));
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

    fn parse(&mut self) -> Result<(), ParserError> {
        let mut previous = None;

        while let Some(mut state) = self.pop_state() {
            state.enter(previous, self)?;
            previous = Some(state);
        }

        Ok(())
    }

    fn recover_statements(&mut self) {
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
    }

    // root

    fn begin_statements_root(&mut self) {
        let location = self.peek_location();
        let root = self.ast.push_root(
            Stat::Compound {
                len: RefLen::default(),
            },
            location,
        );
        self.push_state(State::ContinueStatementsRoot { root });
    }

    fn continue_statements_root(&mut self, root: NodeRef) {
        self.recover_statements();

        self.skip_nl();
        match self.peek() {
            Token::Eof => {}
            _ => {
                self.push_state(State::ContinueStatementsRoot { root });
                self.push_state(State::BeginStatement {
                    push_stat: PushStat::StatementRoot(root),
                });
            }
        }
    }

    // statements

    fn begin_statement(&mut self) {
        self.skip_nl();
        match self.peek() {
            Token::Val | Token::Var => {
                self.push_state(State::BeginVariableDeclaration);
            }
            Token::BraceOpen => {
                self.push_state(State::BeginBlockStatement);
            }
            _ => {
                self.push_state(State::BeginExpressionStatement);
            }
        }
    }

    fn begin_block_statement(&mut self, push_stat: PushStat) {
        let brace_location = self.peek_location();
        match self.peek() {
            Token::BraceOpen => {
                self.advance();
            }
            _ => unreachable!("expected '{{'"),
        };

        let block = push_stat.push(
            &mut self.ast,
            Stat::Compound {
                len: RefLen::default(),
            },
            brace_location,
        );
        self.push_state(State::ContinueBlockStatement { block });
    }

    fn continue_block_statement(&mut self, block: NodeRef) {
        self.recover_statements();

        self.skip_nl();
        match self.peek() {
            Token::Eof | Token::BraceClose => {
                self.push_state(State::EndBlockStatement);
            }
            _ => {
                self.push_state(State::ContinueBlockStatement { block });
                self.push_state(State::BeginStatement {
                    push_stat: PushStat::BlockStatement(block),
                });
            }
        }
    }

    fn end_block_statement(&mut self) {
        match self.peek() {
            Token::BraceClose => {
                self.advance();
            }
            _ => {
                self.on_error(&"Expected '}'", Some(self.peek_location()));
            }
        }

        self.push_state(State::EndStatement);
    }

    fn begin_variable_declaration(&mut self, push_stat: PushStat) {
        let var_location = self.peek_location();
        let mutability = match self.peek() {
            Token::Val => Mutability::Immutable,
            Token::Var => Mutability::Mutable,
            _ => unreachable!("expected 'val' or 'var'"),
        };
        self.advance();

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

        let var_decl = push_stat.push(
            &mut self.ast,
            Stat::VarDecl {
                mutability,
                name,
                def: None,
            },
            var_location,
        );

        self.push_state(State::EndVariableDeclaration);
        if let Token::Eq = self.peek() {
            self.advance();
            self.push_state(State::BeginExpression {
                push_expr: PushExpr::VarDeclDef(var_decl),
                precedence: Precedence::root(),
            });
        }
    }

    fn end_variable_declaration(&mut self) {
        self.push_state(State::EndStatement);
        self.end_of_statement();
    }

    fn begin_expression_statement(&mut self, push_stat: PushStat) {
        let expr_location = self.peek_location();
        let expr_stat = push_stat.push(
            &mut self.ast,
            Stat::Expr {
                expr: NodeRef::default(),
            },
            expr_location,
        );

        self.push_state(State::EndExpressionStatement);
        self.push_state(State::BeginExpression {
            push_expr: PushExpr::ExprStat(expr_stat),
            precedence: Precedence::root(),
        });
    }

    fn end_expression_statement(&mut self) {
        self.push_state(State::EndStatement);
        self.end_of_statement();
    }

    // expressions

    fn begin_expression(&mut self, push_expr: PushExpr, precedence: Precedence) {
        self.skip_nl();
        match *self.peek() {
            Token::Return => {
                let return_location = self.peek_location();
                self.advance();
                let left =
                    push_expr.push(&mut self.ast, Expr::Return { right: None }, return_location);

                self.push_state(State::BeginExpressionInfix { precedence, left });
                if self.peek().is_expression() {
                    self.push_state(State::BeginExpression {
                        push_expr: PushExpr::ReturnExprRight(left),
                        precedence: Precedence::root(),
                    })
                }
            }
            Token::BraceOpen => {
                let brace_location = self.peek_location();
                self.advance();
                let block = push_expr.push(
                    &mut self.ast,
                    Expr::Block {
                        stats_len: RefLen::default(),
                        tail_expr: NodeRef::default(),
                    },
                    brace_location,
                );
                self.push_state(State::ContinueBlockExpression { precedence, block });
            }
            Token::ParenOpen => {
                self.advance();
                self.push_state(State::EndParenExpression { precedence });
                self.push_state(State::BeginExpression {
                    push_expr,
                    precedence: Precedence::root(),
                });
            }
            Token::Sub => {
                let op_location = self.peek_location();
                self.advance();

                self.skip_nl();
                match *self.peek() {
                    Token::Integer(v) => {
                        let location = op_location.merge(self.peek_location());
                        self.advance();
                        let left = push_expr.push(&mut self.ast, Expr::Integer(-v), location);
                        self.push_state(State::BeginExpressionInfix { precedence, left });
                    }
                    Token::Float(v) => {
                        let location = op_location.merge(self.peek_location());
                        self.advance();
                        let left = push_expr.push(&mut self.ast, Expr::Float(-v), location);
                        self.push_state(State::BeginExpressionInfix { precedence, left });
                    }
                    _ => {
                        let left = push_expr.push(
                            &mut self.ast,
                            Expr::UnOp {
                                op: UnOp::Neg,
                                right: NodeRef::default(),
                            },
                            op_location,
                        );
                        self.push_state(State::BeginExpression {
                            push_expr: PushExpr::UnOpExprRight(left),
                            precedence: Precedence::Prefix,
                        });
                    }
                }
            }
            Token::Integer(v) => {
                let expr_location = self.peek_location();
                self.advance();
                let left = push_expr.push(&mut self.ast, Expr::Integer(v), expr_location);
                self.push_state(State::BeginExpressionInfix { precedence, left });
            }
            Token::Float(v) => {
                let expr_location = self.peek_location();
                self.advance();
                let left = push_expr.push(&mut self.ast, Expr::Float(v), expr_location);
                self.push_state(State::BeginExpressionInfix { precedence, left });
            }
            Token::Identifier(ref name) => {
                let name = name.clone();
                let var_location = self.peek_location();
                self.advance();

                let var_expr = push_expr.push(
                    &mut self.ast,
                    Expr::Var {
                        name,
                        assignment: None,
                    },
                    var_location,
                );

                self.push_state(State::BeginExpressionInfix {
                    precedence,
                    left: var_expr,
                });
                if let Token::Eq = self.peek() {
                    self.advance();
                    self.push_state(State::BeginExpression {
                        push_expr: PushExpr::VarExprAssignment(var_expr),
                        precedence: Precedence::root(),
                    });
                }
            }
            _ => {
                let location = self.peek_location();
                self.on_error(&"Expected expression", Some(location));

                // keep the state consistent
                let left = push_expr.push(&mut self.ast, Expr::Integer(0), location);
                self.push_state(State::BeginExpressionInfix { precedence, left });
            }
        }
    }

    fn begin_expression_infix(&mut self, precedence: Precedence, left: NodeRef) {
        let next_location = self.peek_location();
        let next_token = self.peek().clone();
        let next_precedence = next_token.precedence();

        if precedence > next_precedence {
            self.push_state(State::EndExpression { expr: left });
            return;
        }

        let mut bin_expr = |op| {
            self.advance();
            let left = self.ast.patch_infix_expr(
                left,
                |left| Expr::BinOp {
                    op,
                    left,
                    right: NodeRef::default(),
                },
                next_location,
            );
            self.push_state(State::BeginExpressionInfix { precedence, left });
            self.push_state(State::BeginExpression {
                push_expr: PushExpr::BinExprRight(left),
                precedence: next_precedence.next_precedence(),
            });
        };

        match next_token {
            Token::Add => bin_expr(BinOp::Add),
            Token::Sub => bin_expr(BinOp::Sub),
            Token::Mul => bin_expr(BinOp::Mul),
            Token::Div => bin_expr(BinOp::Div),
            _ => self.push_state(State::EndExpression { expr: left }),
        }
    }

    fn end_paren_expression(&mut self, precedence: Precedence, expr: NodeRef) {
        match self.peek() {
            Token::ParenClose => {
                self.advance();
            }
            _ => {
                self.on_error(&"Expected ')'", Some(self.peek_location()));
            }
        };
        self.push_state(State::BeginExpressionInfix {
            precedence,
            left: expr,
        });
    }

    fn continue_block_expression(&mut self, precedence: Precedence, block: NodeRef) {
        self.recover_statements();

        self.skip_nl();
        match self.peek() {
            Token::Eof | Token::BraceClose => {
                self.push_state(State::EndBlockExpression);
            }
            _ => {
                self.push_state(State::ContinueBlockExpression { precedence, block });
                self.push_state(State::BeginStatement {
                    push_stat: PushStat::BlockExpression(block),
                });
            }
        }
    }

    fn end_block_expression(&mut self, precedence: Precedence, block: NodeRef) {
        let brace_location = self.peek_location();
        match self.peek() {
            Token::BraceClose => {
                self.advance();
            }
            _ => {
                self.on_error(&"Expected '}'", Some(self.peek_location()));
            }
        }

        if let Err(location) = self.ast.patch_block_expr_tail(block) {
            self.on_error(&"Expected expression", location.or(Some(brace_location)));
        }

        self.push_state(State::BeginExpressionInfix {
            precedence,
            left: block,
        });
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
            Token::BraceOpen => true,
            _ if self.is_expression() => true,
            _ => false,
        }
    }

    fn is_expression(&self) -> bool {
        // don't care
        #[allow(clippy::match_like_matches_macro)]
        match self {
            Token::Return
            | Token::BraceOpen
            | Token::ParenOpen
            | Token::Integer(_)
            | Token::Float(_)
            | Token::Identifier(_) => true,
            _ => false,
        }
    }
}
