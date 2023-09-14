use std::borrow::Borrow;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::rc::Rc;

use crate::compiler::ast::{Ast, Expr, Mutability, Node, NodeRef, RefLen, Root, Stat};
use crate::compiler::callback::Callback;
use crate::compiler::lexer::Span;
use crate::compiler::string::CompileString;

#[derive(Debug)]
pub enum SemCheckMsg<S> {
    VariableAlreadyDeclared(S),
    CannotAssignToVal(S),

    VariableNotFound(S),
}

impl<S: CompileString> Display for SemCheckMsg<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            SemCheckMsg::VariableAlreadyDeclared(name) => {
                write!(f, "Variable '{}' already declared in this scope", name)
            }
            SemCheckMsg::CannotAssignToVal(name) => {
                write!(f, "Cannot assign to val '{}'", name)
            }

            SemCheckMsg::VariableNotFound(name) => {
                write!(f, "Variable '{}' not found", name)
            }
        }
    }
}

#[derive(Debug)]
pub enum SemCheckError {
    FailedSemCheck,
    UnexpectedNode,
    ExpectedNode,
    ExpectedRoot,
    ExpectedStat,
    ExpectedExpr,
    MissingContext,
    MissingScope,
}

#[derive(Debug)]
pub struct Ast2<S> {
    pub ast: Ast<S>,
    pub bindings: HashMap<NodeRef, Rc<Binding<S>>>,
}

#[derive(Debug)]
pub struct Binding<S> {
    pub mutability: Mutability,
    pub name: S,
    pub index: Local,
}

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Local(pub u32);

type Result<T> = std::result::Result<T, SemCheckError>;

pub fn sem_check<C: Callback, S: CompileString>(callback: C, ast: Ast<S>) -> Result<Ast2<S>> {
    let mut sem_check = SemCheck {
        callback,
        had_error: false,

        nodes: &ast.nodes,
        locations: &ast.locations,
        cursor: 0,

        state: Vec::with_capacity(32),

        contexts: Vec::with_capacity(1),
        bindings: HashMap::new(),
    };

    sem_check.contexts.push(Context {
        scopes: vec![],
        num_locals: 0,
    });
    sem_check.push_state(State::EnterRoot);

    sem_check.visit()?;

    match sem_check.had_error {
        true => Err(SemCheckError::FailedSemCheck),
        false => {
            let bindings = sem_check.bindings;
            Ok(Ast2 { ast, bindings })
        }
    }
}

#[derive(Debug)]
enum State {
    // root
    EnterRoot,

    // statements
    EnterStat,
    ContinueCompoundStat(RefLen),

    // expressions
    EnterExpr,

    // ..
    EndScope,
    DeclareVar(NodeRef),
}

struct SemCheck<'ast, C, S> {
    callback: C,
    had_error: bool,

    nodes: &'ast [Node<S>],
    locations: &'ast [Span],
    cursor: usize,

    state: Vec<State>,

    contexts: Vec<Context<S>>,
    bindings: HashMap<NodeRef, Rc<Binding<S>>>,
}

impl<'ast, C: Callback, S: CompileString> SemCheck<'ast, C, S> {
    fn on_error(&mut self, message: &dyn Display, source: Option<Span>) {
        self.had_error = true;
        (self.callback)(message, source);
    }

    fn next(&mut self) -> Result<&'ast Node<S>> {
        let node = self.nodes.get(self.cursor);
        self.cursor += 1;
        node.ok_or(SemCheckError::ExpectedNode)
    }

    fn next_root(&mut self) -> Result<&'ast Root> {
        match self.next()? {
            Node::Root(node) => Ok(node),
            _ => Err(SemCheckError::ExpectedRoot),
        }
    }

    fn next_stat(&mut self) -> Result<&'ast Stat<S>> {
        match self.next()? {
            Node::Stat(node) => Ok(node),
            _ => Err(SemCheckError::ExpectedStat),
        }
    }

    fn next_expr(&mut self) -> Result<&'ast Expr<S>> {
        match self.next()? {
            Node::Expr(node) => Ok(node),
            _ => Err(SemCheckError::ExpectedExpr),
        }
    }

    fn get_stat(&mut self, node: NodeRef) -> Result<&'ast Stat<S>> {
        match self.nodes.get(node.get()) {
            Some(Node::Stat(node)) => Ok(node),
            _ => Err(SemCheckError::ExpectedStat),
        }
    }

    fn last_node_as_ref(&self) -> NodeRef {
        NodeRef::new(self.cursor.saturating_sub(1))
    }

    fn get_location(&self, index: NodeRef) -> Span {
        self.locations[index.get()]
    }

    fn push_state(&mut self, state: State) {
        self.state.push(state);
    }

    fn pop_state(&mut self) -> Option<State> {
        self.state.pop()
    }

    fn get_context(&self) -> Result<&Context<S>> {
        self.contexts.last().ok_or(SemCheckError::MissingContext)
    }

    fn get_context_mut(&mut self) -> Result<&mut Context<S>> {
        self.contexts
            .last_mut()
            .ok_or(SemCheckError::MissingContext)
    }

    fn begin_scope(&mut self) -> Result<()> {
        self.get_context_mut()?.scopes.push(Scope::new());
        Ok(())
    }

    fn end_scope(&mut self) -> Result<()> {
        let context = self.get_context_mut()?;
        let scope = context.scopes.pop().ok_or(SemCheckError::MissingScope)?;
        context.num_locals -= scope.len() as u32;
        Ok(())
    }

    fn declare(
        &mut self,
        node: NodeRef,
        mutability: Mutability,
        name: S,
    ) -> Result<Rc<Binding<S>>> {
        let context = self.get_context_mut()?;
        let scope = context
            .scopes
            .last_mut()
            .ok_or(SemCheckError::MissingScope)?;

        match scope.0.entry(name.clone()) {
            Entry::Occupied(entry) => {
                let binding = entry.get().clone();
                self.on_error(
                    &SemCheckMsg::VariableAlreadyDeclared(name),
                    Some(self.get_location(node)),
                );
                Ok(binding)
            }
            Entry::Vacant(entry) => {
                let local = Local(context.num_locals);
                context.num_locals += 1;

                let binding = Rc::new(Binding {
                    mutability,
                    name,
                    index: local,
                });

                entry.insert(binding.clone());
                self.bindings.insert(node, binding.clone());

                Ok(binding)
            }
        }
    }

    fn lookup(&self, name: impl Borrow<[u8]>) -> Option<Rc<Binding<S>>> {
        let name = name.borrow();
        let context = self.get_context().ok()?;

        for scope in context.scopes.iter().rev() {
            if let Some(binding) = scope.0.get(name) {
                return Some(binding.clone());
            }
        }

        None
    }

    // visit

    fn visit(&mut self) -> Result<()> {
        while let Some(state) = self.pop_state() {
            match state {
                State::EnterRoot => match self.next_root()? {
                    Root::Statements => {
                        self.push_state(State::EnterStat);
                    }
                },
                State::EnterStat => match *self.next_stat()? {
                    Stat::Compound { len, .. } => {
                        self.begin_scope()?;
                        self.push_state(State::EndScope);
                        self.push_state(State::ContinueCompoundStat(len));
                    }
                    Stat::VarDecl { def, .. } => {
                        self.push_state(State::DeclareVar(self.last_node_as_ref()));
                        if def {
                            self.push_state(State::EnterExpr);
                        }
                    }
                    Stat::Expr => {
                        self.push_state(State::EnterExpr);
                    }
                },
                State::ContinueCompoundStat(len) => {
                    if len > 0 {
                        self.push_state(State::ContinueCompoundStat(len - 1));
                        self.push_state(State::EnterStat);
                    }
                }
                State::EnterExpr => match *self.next_expr()? {
                    Expr::Null | Expr::Bool(_) | Expr::Integer(_) | Expr::Float(_) => {}
                    Expr::Var {
                        ref name,
                        assignment,
                    } => {
                        let node = self.last_node_as_ref();

                        match self.lookup(name.clone()) {
                            Some(binding) => {
                                match binding.mutability {
                                    Mutability::Immutable if assignment => {
                                        self.on_error(
                                            &SemCheckMsg::CannotAssignToVal(binding.name.clone()),
                                            Some(self.get_location(node)),
                                        );
                                    }
                                    _ => {}
                                }

                                self.bindings.insert(node, binding);
                            }
                            None => {
                                self.on_error(
                                    &SemCheckMsg::VariableNotFound(name.clone()),
                                    Some(self.get_location(node)),
                                );
                            }
                        };

                        if assignment {
                            self.push_state(State::EnterExpr);
                        }
                    }
                    Expr::Return { right } => {
                        if right {
                            self.push_state(State::EnterExpr);
                        }
                    }
                    Expr::UnOp { .. } => {
                        self.push_state(State::EnterExpr);
                    }
                    Expr::BinOp { len, .. } => {
                        for _ in 0..len {
                            self.push_state(State::EnterExpr);
                        }
                        self.push_state(State::EnterExpr);
                    }
                    Expr::Block { len, .. } => {
                        self.push_state(State::EnterExpr);
                        self.push_state(State::ContinueCompoundStat(len));
                    }
                },
                State::EndScope => {
                    self.end_scope()?;
                }
                State::DeclareVar(node) => match self.get_stat(node)? {
                    Stat::VarDecl {
                        mutability, name, ..
                    } => {
                        self.declare(node, *mutability, name.clone())?;
                    }
                    _ => return Err(SemCheckError::UnexpectedNode),
                },
            }
        }

        Ok(())
    }
}

struct Context<S> {
    scopes: Vec<Scope<S>>,
    num_locals: u32,
}

struct Scope<S>(HashMap<S, Rc<Binding<S>>>);

impl<S: CompileString> Scope<S> {
    fn new() -> Self {
        Self(HashMap::new())
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}
