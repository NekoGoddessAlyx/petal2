use std::borrow::Borrow;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::rc::Rc;

use crate::compiler::ast::{Ast, Expr, Node, NodeRef, RefLen, Stat};
use crate::compiler::callback::Callback;
use crate::compiler::lexer::Span;
use crate::compiler::string::CompileString;

#[derive(Debug)]
pub enum SemCheckMsg<S> {
    VariableAlreadyDeclared(S),
    VariableNotFound(S),
}

impl<S: CompileString> Display for SemCheckMsg<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            SemCheckMsg::VariableAlreadyDeclared(name) => {
                write!(f, "Variable '{}' already declared in this scope", name)
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
    BadStateTransfer,
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
        refs: &ast.refs,
        ref_cursor: ast.refs.len(),

        state: Vec::with_capacity(32),

        contexts: Vec::with_capacity(1),
        bindings: HashMap::new(),
    };

    sem_check.contexts.push(Context {
        scopes: vec![],
        num_locals: 0,
    });
    let root = ast.root();
    sem_check.push_state(State::EnterStat(root));

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
    // statements
    EnterStat(NodeRef),
    ExitStat(NodeRef),

    // expressions
    EnterExpr(NodeRef),
}

impl State {
    fn enter<C: Callback, S: CompileString>(
        &mut self,
        from: Option<State>,
        sem_check: &mut SemCheck<C, S>,
    ) -> Result<()> {
        macro_rules! fail_transfer {
            () => {{
                dbg!(&from);
                Err(SemCheckError::BadStateTransfer)
            }};
        }

        match *self {
            // statements
            State::EnterStat(statement) => match from {
                None
                | Some(State::EnterStat(..))
                | Some(State::ExitStat(..))
                | Some(State::EnterExpr(..)) => sem_check.enter_statement(statement),
                //_ => fail_transfer!(),
            },
            State::ExitStat(statement) => match from {
                Some(State::EnterStat(..)) | Some(State::EnterExpr(..)) => {
                    sem_check.exit_statement(statement)
                }
                _ => fail_transfer!(),
            },

            // expressions
            State::EnterExpr(expression) => match from {
                Some(State::EnterStat(..)) | Some(State::EnterExpr(..)) => {
                    sem_check.enter_expression(expression)
                }
                _ => fail_transfer!(),
            },
        }
    }
}

struct SemCheck<'ast, C, S> {
    callback: C,
    had_error: bool,

    nodes: &'ast [Node<S>],
    refs: &'ast [NodeRef],
    ref_cursor: usize,

    state: Vec<State>,

    contexts: Vec<Context<S>>,
    bindings: HashMap<NodeRef, Rc<Binding<S>>>,
}

impl<'ast, C: Callback, S: CompileString> SemCheck<'ast, C, S> {
    fn on_error(&mut self, message: &dyn Display, source: Option<Span>) {
        self.had_error = true;
        (self.callback)(message, source);
    }

    fn get_node(&self, index: NodeRef) -> &'ast Node<S> {
        &self.nodes[index.0 as usize]
    }

    fn get_statement(&self, index: NodeRef) -> Result<&'ast Stat<S>> {
        match self.get_node(index) {
            Node::Stat(node) => Ok(node),
            _ => Err(SemCheckError::UnexpectedNode),
        }
    }

    fn get_expression(&self, index: NodeRef) -> Result<&'ast Expr<S>> {
        match self.get_node(index) {
            Node::Expr(node) => Ok(node),
            _ => Err(SemCheckError::UnexpectedNode),
        }
    }

    fn get_refs(&mut self, len: RefLen) -> &'ast [NodeRef] {
        let end_index = self.ref_cursor;
        let start_index = end_index - len.0 as usize;
        self.ref_cursor = start_index;
        &self.refs[start_index..end_index]
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

    fn declare(&mut self, node: NodeRef, name: S) -> Result<Rc<Binding<S>>> {
        let context = self.get_context_mut()?;
        let scope = context
            .scopes
            .last_mut()
            .ok_or(SemCheckError::MissingScope)?;

        match scope.0.entry(name.clone()) {
            Entry::Occupied(entry) => {
                let binding = entry.get().clone();
                self.on_error(&SemCheckMsg::VariableAlreadyDeclared(name.clone()), None);
                Ok(binding)
            }
            Entry::Vacant(entry) => {
                let local = Local(context.num_locals);
                context.num_locals += 1;

                let binding = Rc::new(Binding {
                    name: name.clone(),
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
        let mut previous = None;
        while let Some(mut state) = self.pop_state() {
            state.enter(previous, self)?;
            previous = Some(state);
        }

        Ok(())
    }

    // statements

    fn enter_statement(&mut self, node: NodeRef) -> Result<()> {
        let statement = self.get_statement(node)?;
        match statement {
            Stat::Compound(len) => {
                self.push_state(State::ExitStat(node));

                self.begin_scope()?;

                let statements = self.get_refs(*len).iter().copied().rev();
                for statement in statements {
                    self.push_state(State::EnterStat(statement));
                }

                Ok(())
            }
            Stat::VarDecl(_, definition) => {
                self.push_state(State::ExitStat(node));

                if let Some(definition) = definition {
                    self.push_state(State::EnterExpr(*definition));
                }

                Ok(())
            }
            Stat::Expr(expression) => {
                self.push_state(State::EnterExpr(*expression));

                Ok(())
            }
        }
    }

    fn exit_statement(&mut self, node: NodeRef) -> Result<()> {
        let statement = self.get_statement(node)?;
        match statement {
            Stat::Compound(_) => {
                self.end_scope()?;

                Ok(())
            }
            Stat::VarDecl(name, _) => {
                self.declare(node, name.clone())?;

                Ok(())
            }
            Stat::Expr(_) => Ok(()),
        }
    }

    // expressions

    fn enter_expression(&mut self, node: NodeRef) -> Result<()> {
        let expression = self.get_expression(node)?;
        match expression {
            Expr::Integer(..) | Expr::Float(..) => Ok(()),
            Expr::Var(var) => {
                match self.lookup(var.clone()) {
                    Some(binding) => {
                        self.bindings.insert(node, binding.clone());
                    }
                    None => {
                        self.on_error(&SemCheckMsg::VariableNotFound(var.clone()), None);
                    }
                }

                Ok(())
            }
            Expr::Return(right) => {
                if let Some(right) = right {
                    self.push_state(State::EnterExpr(*right));
                }

                Ok(())
            }
            Expr::UnOp(_, right) => {
                self.push_state(State::EnterExpr(*right));

                Ok(())
            }
            Expr::BinOp(_, left, right) => {
                self.push_state(State::EnterExpr(*left));
                self.push_state(State::EnterExpr(*right));

                Ok(())
            }
        }
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
