use std::borrow::Borrow;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter};
use std::iter::zip;
use std::rc::Rc;

use smallvec::{smallvec, SmallVec};
use thiserror::Error;

use crate::compiler::ast::{
    Ast1, Ast1Iterator, Ast2, BinOp, Expr, Mutability, NodeError, NodeRef, RefLen, Root, Stat,
};
use crate::compiler::callback::Callback;
use crate::compiler::lexer::Span;
use crate::compiler::string::CompileString;
use crate::compiler::Diagnostic;
use crate::MessageKind;

#[derive(Debug)]
pub enum SemCheckMsg<S> {
    VariableAlreadyDeclared(S),
    CannotAssignToVal(S),

    VariableNotFound(S),

    TypeError(TypeError<S>),
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

            SemCheckMsg::TypeError(type_error) => {
                write!(f, "{}", type_error)
            }
        }
    }
}

impl<S: CompileString> Diagnostic for SemCheckMsg<S> {
    fn kind(&self) -> MessageKind {
        MessageKind::Error
    }

    fn message(&self) -> &dyn Display {
        self
    }
}

impl<S> From<TypeError<S>> for SemCheckMsg<S> {
    fn from(value: TypeError<S>) -> Self {
        Self::TypeError(value)
    }
}

#[derive(Debug, Error)]
pub enum SemCheckError {
    #[error("SemCheck failed")]
    FailedSemCheck,
    #[error(transparent)]
    NodeError(#[from] NodeError),
    #[error("Context is missing")]
    MissingContext,
    #[error("Scope is missing")]
    MissingScope,
}

#[derive(Debug)]
pub struct Binding<S> {
    pub mutability: Mutability,
    pub name: S,
    pub index: Local,
    pub ty: Type<S>,
}

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Local(pub u32);

type Result<T> = std::result::Result<T, SemCheckError>;

pub fn sem_check<C: Callback, S: CompileString>(callback: C, ast: Ast1<S>) -> Result<Ast2<S>> {
    let mut sem_check = SemCheck {
        callback,
        had_error: false,

        ast: ast.iterator(),

        state: smallvec![],

        contexts: smallvec![],
        bindings: HashMap::new(),
        next_type: 0,
        substitutions: HashMap::new(),
    };

    sem_check.contexts.push(Context {
        scopes: vec![],
        num_locals: 0,
    });
    sem_check.push_state(State::EnterRoot);

    sem_check.visit()?;
    println!("SUBSTITUTIONS: {:#?}", sem_check.substitutions);
    let mut bindings = sem_check.bindings.iter().collect::<Vec<_>>();
    bindings.sort_by(|x, y| x.0.get().cmp(&y.0.get()));
    let bindings = bindings
        .into_iter()
        .map(|(_, binding)| (binding.name.to_string(), binding.ty.clone()))
        .collect::<HashSet<_>>();
    println!("BINDINGS {:#?}", bindings);

    match sem_check.had_error {
        true => Err(SemCheckError::FailedSemCheck),
        false => {
            let bindings = sem_check.bindings;
            Ok(ast.into_semantics(bindings))
        }
    }
}

#[derive(Debug)]
enum State<S> {
    // root
    EnterRoot,

    // statements
    EnterStat,
    ContinueCompoundStat(RefLen),

    // expressions
    EnterExpr,
    ExitExpr(Type<S>),

    ExitVarAssignment(Type<S>),
    ContinueBinExpr(BinOp),
    ExitBinExpr(BinOp, Type<S>),

    // ..
    EndScope,
    DeclareVar(NodeRef),
}

struct SemCheck<'ast, C, S> {
    callback: C,
    had_error: bool,

    ast: Ast1Iterator<'ast, S>,

    state: SmallVec<[State<S>; 32]>,

    contexts: SmallVec<[Context<S>; 16]>,
    bindings: HashMap<NodeRef, Rc<Binding<S>>>,

    next_type: u32,
    substitutions: HashMap<TypeVariable, Type<S>>,
}

impl<'ast, C: Callback, S: CompileString> SemCheck<'ast, C, S> {
    fn on_error(&mut self, message: &dyn Diagnostic, source: Option<Span>) {
        self.had_error = true;
        (self.callback)(message, source);
    }

    fn push_state(&mut self, state: State<S>) {
        self.state.push(state);
    }

    fn pop_state(&mut self) -> Option<State<S>> {
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
        ty: Type<S>,
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
                    self.ast.location_of(node),
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
                    ty,
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

    fn next_type(&mut self) -> Type<S> {
        let index = self.next_type;
        self.next_type += 1;
        Type::Variable(TypeVariable(index))
    }

    // visit

    fn visit(&mut self) -> Result<()> {
        let mut previous = None;
        while let Some(state) = self.pop_state() {
            match state {
                State::EnterRoot => match self.ast.next_root()? {
                    Root::Statements => {
                        self.push_state(State::EnterStat);
                    }
                },
                State::EnterStat => match *self.ast.next_stat()? {
                    Stat::Compound { len, .. } => {
                        self.begin_scope()?;
                        self.push_state(State::EndScope);
                        self.push_state(State::ContinueCompoundStat(len));
                    }
                    Stat::VarDecl { def, .. } => {
                        self.push_state(State::DeclareVar(self.ast.previous_node()));
                        if def {
                            self.push_state(State::EnterExpr);
                        }
                    }
                    Stat::If { has_else } => {
                        if has_else {
                            self.push_state(State::EnterStat);
                        }
                        self.push_state(State::EnterStat);
                        self.push_state(State::EnterExpr);
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
                State::EnterExpr => match *self.ast.next_expr()? {
                    Expr::Null => {
                        let ty = Type::Null;
                        // let ty = self.next_type();
                        self.push_state(State::ExitExpr(ty));
                    }
                    Expr::Bool(_) => {
                        let ty = Type::Boolean(false);
                        self.push_state(State::ExitExpr(ty));
                    }
                    Expr::Integer(_) => {
                        let ty = Type::Integer(false);
                        self.push_state(State::ExitExpr(ty));
                    }
                    Expr::Float(_) => {
                        let ty = Type::Float(false);
                        self.push_state(State::ExitExpr(ty));
                    }
                    Expr::String(_) => {
                        let ty = Type::String(false);
                        self.push_state(State::ExitExpr(ty));
                    }
                    Expr::Var {
                        ref name,
                        assignment,
                    } => {
                        let node = self.ast.previous_node();

                        let ty = match self.lookup(name.clone()) {
                            Some(binding) => {
                                match binding.mutability {
                                    Mutability::Immutable if assignment => {
                                        self.on_error(
                                            &SemCheckMsg::CannotAssignToVal(binding.name.clone()),
                                            self.ast.location_of(node),
                                        );
                                    }
                                    _ => {}
                                }

                                self.bindings.insert(node, binding.clone());

                                binding.ty.clone()
                            }
                            None => {
                                self.on_error(
                                    &SemCheckMsg::VariableNotFound(name.clone()),
                                    self.ast.location_of(node),
                                );

                                Type::Dynamic(true)
                                // self.next_type()
                            }
                        };

                        match assignment {
                            true => {
                                self.push_state(State::ExitVarAssignment(ty));
                                self.push_state(State::EnterExpr);
                            }
                            false => {
                                self.push_state(State::ExitExpr(ty));
                            }
                        }
                    }
                    Expr::Return { right } => match right {
                        true => {
                            self.push_state(State::EnterExpr);
                        }
                        false => {
                            let ty = Type::Null;
                            self.push_state(State::ExitExpr(ty));
                        }
                    },
                    Expr::UnOp { .. } => {
                        self.push_state(State::EnterExpr);
                    }
                    Expr::BinOp { op, len } => {
                        self.push_state(State::ContinueBinExpr(op));
                        for _ in 1..len {
                            self.push_state(State::ContinueBinExpr(op));
                        }
                        self.push_state(State::EnterExpr);
                    }
                    Expr::Block { len, .. } => {
                        self.push_state(State::EnterExpr);
                        self.push_state(State::ContinueCompoundStat(len));
                    }
                },
                State::ExitExpr(..) => {}
                State::ExitVarAssignment(ref left_ty) => match previous {
                    Some(State::ExitExpr(ref right_ty)) => {
                        match unify(left_ty.clone(), right_ty.clone(), &mut self.substitutions) {
                            Ok(_) => {}
                            Err(e) => {
                                self.on_error(&SemCheckMsg::TypeError(e), None);
                                // todo
                            }
                        }
                    }
                    _ => todo!("state error"),
                },
                State::ContinueBinExpr(op) => match previous {
                    Some(State::ExitExpr(ty)) => {
                        self.push_state(State::ExitBinExpr(op, ty));
                        self.push_state(State::EnterExpr);
                    }
                    _ => todo!("state error"),
                },
                State::ExitBinExpr(op, ref left_ty) => match previous {
                    Some(State::ExitExpr(ref right_ty)) => {
                        // todo extract into function
                        let ty = match op {
                            BinOp::Add => match (left_ty, right_ty) {
                                (Type::Dynamic(_), _) => left_ty.clone(),
                                (_, Type::Dynamic(_)) => right_ty.clone(),
                                (Type::Integer(_), Type::Integer(_)) => left_ty.clone(),
                                (Type::Integer(_), Type::Float(_)) => right_ty.clone(),
                                (Type::Float(_), Type::Integer(_))
                                | (Type::Float(_), Type::Float(_)) => left_ty.clone(),
                                (Type::String(_), _) => left_ty.clone(),
                                (_, Type::String(_)) => right_ty.clone(),
                                (_, _) => {
                                    self.on_error(
                                        &SemCheckMsg::TypeError(TypeError::CannotAdd(
                                            left_ty.clone(),
                                            right_ty.clone(),
                                        )),
                                        None,
                                    ); // todo
                                    Type::Dynamic(true)
                                }
                            },
                            BinOp::Sub | BinOp::Mul | BinOp::Div => match (left_ty, right_ty) {
                                (Type::Dynamic(_), _) => left_ty.clone(),
                                (_, Type::Dynamic(_)) => right_ty.clone(),
                                (Type::Integer(_), Type::Integer(_)) => left_ty.clone(),
                                (Type::Integer(_), Type::Float(_)) => right_ty.clone(),
                                (Type::Float(_), Type::Integer(_))
                                | (Type::Float(_), Type::Float(_)) => left_ty.clone(),
                                (_, _) => {
                                    self.on_error(
                                        &SemCheckMsg::TypeError(TypeError::CannotBinOp(
                                            left_ty.clone(),
                                            right_ty.clone(),
                                        )),
                                        None,
                                    ); // todo
                                    Type::Dynamic(true)
                                }
                            },
                        };

                        self.push_state(State::ExitExpr(ty));
                    }
                    _ => todo!("state error"),
                },

                State::EndScope => {
                    self.end_scope()?;
                }
                State::DeclareVar(node) => {
                    let b = if let Some(State::ExitExpr(ty)) = previous {
                        ty
                    } else {
                        Type::Dynamic(true)
                    };
                    // let ty = self.next_type();
                    // let ty = Type::Dynamic(true);
                    let ty = Type::Dynamic(false);
                    match unify(ty.clone(), b, &mut self.substitutions) {
                        Ok(_) => {}
                        Err(e) => {
                            self.on_error(&SemCheckMsg::TypeError(e), None);
                            // todo
                        }
                    }
                    let ty = ty.substitute(&self.substitutions);

                    match self.ast.get_stat_at(node)? {
                        Stat::VarDecl {
                            mutability, name, ..
                        } => {
                            self.declare(node, *mutability, name.clone(), ty)?;
                        }
                        _ => return Err(SemCheckError::NodeError(NodeError::ExpectedStat)),
                    }
                }
            }
            previous = Some(state);
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

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Type<S> {
    Null,
    Dynamic(bool),
    Boolean(bool),
    Integer(bool),
    Float(bool),
    String(bool),
    #[allow(unused)]
    NewType(Rc<NewType<S>>),
    Variable(TypeVariable),
}

impl<S: CompileString> Type<S> {
    fn substitute(&self, substitutions: &HashMap<TypeVariable, Type<S>>) -> Type<S> {
        match self {
            Type::Null => Type::Null,
            Type::Dynamic(nullable) => Type::Dynamic(*nullable),
            Type::Boolean(nullable) => Type::Boolean(*nullable),
            Type::Integer(nullable) => Type::Integer(*nullable),
            Type::Float(nullable) => Type::Float(*nullable),
            Type::String(nullable) => Type::String(*nullable),
            Type::NewType(ty) => Type::NewType(Rc::new(NewType {
                name: ty.name.clone(),
                generics: ty
                    .generics
                    .iter()
                    .map(|t| t.substitute(substitutions))
                    .collect(),
            })),
            Type::Variable(TypeVariable(i)) => match substitutions.get(&TypeVariable(*i)) {
                None => self.clone(),
                Some(t) => t.substitute(substitutions),
            },
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct NewType<S> {
    name: S,
    generics: Vec<Type<S>>,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct TypeVariable(u32);

impl TypeVariable {
    fn occurs_in<S: CompileString>(
        &self,
        ty: Type<S>,
        substitutions: &HashMap<TypeVariable, Type<S>>,
    ) -> bool {
        match ty {
            Type::Null => false,
            Type::Dynamic(_) => false,
            Type::Boolean(_) | Type::Integer(_) | Type::Float(_) | Type::String(_) => false,
            Type::NewType(ty) => {
                for generic in &ty.generics {
                    if self.occurs_in(generic.clone(), substitutions) {
                        return true;
                    }
                }

                false
            }
            Type::Variable(v @ TypeVariable(i)) => {
                if let Some(substitution) = substitutions.get(&v) {
                    if substitution != &Type::Variable(v) {
                        return self.occurs_in(substitution.clone(), substitutions);
                    }
                }

                self.0 == i
            }
        }
    }
}

fn unify<S: CompileString>(
    left: Type<S>,
    right: Type<S>,
    substitutions: &mut HashMap<TypeVariable, Type<S>>,
) -> std::result::Result<(), TypeError<S>> {
    match (&left, &right) {
        (Type::Dynamic(true), _) => Ok(()),
        (
            Type::Dynamic(false),
            Type::Dynamic(false)
            | Type::Boolean(false)
            | Type::Integer(false)
            | Type::Float(false)
            | Type::String(false),
        ) => Ok(()),
        (
            Type::Dynamic(false),
            Type::Dynamic(true)
            | Type::Boolean(true)
            | Type::Integer(true)
            | Type::Float(true)
            | Type::String(true),
        ) => Err(TypeError::TypeNotEqual(left, right)),
        (Type::Boolean(a), Type::Boolean(b))
        | (Type::Integer(a), Type::Integer(b))
        | (Type::Float(a), Type::Float(b))
        | (Type::String(a), Type::String(b)) => match *a || !*b {
            true => Ok(()),
            false => Err(TypeError::TypeNotEqual(left, right)),
        },
        (Type::NewType(ty1), Type::NewType(ty2)) => {
            assert_eq!(ty1.name, ty2.name);
            assert_eq!(ty1.generics.len(), ty2.generics.len());

            for (left, right) in zip(&ty1.generics, &ty2.generics) {
                unify(left.clone(), right.clone(), substitutions)?;
            }

            Ok(())
        }
        (Type::Variable(TypeVariable(a)), Type::Variable(TypeVariable(b))) if a == b => Ok(()),
        (_, &Type::Variable(v @ TypeVariable(..))) => {
            if let Some(substitution) = substitutions.get(&v) {
                unify(left, substitution.clone(), substitutions)?;

                return Ok(());
            }

            if v.occurs_in(left.clone(), substitutions) {
                return Err(TypeError::InfiniteType(v, left));
            }

            substitutions.insert(v, left);

            Ok(())
        }
        (&Type::Variable(v @ TypeVariable(..)), _) => {
            if let Some(substitution) = substitutions.get(&v) {
                unify(right, substitution.clone(), substitutions)?;

                return Ok(());
            }

            if v.occurs_in(right.clone(), substitutions) {
                return Err(TypeError::InfiniteType(v, right));
            }

            substitutions.insert(v, right);

            Ok(())
        }
        (_, _) => Err(TypeError::TypeNotEqual(left, right)),
    }
}

#[derive(Debug)]
pub enum TypeError<S> {
    TypeNotEqual(Type<S>, Type<S>),
    InfiniteType(TypeVariable, Type<S>),
    CannotAdd(Type<S>, Type<S>),
    CannotBinOp(Type<S>, Type<S>),
}

impl<S: CompileString> Display for TypeError<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TypeError::TypeNotEqual(a, b) => {
                write!(f, "Types not equal ({:?}, {:?})", a, b)
            }
            TypeError::InfiniteType(a, b) => {
                write!(f, "Infinite type ({:?}, {:?})", a, b)
            }
            TypeError::CannotAdd(a, b) => {
                write!(f, "Cannot add types ({:?}, {:?})", a, b)
            }
            TypeError::CannotBinOp(a, b) => {
                // TODO op
                write!(f, "Cannot perform operation (?) ({:?}, {:?})", a, b)
            }
        }
    }
}
