use std::borrow::Borrow;
use std::cell::Cell;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::iter::zip;
use std::rc::Rc;

use smallvec::{smallvec, SmallVec};
use thiserror::Error;

use crate::compiler::ast::{
    Ast1, Ast1Iterator, Ast2, BinOp, Expr, Mutability, NodeError, NodeRef, RefLen, Root, Stat,
    TypeSpec, UnOp,
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
    VariableNotInitialized(S),

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
            SemCheckMsg::VariableNotInitialized(name) => {
                write!(f, "Variable '{}' is not initialized", name)
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
    #[error("Invalid state transition")]
    BadTransition,
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
    initialized: Cell<bool>,
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
    ExitUnExpr(UnOp),
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

    #[allow(unused)]
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
        initialized: bool,
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
                    initialized: Cell::new(initialized),
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

    #[allow(unused)]
    fn next_type(&mut self) -> Type<S> {
        let index = self.next_type;
        self.next_type += 1;
        Type::Variable(TypeVariable(index))
    }

    // visit

    fn visit(&mut self) -> Result<()> {
        let mut previous = None;

        macro_rules! fail_transfer {
            () => {{
                dbg!(&previous);
                return Err(SemCheckError::BadTransition);
            }};
        }

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
                                self.bindings.insert(node, binding.clone());

                                let initialized = binding.initialized.get();
                                let mutability = binding.mutability;
                                match (initialized, assignment, mutability) {
                                    (true, true, Mutability::Immutable) => {
                                        self.on_error(
                                            &SemCheckMsg::CannotAssignToVal(binding.name.clone()),
                                            self.ast.location_of(node),
                                        );
                                    }
                                    (false, true, _) => {
                                        binding.initialized.set(true);
                                    }
                                    (false, false, _) => {
                                        self.on_error(
                                            &SemCheckMsg::VariableNotInitialized(
                                                binding.name.clone(),
                                            ),
                                            self.ast.location_of(node),
                                        );
                                    }
                                    _ => {}
                                }

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
                    Expr::Return { right } => {
                        let ty = Type::Never;
                        self.push_state(State::ExitExpr(ty));
                        if right {
                            self.push_state(State::EnterExpr);
                        }
                    }
                    Expr::UnOp { op } => {
                        self.push_state(State::ExitUnExpr(op));
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
                        self.push_state(State::ExitExpr(left_ty.clone()));
                    }
                    _ => fail_transfer!(),
                },
                State::ExitUnExpr(op) => match previous {
                    Some(State::ExitExpr(ref right_ty)) => {
                        let ty = self.un_op_type(op, right_ty);
                        self.push_state(State::ExitExpr(ty));
                    }
                    _ => fail_transfer!(),
                },
                State::ContinueBinExpr(op) => match previous {
                    Some(State::ExitExpr(ty)) => {
                        self.push_state(State::ExitBinExpr(op, ty));
                        self.push_state(State::EnterExpr);
                    }
                    _ => fail_transfer!(),
                },
                State::ExitBinExpr(op, ref left_ty) => match previous {
                    Some(State::ExitExpr(ref right_ty)) => {
                        let ty = self.bin_op_type(op, left_ty, right_ty);
                        self.push_state(State::ExitExpr(ty));
                    }
                    _ => fail_transfer!(),
                },

                State::EndScope => {
                    self.end_scope()?;
                }
                State::DeclareVar(node) => {
                    let mut def_ty = if let Some(State::ExitExpr(ty)) = previous {
                        Some(ty)
                    } else {
                        None
                    };

                    let ty = match self.ast.get_stat_at(node)? {
                        Stat::VarDecl { mutability, ty, .. } => {
                            if let Mutability::Mutable = mutability {
                                if let None = def_ty {
                                    if ty.is_nullable() {
                                        def_ty = Some(Type::Null);
                                    }
                                }
                            }

                            match self.get_ty(ty) {
                                Ok(ty) => ty,
                                Err(e) => {
                                    // todo location
                                    self.on_error(&SemCheckMsg::TypeError(e), None);
                                    Type::Dynamic(ty.is_nullable())
                                }
                            }
                        }
                        _ => return Err(SemCheckError::NodeError(NodeError::ExpectedStat)),
                    };
                    // let ty = self.next_type();
                    // let ty = Type::Dynamic(true);
                    // let ty = Type::Dynamic(false);

                    let initialized = def_ty.is_some();
                    if let Some(b) = def_ty {
                        match unify(ty.clone(), b, &mut self.substitutions) {
                            Ok(_) => {}
                            Err(e) => {
                                self.on_error(&SemCheckMsg::TypeError(e), None);
                                // todo
                            }
                        }
                    }
                    let ty = ty.substitute(&self.substitutions);

                    match self.ast.get_stat_at(node)? {
                        Stat::VarDecl {
                            mutability, name, ..
                        } => {
                            self.declare(node, *mutability, name.clone(), ty, initialized)?;
                        }
                        _ => return Err(SemCheckError::NodeError(NodeError::ExpectedStat)),
                    }
                }
            }
            previous = Some(state);
        }

        Ok(())
    }

    fn get_ty(&mut self, ty: &TypeSpec<S>) -> std::result::Result<Type<S>, TypeError<S>> {
        Ok(match ty {
            TypeSpec::Dyn(n) => Type::Dynamic(*n),
            TypeSpec::Ty(ty, n) => match ty.as_ref() {
                b"Null" => Type::Null,
                b"Bool" => Type::Boolean(*n),
                b"Int" => Type::Integer(*n),
                b"Float" => Type::Float(*n),
                b"String" => Type::String(*n),
                _ => return Err(TypeError::UnknownType(ty.clone())),
            },
        })
    }

    fn bin_op_type(&mut self, op: BinOp, left_ty: &Type<S>, right_ty: &Type<S>) -> Type<S> {
        match op {
            BinOp::Add => match (left_ty, right_ty) {
                (Type::Dynamic(n), _) | (_, Type::Dynamic(n)) => Type::Dynamic(*n),
                (Type::Never, _) | (_, Type::Never) => Type::Never,
                (Type::Integer(false), Type::Integer(false)) => Type::Integer(false),
                (Type::Integer(false), Type::Float(false))
                | (Type::Float(false), Type::Integer(false))
                | (Type::Float(false), Type::Float(false)) => Type::Float(false),
                (Type::String(false), _) | (_, Type::String(false)) => Type::String(false),
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
                (Type::Dynamic(n), _) | (_, Type::Dynamic(n)) => Type::Dynamic(*n),
                (Type::Never, _) | (_, Type::Never) => Type::Never,
                (Type::Integer(false), Type::Integer(false)) => Type::Integer(false),
                (Type::Integer(false), Type::Float(false))
                | (Type::Float(false), Type::Integer(false))
                | (Type::Float(false), Type::Float(false)) => Type::Float(false),
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
        }
    }

    fn un_op_type(&mut self, op: UnOp, right_ty: &Type<S>) -> Type<S> {
        match op {
            UnOp::Neg => match right_ty {
                Type::Dynamic(n) => Type::Dynamic(*n),
                Type::Never => Type::Never,
                Type::Integer(false) => Type::Integer(false),
                Type::Float(false) => Type::Float(false),
                _ => {
                    self.on_error(
                        &SemCheckMsg::TypeError(TypeError::CannotUnOp(right_ty.clone())),
                        None,
                    ); // todo
                    Type::Dynamic(true)
                }
            },
            UnOp::Not => Type::Boolean(false),
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

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Type<S> {
    Dynamic(bool),
    Never,
    Null,
    Boolean(bool),
    Integer(bool),
    Float(bool),
    String(bool),
    #[allow(unused)]
    NewType(Rc<NewType<S>>),
    #[allow(unused)]
    Variable(TypeVariable),
}

impl<S: CompileString> Type<S> {
    fn substitute(&self, substitutions: &HashMap<TypeVariable, Type<S>>) -> Type<S> {
        match self {
            Type::Dynamic(n) => Type::Dynamic(*n),
            Type::Never => Type::Never,
            Type::Null => Type::Null,
            Type::Boolean(n) => Type::Boolean(*n),
            Type::Integer(n) => Type::Integer(*n),
            Type::Float(n) => Type::Float(*n),
            Type::String(n) => Type::String(*n),
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

impl<S: CompileString> Display for Type<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Dynamic(true) => write!(f, "dyn?"),
            Type::Dynamic(false) => write!(f, "dyn"),
            Type::Never => write!(f, "!"),
            Type::Null => write!(f, "Null"),
            Type::Boolean(true) => write!(f, "Bool?"),
            Type::Boolean(false) => write!(f, "Bool"),
            Type::Integer(true) => write!(f, "Int?"),
            Type::Integer(false) => write!(f, "Int"),
            Type::Float(true) => write!(f, "Float?"),
            Type::Float(false) => write!(f, "Float"),
            Type::String(true) => write!(f, "String?"),
            Type::String(false) => write!(f, "String"),
            Type::NewType(_ty) => todo!(),
            Type::Variable(_v) => todo!(),
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
            Type::Dynamic(_)
            | Type::Never
            | Type::Null
            | Type::Boolean(_)
            | Type::Integer(_)
            | Type::Float(_)
            | Type::String(_) => false,
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
            | Type::Null
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
        (_, Type::Never) => Ok(()),
        (_, _) => Err(TypeError::TypeNotEqual(left, right)),
    }
}

#[derive(Debug)]
pub enum TypeError<S> {
    UnknownType(S),
    TypeNotEqual(Type<S>, Type<S>),
    InfiniteType(TypeVariable, Type<S>),
    CannotAdd(Type<S>, Type<S>),
    CannotUnOp(Type<S>),
    CannotBinOp(Type<S>, Type<S>),
}

impl<S: CompileString> Display for TypeError<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TypeError::UnknownType(ty) => {
                write!(f, "Unknown type ({})", ty)
            }
            TypeError::TypeNotEqual(a, b) => {
                write!(f, "Types not equal ({:?}, {:?})", a, b)
            }
            TypeError::InfiniteType(a, b) => {
                write!(f, "Infinite type ({:?}, {:?})", a, b)
            }
            TypeError::CannotAdd(a, b) => {
                write!(f, "Cannot add types ({:?}, {:?})", a, b)
            }
            TypeError::CannotUnOp(a) => {
                // TODO op
                write!(f, "Cannot perform operation (?) ({:?})", a)
            }
            TypeError::CannotBinOp(a, b) => {
                // TODO op
                write!(f, "Cannot perform operation (?) ({:?}, {:?})", a, b)
            }
        }
    }
}
