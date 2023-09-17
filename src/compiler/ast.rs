use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::marker::PhantomData;
use std::rc::Rc;

use crate::compiler::ast::display::write_ast;
use crate::compiler::lexer::Span;
use crate::compiler::sem_check::Binding;
use crate::compiler::string::CompileString;

pub struct Unchecked;
pub struct Semantics;

// TODO: dedicated AstVisitor struct?
#[derive(Debug)]
pub struct Ast<S, T> {
    nodes: Vec<Node<S>>,
    locations: Vec<Span>,
    bindings: HashMap<NodeRef, Rc<Binding<S>>>,
    state: PhantomData<T>,
}

impl<S: CompileString, T> Display for Ast<S, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write_ast(self, f)
    }
}

impl<S, T> Ast<S, T> {
    pub fn nodes(&self) -> &[Node<S>] {
        &self.nodes
    }

    pub fn locations(&self) -> &[Span] {
        &self.locations
    }
}

impl<S> Ast<S, Unchecked> {
    #[must_use]
    pub fn to_semantics(self, bindings: HashMap<NodeRef, Rc<Binding<S>>>) -> Ast<S, Semantics> {
        Ast {
            nodes: self.nodes,
            locations: self.locations,
            bindings,
            state: PhantomData,
        }
    }
}

impl<S> Ast<S, Semantics> {
    pub fn bindings(&self) -> &HashMap<NodeRef, Rc<Binding<S>>> {
        &self.bindings
    }
}

pub type Ast1<S> = Ast<S, Unchecked>;
pub type Ast2<S> = Ast<S, Semantics>;

#[derive(Debug)]
pub enum Node<S> {
    Root(Root),
    Stat(Stat<S>),
    Expr(Expr<S>),
}

// Assume string type is sized as u64
static_assert_size!(Node<u64>, 24);

impl<S> From<Root> for Node<S> {
    fn from(value: Root) -> Self {
        Node::Root(value)
    }
}

impl<S> From<Stat<S>> for Node<S> {
    fn from(value: Stat<S>) -> Self {
        Node::Stat(value)
    }
}

impl<S> From<Expr<S>> for Node<S> {
    fn from(value: Expr<S>) -> Self {
        Node::Expr(value)
    }
}

#[derive(Debug)]
pub enum Root {
    Statements,
}

#[derive(Debug)]
pub enum Stat<S> {
    Compound {
        len: RefLen,
        last_stat: NodeRef,
    },
    VarDecl {
        mutability: Mutability,
        name: S,
        def: bool,
    },
    If {
        has_else: bool,
    },
    Expr,
}

#[derive(Copy, Clone, Debug)]
pub enum Mutability {
    Immutable,
    Mutable,
}

#[derive(Debug)]
pub enum Expr<S> {
    Null,
    Bool(bool),
    Integer(i64),
    Float(f64),
    String(S),
    Var { name: S, assignment: bool },
    Return { right: bool },
    UnOp { op: UnOp },
    BinOp { op: BinOp, len: u32 },
    Block { len: RefLen, last_stat: NodeRef },
}

#[derive(Copy, Clone, Debug)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Copy, Clone, Default, PartialEq, Eq, Hash, Debug)]
#[repr(transparent)]
pub struct NodeRef(u32);

impl NodeRef {
    #[inline]
    pub fn new(value: usize) -> Self {
        Self(value as u32)
    }

    #[inline]
    pub fn get(self) -> usize {
        self.0 as usize
    }
}

impl std::ops::SubAssign<u32> for NodeRef {
    fn sub_assign(&mut self, rhs: u32) {
        self.0 -= rhs;
    }
}

impl std::ops::Sub<u32> for NodeRef {
    type Output = Self;
    fn sub(self, rhs: u32) -> Self::Output {
        Self(self.0 - rhs)
    }
}

#[derive(Copy, Clone, Default, PartialEq, Eq, Hash, Debug)]
#[repr(transparent)]
pub struct RefLen(u32);

impl RefLen {
    #[inline]
    pub fn get(self) -> usize {
        self.0 as usize
    }
}

impl PartialEq<u32> for RefLen {
    fn eq(&self, other: &u32) -> bool {
        self.0.eq(other)
    }
}

impl PartialOrd<u32> for RefLen {
    fn partial_cmp(&self, other: &u32) -> Option<Ordering> {
        self.0.partial_cmp(other)
    }
}

impl std::ops::Add<u32> for RefLen {
    type Output = Self;
    fn add(self, rhs: u32) -> Self::Output {
        Self(self.0 + rhs)
    }
}

impl std::ops::AddAssign<u32> for RefLen {
    fn add_assign(&mut self, rhs: u32) {
        self.0 += rhs;
    }
}

impl std::ops::SubAssign<u32> for RefLen {
    fn sub_assign(&mut self, rhs: u32) {
        self.0 -= rhs;
    }
}

impl std::ops::Sub<u32> for RefLen {
    type Output = Self;
    fn sub(self, rhs: u32) -> Self::Output {
        Self(self.0 - rhs)
    }
}

pub struct AstBuilder<S> {
    nodes: Vec<Node<S>>,
    locations: Vec<Span>,
}

impl<S: CompileString> AstBuilder<S> {
    pub fn new(capacity: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(capacity),
            locations: Vec::with_capacity(capacity),
        }
    }

    pub fn build(self) -> Ast1<S> {
        assert!(!self.nodes.is_empty(), "Nodes is empty");
        assert!(!self.locations.is_empty(), "Locations is empty");
        assert_eq!(
            self.nodes.len(),
            self.locations.len(),
            "Mismatch between nodes and locations"
        );

        Ast1 {
            nodes: self.nodes,
            locations: self.locations,
            bindings: HashMap::new(),
            state: PhantomData,
        }
    }

    fn push<N: Into<Node<S>>>(&mut self, node: N, location: Span) -> NodeRef {
        let index = self.nodes.len();
        self.nodes.push(node.into());
        self.locations.push(location);
        NodeRef::new(index)
    }

    fn insert<N: Into<Node<S>>>(&mut self, index: NodeRef, node: N, location: Span) -> NodeRef {
        self.nodes.insert(index.get(), node.into());
        self.locations.insert(index.get(), location);
        index
    }

    pub fn push_root<N: Into<Node<S>>>(&mut self, node: N, location: Span) -> NodeRef {
        self.push(node, location)
    }

    // statements

    // TODO: could replace most/all instances of unreachable! with a result?
    // would require having the parser adapt to results everywhere

    pub fn patch_compound_stat(
        &mut self,
        compound_stat: NodeRef,
        node: Stat<S>,
        location: Span,
    ) -> NodeRef {
        let node = self.push(node, location);

        match self.nodes.get_mut(compound_stat.get()) {
            Some(Node::Stat(Stat::Compound { len, last_stat })) => {
                *len += 1;
                *last_stat = node;
            }
            _ => unreachable!("expected Stat::Compound"),
        }

        node
    }

    pub fn patch_var_decl_def(
        &mut self,
        var_decl: NodeRef,
        node: Expr<S>,
        location: Span,
    ) -> NodeRef {
        let node = self.push(node, location);

        match self.nodes.get_mut(var_decl.get()) {
            Some(Node::Stat(Stat::VarDecl { def, .. })) => {
                *def = true;
            }
            _ => unreachable!("expected Stat::VarDecl"),
        }

        node
    }

    pub fn patch_if_stat_cond(
        &mut self,
        _if_stat: NodeRef,
        node: Expr<S>,
        location: Span,
    ) -> NodeRef {
        self.push(node, location)
    }

    pub fn patch_if_stat_body(
        &mut self,
        _if_stat: NodeRef,
        node: Stat<S>,
        location: Span,
    ) -> NodeRef {
        self.push(node, location)
    }

    pub fn patch_if_stat_else_body(
        &mut self,
        if_stat: NodeRef,
        node: Stat<S>,
        location: Span,
    ) -> NodeRef {
        let node = self.push(node, location);

        match self.nodes.get_mut(if_stat.get()) {
            Some(Node::Stat(Stat::If { has_else, .. })) => {
                *has_else = true;
            }
            _ => unreachable!("expected Stat::If"),
        }

        node
    }

    pub fn patch_expr_stat(
        &mut self,
        _expr_stat: NodeRef,
        node: Expr<S>,
        location: Span,
    ) -> NodeRef {
        self.push(node, location)
    }

    // expressions

    pub fn patch_var_expr(&mut self, var_expr: NodeRef, node: Expr<S>, location: Span) -> NodeRef {
        let node = self.push(node, location);

        match self.nodes.get_mut(var_expr.get()) {
            Some(Node::Expr(Expr::Var { assignment, .. })) => {
                *assignment = true;
            }
            _ => unreachable!("expected Expr::Var"),
        }

        node
    }

    pub fn patch_return_expr(
        &mut self,
        return_expr: NodeRef,
        node: Expr<S>,
        location: Span,
    ) -> NodeRef {
        let node = self.push(node, location);

        match self.nodes.get_mut(return_expr.get()) {
            Some(Node::Expr(Expr::Return { right })) => {
                *right = true;
            }
            _ => unreachable!("expected Expr::Return"),
        }

        node
    }

    pub fn patch_un_op_expr(
        &mut self,
        _un_op_expr: NodeRef,
        node: Expr<S>,
        location: Span,
    ) -> NodeRef {
        self.push(node, location)
    }

    pub fn patch_infix_expr<N: Into<Node<S>>>(
        &mut self,
        left: NodeRef,
        node: N,
        location: Span,
    ) -> NodeRef {
        let node = node.into();
        match (self.nodes.get_mut(left.get()), node) {
            (
                Some(Node::Expr(Expr::BinOp {
                    op: op_a,
                    len: len_a,
                })),
                Node::Expr(Expr::BinOp {
                    op: op_b,
                    len: len_b,
                }),
            ) if *op_a == op_b => {
                *len_a += len_b;
                left
            }
            (_, node) => self.insert(left, node, location),
        }
    }

    pub fn patch_bin_op_expr_right(
        &mut self,
        _bin_op_expr: NodeRef,
        node: Expr<S>,
        location: Span,
    ) -> NodeRef {
        self.push(node, location)
    }

    pub fn patch_block_expr_stat(
        &mut self,
        block_expr: NodeRef,
        node: Stat<S>,
        location: Span,
    ) -> NodeRef {
        let node = self.push(node, location);

        match self.nodes.get_mut(block_expr.get()) {
            Some(Node::Expr(Expr::Block { len, last_stat })) => {
                *len += 1;
                *last_stat = node;
            }
            _ => unreachable!("expected Expr::Block"),
        }

        node
    }

    pub fn patch_block_expr_tail(&mut self, block_expr: NodeRef) -> Result<(), Option<Span>> {
        let mut current_block_expr = Some(block_expr);
        let mut num_removed = 0;

        while let Some(mut block_expr) = current_block_expr.take() {
            block_expr -= num_removed;

            let (len, last_stat) = match self.nodes.get(block_expr.get()) {
                Some(Node::Expr(Expr::Block { len, last_stat })) => {
                    if *len == 0 {
                        return Err(self.locations.get(block_expr.get()).copied());
                    }

                    (*len, *last_stat - num_removed)
                }
                _ => unreachable!(),
            };

            match self.nodes.get(last_stat.get()) {
                Some(&Node::Stat(Stat::Compound {
                    len,
                    last_stat: stat,
                })) => {
                    *self.nodes.get_mut(last_stat.get()).unwrap() = Node::Expr(Expr::Block {
                        len,
                        last_stat: stat,
                    });
                    current_block_expr = Some(last_stat);
                }
                Some(Node::Stat(Stat::Expr)) => {
                    // TODO: remove function?
                    self.nodes.remove(last_stat.get());
                    self.locations.remove(last_stat.get());
                    num_removed += 1;
                }
                _ => return Err(self.locations.get(last_stat.get()).copied()),
            }

            match len.get() {
                1 => {
                    // TODO: remove function?
                    self.nodes.remove(block_expr.get());
                    self.locations.remove(block_expr.get());
                    num_removed += 1;
                }
                _ => {
                    match self.nodes.get_mut(block_expr.get()) {
                        Some(Node::Expr(Expr::Block { len, .. })) => {
                            *len -= 1;
                        }
                        _ => unreachable!("Expected Expr::Block"),
                    };
                }
            }
        }

        Ok(())
    }
}

// display

mod display {
    use std::fmt::{Debug, Error, Formatter, Write};
    use std::iter::Peekable;

    use smallvec::{smallvec, SmallVec};

    use crate::compiler::ast::{Ast, BinOp, Expr, Mutability, Node, RefLen, Root, Stat, UnOp};
    use crate::compiler::string::CompileString;
    use crate::pretty_formatter::PrettyFormatter;

    pub fn write_ast<S: CompileString, T>(
        ast: &Ast<S, T>,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        let mut pretty_formatter = AstPrettyPrinter {
            f: PrettyFormatter::new(f),
            nodes: ast.nodes.iter().peekable(),
            state: smallvec![],
        };
        pretty_formatter.push_state(State::EnterRoot);
        match pretty_formatter.visit() {
            Ok(_) => Ok(()),
            Err(PrinterErr::ExpectedNode) => {
                write!(f, "<invalid ast: expected node>")?;
                Ok(())
            }
            Err(PrinterErr::ExpectedRoot) => {
                write!(f, "<invalid ast: expected root>")?;
                Ok(())
            }
            Err(PrinterErr::ExpectedStat) => {
                write!(f, "<invalid ast: expected statement>")?;
                Ok(())
            }
            Err(PrinterErr::ExpectedExpr) => {
                write!(f, "<invalid ast: expected expression>")?;
                Ok(())
            }
            Err(PrinterErr::WriteErr(err)) => Err(err),
        }
    }

    #[derive(Debug)]
    enum State {
        EnterRoot,

        ContinueCompoundStat(RefLen),
        EnterStatAsBlock,
        EnterStat,
        ContinueIfStatBody(bool),
        ContinueIfStatElse,

        EnterExpr,
        ContinueBinOp(BinOp),
        ExitBinOp,
        ContinueBlockExpr(RefLen),
        ExitBlockExpr,
    }

    type Result<T> = std::result::Result<T, PrinterErr>;

    enum PrinterErr {
        ExpectedNode,
        ExpectedRoot,
        ExpectedStat,
        ExpectedExpr,
        WriteErr(Error),
    }

    impl From<Error> for PrinterErr {
        fn from(value: Error) -> Self {
            Self::WriteErr(value)
        }
    }

    pub struct AstPrettyPrinter<'formatter, W, I: Iterator> {
        f: PrettyFormatter<'formatter, W>,
        nodes: Peekable<I>,
        state: SmallVec<[State; 16]>,
    }

    impl<W: Write, I: Iterator> Write for AstPrettyPrinter<'_, W, I> {
        fn write_str(&mut self, s: &str) -> std::fmt::Result {
            self.f.write_str(s)
        }
    }

    impl<'formatter, 'ast, W, I, S> AstPrettyPrinter<'formatter, W, I>
    where
        W: Write,
        I: Iterator<Item = &'ast Node<S>>,
        S: CompileString + 'ast,
    {
        #[inline]
        fn indent(&mut self) {
            self.f.indent();
        }

        #[inline]
        fn unindent(&mut self) {
            self.f.unindent();
        }

        #[inline]
        fn next(&mut self) -> Result<&'ast Node<S>> {
            self.nodes.next().ok_or(PrinterErr::ExpectedNode)
        }

        #[inline]
        fn next_root(&mut self) -> Result<&'ast Root> {
            match self.next()? {
                Node::Root(node) => Ok(node),
                _ => Err(PrinterErr::ExpectedRoot),
            }
        }

        #[inline]
        fn advance(&mut self) {
            self.nodes.next();
        }

        #[inline]
        fn peek_stat(&mut self) -> Result<&'ast Stat<S>> {
            match self.nodes.peek().ok_or(PrinterErr::ExpectedNode)? {
                Node::Stat(node) => Ok(node),
                _ => Err(PrinterErr::ExpectedStat),
            }
        }

        #[inline]
        fn next_stat(&mut self) -> Result<&'ast Stat<S>> {
            match self.next()? {
                Node::Stat(node) => Ok(node),
                _ => Err(PrinterErr::ExpectedStat),
            }
        }

        #[inline]
        fn next_expr(&mut self) -> Result<&'ast Expr<S>> {
            match self.next()? {
                Node::Expr(node) => Ok(node),
                _ => Err(PrinterErr::ExpectedExpr),
            }
        }

        #[inline]
        fn push_state(&mut self, state: State) {
            self.state.push(state);
        }

        #[inline]
        fn pop_state(&mut self) -> Option<State> {
            self.state.pop()
        }

        fn visit(mut self) -> Result<()> {
            while let Some(state) = self.pop_state() {
                match state {
                    State::EnterRoot => match self.next_root()? {
                        Root::Statements => {
                            self.push_state(State::EnterStat);
                        }
                    },
                    State::ContinueCompoundStat(len) => match len.get() {
                        0 => {
                            self.unindent();
                            writeln!(self)?;
                            write!(self, "}}")?;
                        }
                        _ => {
                            self.push_state(State::ContinueCompoundStat(len - 1));
                            self.push_state(State::EnterStat);
                        }
                    },
                    State::EnterStatAsBlock => {
                        write!(self, "{{")?;
                        self.indent();
                        match *self.peek_stat()? {
                            Stat::Compound { len, .. } => {
                                self.advance();
                                self.push_state(State::ContinueCompoundStat(len));
                            }
                            _ => {
                                self.push_state(State::ContinueCompoundStat(RefLen(1)));
                            }
                        }
                    }
                    State::EnterStat => {
                        writeln!(self)?;
                        match *self.next_stat()? {
                            Stat::Compound { len, .. } => {
                                write!(self, "{{")?;
                                self.indent();
                                self.push_state(State::ContinueCompoundStat(len));
                            }
                            Stat::VarDecl {
                                mutability,
                                ref name,
                                def,
                            } => {
                                match mutability {
                                    Mutability::Immutable => write!(self, "val {}", name)?,
                                    Mutability::Mutable => write!(self, "var {}", name)?,
                                };
                                if def {
                                    write!(self, " = ")?;
                                    self.push_state(State::EnterExpr);
                                }
                            }
                            Stat::If { has_else } => {
                                write!(self, "if (")?;
                                self.push_state(State::ContinueIfStatBody(has_else));
                                self.push_state(State::EnterExpr);
                            }
                            Stat::Expr => {
                                self.push_state(State::EnterExpr);
                            }
                        }
                    }
                    State::ContinueIfStatBody(has_else) => {
                        write!(self, ") ")?;
                        if has_else {
                            self.push_state(State::ContinueIfStatElse);
                        }
                        self.push_state(State::EnterStatAsBlock);
                    }
                    State::ContinueIfStatElse => {
                        write!(self, " else ")?;
                        self.push_state(State::EnterStatAsBlock);
                    }
                    State::EnterExpr => match *self.next_expr()? {
                        Expr::Null => write!(self, "null")?,
                        Expr::Bool(true) => write!(self, "true")?,
                        Expr::Bool(false) => write!(self, "false")?,
                        Expr::Integer(v) => write!(self, "{}", v)?,
                        Expr::Float(v) => write!(self, "{}", v)?,
                        Expr::String(ref v) => write!(self, "{:?}", v)?,
                        Expr::Var {
                            ref name,
                            assignment,
                        } => {
                            write!(self, "{}", name)?;
                            if assignment {
                                write!(self, " = ")?;
                                self.push_state(State::EnterExpr);
                            }
                        }
                        Expr::Return { right } => {
                            write!(self, "return")?;
                            if right {
                                write!(self, " ")?;
                                self.push_state(State::EnterExpr);
                            }
                        }
                        Expr::UnOp { op } => {
                            match op {
                                UnOp::Neg => write!(self, "-")?,
                                UnOp::Not => write!(self, "!")?,
                            };
                            self.push_state(State::EnterExpr);
                        }
                        Expr::BinOp { op, len } => {
                            for _ in 0..len {
                                write!(self, "(")?;
                                self.push_state(State::ContinueBinOp(op));
                            }
                            self.push_state(State::EnterExpr);
                        }
                        Expr::Block { len, .. } => {
                            write!(self, "{{")?;
                            self.indent();
                            self.push_state(State::ContinueBlockExpr(len));
                        }
                    },
                    State::ContinueBinOp(op) => {
                        match op {
                            BinOp::Add => write!(self, " + ")?,
                            BinOp::Sub => write!(self, " - ")?,
                            BinOp::Mul => write!(self, " * ")?,
                            BinOp::Div => write!(self, " / ")?,
                        };
                        self.push_state(State::ExitBinOp);
                        self.push_state(State::EnterExpr);
                    }
                    State::ExitBinOp => {
                        write!(self, ")")?;
                    }
                    State::ContinueBlockExpr(len) => match len.get() {
                        0 => {
                            writeln!(self)?;
                            self.push_state(State::ExitBlockExpr);
                            self.push_state(State::EnterExpr);
                        }
                        _ => {
                            self.push_state(State::ContinueBlockExpr(len - 1));
                            self.push_state(State::EnterStat);
                        }
                    },
                    State::ExitBlockExpr => {
                        self.unindent();
                        writeln!(self)?;
                        write!(self, "}}")?;
                    }
                };
            }

            Ok(())
        }
    }
}
