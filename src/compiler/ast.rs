use std::cmp::Ordering;
use std::fmt::{Display, Error, Formatter, Write};

use crate::compiler::lexer::Span;
use crate::pretty_formatter::PrettyFormatter;

#[derive(Debug)]
pub struct Ast<S> {
    pub nodes: Box<[Node<S>]>,
    pub refs: Box<[NodeRef]>,
    pub locations: Box<[Span]>,
}

impl<S: Display> Display for Ast<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut pretty_printer = AstPrettyPrinter::new(self, f);

        let root = NodeRef(0);
        pretty_printer.push_state(State::EnterStat(root));

        match pretty_printer.visit() {
            Ok(_) => Ok(()),
            Err(PrinterErr::UnexpectedNode) => {
                write!(f, "<error occurred while visiting ast>")?;
                Ok(())
            }
            Err(PrinterErr::WriteErr(err)) => Err(err),
        }
    }
}

#[derive(Debug)]
pub enum Node<S> {
    Stat(Stat<S>),
    Expr(Expr<S>),
}

// Assume string type is sized as u64
static_assert_size!(Node<u64>, 32);

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
pub enum Stat<S> {
    Compound {
        len: RefLen,
    },
    VarDecl {
        mutability: Mutability,
        name: S,
        def: Option<NodeRef>,
    },
    Expr {
        expr: NodeRef,
    },
}

#[derive(Copy, Clone, Debug)]
pub enum Mutability {
    Immutable,
    Mutable,
}

#[derive(Debug)]
pub enum Expr<S> {
    Integer(i64),
    Float(f64),
    Var {
        name: S,
        assignment: Option<NodeRef>,
    },
    Return {
        right: Option<NodeRef>,
    },
    UnOp {
        op: UnOp,
        right: NodeRef,
    },
    BinOp {
        op: BinOp,
        left: NodeRef,
        right: NodeRef,
    },
    Block {
        stats_len: RefLen,
        tail_expr: NodeRef,
    },
}

#[derive(Copy, Clone, Debug)]
pub enum UnOp {
    Neg,
}

#[derive(Copy, Clone, Debug)]
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
        self.0 += rhs
    }
}

impl std::ops::SubAssign<u32> for RefLen {
    fn sub_assign(&mut self, rhs: u32) {
        self.0 -= rhs
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
    refs: Vec<NodeRef>,
    locations: Vec<Span>,
}

impl<S> AstBuilder<S> {
    pub fn new(capacity: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(capacity),
            refs: Vec::with_capacity(capacity),
            locations: Vec::with_capacity(capacity),
        }
    }

    pub fn build(self) -> Ast<S> {
        assert!(!self.nodes.is_empty(), "Nodes is empty");
        assert!(!self.locations.is_empty(), "Locations is empty");
        assert_eq!(
            self.nodes.len(),
            self.locations.len(),
            "Mismatch between nodes and locations"
        );

        Ast {
            nodes: self.nodes.into_boxed_slice(),
            refs: self.refs.into_boxed_slice(),
            locations: self.locations.into_boxed_slice(),
        }
    }

    fn push<N: Into<Node<S>>>(&mut self, node: N, location: Span) -> NodeRef {
        let index = self.nodes.len();
        self.nodes.push(node.into());
        self.locations.push(location);
        NodeRef::new(index)
    }

    fn swap(&mut self, a: &mut NodeRef, b: &mut NodeRef) {
        let a_index = a.get();
        let b_index = b.get();
        self.nodes.swap(a_index, b_index);
        self.locations.swap(a_index, b_index);
        std::mem::swap(a, b);
    }

    // statements

    pub fn push_root<N: Into<Node<S>>>(&mut self, node: N, location: Span) -> NodeRef {
        self.push(node, location)
    }

    pub fn patch_compound_stat(
        &mut self,
        compound_stat: NodeRef,
        node: Stat<S>,
        location: Span,
    ) -> NodeRef {
        let node = self.push(node, location);
        self.refs.push(node);

        match self.nodes.get_mut(compound_stat.get()) {
            Some(Node::Stat(Stat::Compound { len })) => {
                *len += 1;
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
                *def = Some(node);
            }
            _ => unreachable!("expected Stat::VarDecl"),
        }

        node
    }

    pub fn patch_expr_stat(
        &mut self,
        expr_stat: NodeRef,
        node: Expr<S>,
        location: Span,
    ) -> NodeRef {
        let node = self.push(node, location);

        match self.nodes.get_mut(expr_stat.get()) {
            Some(Node::Stat(Stat::Expr { expr, .. })) => {
                *expr = node;
            }
            _ => unreachable!("expected Stat::Expr"),
        }

        node
    }

    // expressions

    pub fn patch_var_expr(&mut self, var_expr: NodeRef, node: Expr<S>, location: Span) -> NodeRef {
        let node = self.push(node, location);

        match self.nodes.get_mut(var_expr.get()) {
            Some(Node::Expr(Expr::Var { assignment, .. })) => {
                *assignment = Some(node);
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
                *right = Some(node);
            }
            _ => unreachable!("expected Expr::Return"),
        }

        node
    }

    pub fn patch_un_op_expr(
        &mut self,
        un_op_expr: NodeRef,
        node: Expr<S>,
        location: Span,
    ) -> NodeRef {
        let node = self.push(node, location);

        match self.nodes.get_mut(un_op_expr.get()) {
            Some(Node::Expr(Expr::UnOp { right, .. })) => {
                *right = node;
            }
            _ => unreachable!("expected Expr::UnOp"),
        }

        node
    }

    pub fn patch_infix_expr<F: FnOnce(NodeRef) -> Expr<S>>(
        &mut self,
        mut left: NodeRef,
        f: F,
        location: Span,
    ) -> NodeRef {
        let mut node = self.push(Expr::Integer(0), location);

        self.swap(&mut left, &mut node);
        match self.nodes.get_mut(node.get()) {
            Some(Node::Expr(bin)) => {
                *bin = f(left);
            }
            _ => unreachable!("missing node"),
        }

        node
    }

    pub fn patch_bin_op_expr_right(
        &mut self,
        bin_op_expr: NodeRef,
        node: Expr<S>,
        location: Span,
    ) -> NodeRef {
        let node = self.push(node, location);

        match self.nodes.get_mut(bin_op_expr.get()) {
            Some(Node::Expr(Expr::BinOp { right, .. })) => {
                *right = node;
            }
            _ => unreachable!("expected Expr::BinOp"),
        }

        node
    }

    pub fn patch_block_expr_stat(
        &mut self,
        block_expr: NodeRef,
        node: Stat<S>,
        location: Span,
    ) -> NodeRef {
        let node = self.push(node, location);
        self.refs.push(node);

        match self.nodes.get_mut(block_expr.get()) {
            Some(Node::Expr(Expr::Block { stats_len, .. })) => {
                *stats_len += 1;
            }
            _ => unreachable!("expected Expr::Block"),
        }

        node
    }

    pub fn patch_block_expr_tail(
        &mut self,
        block_expr: NodeRef,
    ) -> std::result::Result<(), Option<Span>> {
        match self.nodes.get(block_expr.get()) {
            Some(Node::Expr(Expr::Block { stats_len, .. })) => {
                if *stats_len == 0 {
                    return Err(None);
                }
            }
            _ => unreachable!("expected Expr::Block"),
        }

        let last_stat = match self.refs.pop() {
            Some(last_stat) => last_stat,
            None => return Err(None),
        };

        let expr = match self.nodes.get_mut(last_stat.get()) {
            Some(Node::Stat(Stat::Expr { expr })) => *expr,
            _ => return Err(self.locations.get(last_stat.get()).copied()),
        };

        match self.nodes.get_mut(block_expr.get()) {
            Some(Node::Expr(Expr::Block {
                stats_len,
                tail_expr,
            })) => {
                *stats_len -= 1;
                *tail_expr = expr;
            }
            _ => unreachable!("expected Expr::Block"),
        }

        Ok(())
    }
}

// display

struct AstPrettyPrinter<'formatter, 'ast, S> {
    f: PrettyFormatter<'formatter>,
    nodes: &'ast [Node<S>],
    refs: &'ast [NodeRef],
    ref_cursor: usize,
    state: Vec<State>,
}

impl<S> Write for AstPrettyPrinter<'_, '_, S> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.f.write_str(s)
    }
}

#[derive(Debug)]
enum State {
    EnterStat(NodeRef),
    ContinueCompoundStat(RefLen),
    EnterExpr(NodeRef),
    ContinueBinExpr(BinOp),
    EndBinExpr,
    ContinueBlockExpr(RefLen),
    EnterTailExpr(NodeRef),
    EndBlockExpr,
}

type Result<T> = std::result::Result<T, PrinterErr>;

enum PrinterErr {
    UnexpectedNode,
    WriteErr(Error),
}

impl From<Error> for PrinterErr {
    fn from(value: Error) -> Self {
        Self::WriteErr(value)
    }
}

impl<'formatter, 'ast, S: Display> AstPrettyPrinter<'formatter, 'ast, S> {
    fn new(ast: &'ast Ast<S>, f: &'formatter mut Formatter<'_>) -> Self {
        assert!(!ast.nodes.is_empty(), "Ast is empty");

        Self {
            f: PrettyFormatter::new(f),
            nodes: &ast.nodes,
            refs: &ast.refs,
            ref_cursor: 0,
            state: Vec::with_capacity(32),
        }
    }

    #[inline]
    fn indent(&mut self) {
        self.f.indent();
    }

    #[inline]
    fn unindent(&mut self) {
        self.f.unindent();
    }

    fn get_node(&self, index: NodeRef) -> &'ast Node<S> {
        &self.nodes[index.get()]
    }

    fn get_statement(&self, index: NodeRef) -> Result<&'ast Stat<S>> {
        match self.get_node(index) {
            Node::Stat(node) => Ok(node),
            _ => Err(PrinterErr::UnexpectedNode),
        }
    }

    fn get_expression(&self, index: NodeRef) -> Result<&'ast Expr<S>> {
        match self.get_node(index) {
            Node::Expr(node) => Ok(node),
            _ => Err(PrinterErr::UnexpectedNode),
        }
    }

    fn get_next_ref(&mut self) -> NodeRef {
        let index = self.ref_cursor;
        self.ref_cursor += 1;
        self.refs[index]
    }

    fn push_state(&mut self, state: State) {
        self.state.push(state);
    }

    fn pop_state(&mut self) -> Option<State> {
        self.state.pop()
    }

    fn visit(mut self) -> Result<()> {
        while let Some(state) = self.pop_state() {
            match state {
                State::EnterStat(node) => self.enter_stat(node)?,
                State::ContinueCompoundStat(len) => self.continue_compound_stat(len)?,
                State::EnterExpr(node) => self.enter_expr(node)?,
                State::ContinueBinExpr(op) => self.continue_bin_expr(op)?,
                State::EndBinExpr => self.end_bin_expr()?,
                State::ContinueBlockExpr(len) => self.continue_block_expr(len)?,
                State::EnterTailExpr(tail_expr) => self.enter_tail_expr(tail_expr)?,
                State::EndBlockExpr => self.end_block_expr()?,
            };
        }

        Ok(())
    }

    fn enter_stat(&mut self, node: NodeRef) -> Result<()> {
        writeln!(self)?;

        let statement = self.get_statement(node)?;
        match statement {
            Stat::Compound { len } => {
                write!(self, "{{")?;
                self.indent();
                self.push_state(State::ContinueCompoundStat(*len));
            }
            Stat::VarDecl {
                mutability,
                name,
                def,
            } => {
                match mutability {
                    Mutability::Immutable => write!(self, "val {}", name)?,
                    Mutability::Mutable => write!(self, "var {}", name)?,
                };
                if let Some(def) = def {
                    write!(self, " = ")?;
                    self.push_state(State::EnterExpr(*def));
                }
            }
            Stat::Expr { expr } => {
                self.push_state(State::EnterExpr(*expr));
            }
        };

        Ok(())
    }

    fn continue_compound_stat(&mut self, len: RefLen) -> Result<()> {
        match len.get() {
            0 => {
                self.unindent();
                writeln!(self)?;
                write!(self, "}}")?;
            }
            _ => {
                let new_len = len - 1;
                self.push_state(State::ContinueCompoundStat(new_len));
                let next_statement = self.get_next_ref();
                self.push_state(State::EnterStat(next_statement));
            }
        };

        Ok(())
    }

    fn enter_expr(&mut self, node: NodeRef) -> Result<()> {
        let expression = self.get_expression(node)?;
        match *expression {
            Expr::Integer(v) => write!(self, "{}", v)?,
            Expr::Float(v) => write!(self, "{}", v)?,
            Expr::Var {
                name: ref v,
                assignment,
            } => {
                write!(self, "{}", v)?;

                if let Some(assignment) = assignment {
                    write!(self, " = ")?;
                    self.push_state(State::EnterExpr(assignment));
                }
            }
            Expr::Return { right } => {
                write!(self, "return")?;
                if let Some(right) = right {
                    write!(self, " ")?;
                    self.push_state(State::EnterExpr(right));
                }
            }
            Expr::UnOp { op, right } => {
                match op {
                    UnOp::Neg => write!(self, "-")?,
                }
                self.push_state(State::EnterExpr(right));
            }
            Expr::BinOp { op, left, right } => {
                write!(self, "(")?;
                self.push_state(State::EndBinExpr);
                self.push_state(State::EnterExpr(right));
                self.push_state(State::ContinueBinExpr(op));
                self.push_state(State::EnterExpr(left));
            }
            Expr::Block {
                stats_len,
                tail_expr,
            } => {
                write!(self, "{{")?;
                self.indent();
                self.push_state(State::EnterTailExpr(tail_expr));
                self.push_state(State::ContinueBlockExpr(stats_len));
            }
        };

        Ok(())
    }

    fn continue_bin_expr(&mut self, op: BinOp) -> Result<()> {
        match op {
            BinOp::Add => write!(self, " + ")?,
            BinOp::Sub => write!(self, " - ")?,
            BinOp::Mul => write!(self, " * ")?,
            BinOp::Div => write!(self, " / ")?,
        };

        Ok(())
    }

    fn end_bin_expr(&mut self) -> Result<()> {
        write!(self, ")")?;

        Ok(())
    }

    fn continue_block_expr(&mut self, len: RefLen) -> Result<()> {
        if len > 0 {
            let new_len = len - 1;
            self.push_state(State::ContinueBlockExpr(new_len));

            let next_statement = self.get_next_ref();
            self.push_state(State::EnterStat(next_statement));
        }

        Ok(())
    }

    fn enter_tail_expr(&mut self, tail_expr: NodeRef) -> Result<()> {
        writeln!(self)?;
        self.push_state(State::EndBlockExpr);
        self.push_state(State::EnterExpr(tail_expr));

        Ok(())
    }

    fn end_block_expr(&mut self) -> Result<()> {
        self.unindent();
        writeln!(self)?;
        write!(self, "}}")?;

        Ok(())
    }
}
