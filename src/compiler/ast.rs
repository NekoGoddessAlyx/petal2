use std::fmt::{Display, Error, Formatter, Write};

use crate::pretty_formatter::PrettyFormatter;

#[derive(Debug)]
pub struct Ast<S> {
    pub nodes: Vec<Node<S>>,
    pub refs: Vec<NodeRef>,
}

impl<S> Ast<S> {
    pub fn new(capacity: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(capacity),
            refs: Vec::with_capacity(capacity / 2),
        }
    }

    pub fn push_node<N: Into<Node<S>>>(&mut self, node: N) -> NodeRef {
        let index = self.nodes.len();
        self.nodes.push(node.into());
        NodeRef(index as u32)
    }

    pub fn push_ref(&mut self, node_ref: NodeRef) {
        self.refs.push(node_ref);
    }

    pub fn root(&self) -> NodeRef {
        NodeRef(self.nodes.len().saturating_sub(1) as u32)
    }
}

impl<S: Display> Display for Ast<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let pretty_printer = AstPrettyPrinter::new(self, f);
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
    Expr(Expr),
}

impl<S> From<Stat<S>> for Node<S> {
    fn from(value: Stat<S>) -> Self {
        Node::Stat(value)
    }
}

impl<S> From<Expr> for Node<S> {
    fn from(value: Expr) -> Self {
        Node::Expr(value)
    }
}

#[derive(Debug)]
pub enum Stat<S> {
    Compound(RefLen),
    VarDecl(S, Option<NodeRef>),
    Expr(NodeRef),
}

#[derive(Debug)]
pub enum Expr {
    Integer(i64),
    Float(f64),
    UnOp(UnOp, NodeRef),
    BinOp(BinOp, NodeRef, NodeRef),
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

#[derive(Copy, Clone, Debug)]
pub struct NodeRef(pub u32);

#[derive(Copy, Clone, Debug)]
pub struct RefLen(pub u32);

// display

struct AstPrettyPrinter<'formatter, 'ast, S> {
    f: PrettyFormatter<'formatter>,
    nodes: &'ast [Node<S>],
    refs: &'ast [NodeRef],
    ref_cursor: usize,
    state: Vec<State<'ast, S>>,
}

impl<S> Write for AstPrettyPrinter<'_, '_, S> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.f.write_str(s)
    }
}

#[derive(Debug)]
enum State<'ast, S> {
    EnterStat(&'ast Stat<S>),
    ExitStat(&'ast Stat<S>),
    EnterExpr(&'ast Expr),
    ContinueBinExpr(BinOp),
    EndBinExpr,
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
            ref_cursor: ast.refs.len(),
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
        &self.nodes[index.0 as usize]
    }

    fn get_statement(&self, index: NodeRef) -> Result<&'ast Stat<S>> {
        match self.get_node(index) {
            Node::Stat(node) => Ok(node),
            _ => Err(PrinterErr::UnexpectedNode),
        }
    }

    fn get_expression(&self, index: NodeRef) -> Result<&'ast Expr> {
        match self.get_node(index) {
            Node::Expr(node) => Ok(node),
            _ => Err(PrinterErr::UnexpectedNode),
        }
    }

    fn get_refs(&mut self, len: RefLen) -> &'ast [NodeRef] {
        let end_index = self.ref_cursor;
        let start_index = end_index - len.0 as usize;
        self.ref_cursor = start_index;
        &self.refs[start_index..end_index]
    }

    fn push_state(&mut self, state: State<'ast, S>) {
        self.state.push(state);
    }

    fn pop_state(&mut self) -> Option<State<'ast, S>> {
        self.state.pop()
    }

    fn visit(mut self) -> Result<()> {
        let root_ref = NodeRef(self.nodes.len().saturating_sub(1) as u32);
        let root = self.get_statement(root_ref)?;

        self.push_state(State::ExitStat(root));
        self.push_state(State::EnterStat(root));

        while let Some(state) = self.pop_state() {
            match state {
                State::EnterStat(node) => self.enter_stat(node)?,
                State::ExitStat(node) => self.exit_stat(node)?,
                State::EnterExpr(node) => self.enter_expr(node)?,
                State::ContinueBinExpr(op) => self.continue_bin_expr(op)?,
                State::EndBinExpr => self.end_bin_expr()?,
            };
        }

        Ok(())
    }

    fn enter_stat(&mut self, node: &Stat<S>) -> Result<()> {
        writeln!(self)?;

        match node {
            Stat::Compound(len) => {
                write!(self, "{{")?;
                self.indent();
                let statements = self.get_refs(*len).iter().rev();
                for statement in statements {
                    let statement = self.get_statement(*statement)?;
                    self.push_state(State::ExitStat(statement));
                    self.push_state(State::EnterStat(statement));
                }
                Ok(())
            }
            Stat::VarDecl(name, def) => {
                write!(self, "var {}", name)?;
                if let Some(def) = def {
                    write!(self, " = ")?;
                    let def = self.get_expression(*def)?;
                    self.push_state(State::EnterExpr(def));
                }
                Ok(())
            }
            Stat::Expr(expr) => {
                let expr = self.get_expression(*expr)?;
                self.push_state(State::EnterExpr(expr));
                Ok(())
            }
        }
    }

    fn exit_stat(&mut self, node: &Stat<S>) -> Result<()> {
        match node {
            Stat::Compound(..) => {
                self.unindent();
                writeln!(self)?;
                write!(self, "}}")?;
                Ok(())
            }
            Stat::VarDecl(..) => Ok(()),
            Stat::Expr(..) => Ok(()),
        }
    }

    fn enter_expr(&mut self, node: &Expr) -> Result<()> {
        match *node {
            Expr::Integer(v) => write!(self, "{}", v)?,
            Expr::Float(v) => write!(self, "{}", v)?,
            Expr::UnOp(op, right) => {
                match op {
                    UnOp::Neg => write!(self, "-")?,
                }
                let right = self.get_expression(right)?;
                self.push_state(State::EnterExpr(right));
            }
            Expr::BinOp(op, left, right) => {
                write!(self, "(")?;
                let right = self.get_expression(right)?;
                self.push_state(State::EndBinExpr);
                self.push_state(State::EnterExpr(right));
                self.push_state(State::ContinueBinExpr(op));
                let left = self.get_expression(left)?;
                self.push_state(State::EnterExpr(left));
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
}
