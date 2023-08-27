use std::fmt::{Display, Error, Formatter, Write};
use std::ops::Index;

use crate::pretty_formatter::PrettyFormatter;

#[derive(Debug)]
pub struct Ast(Vec<Node>);

impl Ast {
    pub(super) fn new(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub(super) fn push<N: Into<Node>>(&mut self, node: N) -> NodeRef {
        let index = self.0.len();
        self.0.push(node.into());
        NodeRef(index as u32)
    }
}

impl IntoIterator for Ast {
    type Item = Node;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl Index<NodeRef> for Ast {
    type Output = Node;

    fn index(&self, index: NodeRef) -> &Self::Output {
        &self.0[index.0 as usize]
    }
}

impl Display for Ast {
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
pub enum Node {
    Stat(Stat),
    Expr(Expr),
}

impl From<Stat> for Node {
    fn from(value: Stat) -> Self {
        Node::Stat(value)
    }
}

impl From<Expr> for Node {
    fn from(value: Expr) -> Self {
        Node::Expr(value)
    }
}

#[derive(Debug)]
pub enum Stat {
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
pub struct NodeRef(u32);

impl From<NodeRef> for usize {
    fn from(value: NodeRef) -> Self {
        value.0 as usize
    }
}

// display

struct AstPrettyPrinter<'formatter, 'ast> {
    f: PrettyFormatter<'formatter>,
    nodes: &'ast [Node],
    state: Vec<State<'ast>>,
}

impl Write for AstPrettyPrinter<'_, '_> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.f.write_str(s)
    }
}

#[derive(Debug)]
enum State<'ast> {
    EnterStat(&'ast Stat),
    // ExitStat(&'ast Stat),
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

impl<'formatter, 'ast> AstPrettyPrinter<'formatter, 'ast> {
    fn new(ast: &'ast Ast, f: &'formatter mut Formatter<'_>) -> Self {
        assert_ne!(ast.len(), 0, "Ast is empty");

        Self {
            f: PrettyFormatter::new(f),
            nodes: &ast.0,
            state: Vec::with_capacity(32),
        }
    }

    // #[inline]
    // fn indent(&mut self) {
    //     self.f.indent();
    // }
    //
    // #[inline]
    // fn unindent(&mut self) {
    //     self.f.unindent();
    // }

    fn get_node(&self, index: NodeRef) -> &'ast Node {
        &self.nodes[index.0 as usize]
    }

    fn get_statement(&self, index: NodeRef) -> Result<&'ast Stat> {
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

    fn push_state(&mut self, state: State<'ast>) {
        self.state.push(state);
    }

    fn pop_state(&mut self) -> Option<State<'ast>> {
        self.state.pop()
    }

    fn visit(mut self) -> Result<()> {
        let root_ref = NodeRef(self.nodes.len().saturating_sub(1) as u32);
        let root = self.get_statement(root_ref)?;

        self.push_state(State::EnterStat(root));

        while let Some(state) = self.pop_state() {
            match state {
                State::EnterStat(node) => self.enter_stat(node)?,
                // State::ExitStat(node) => self.exit_stat(node)?,
                State::EnterExpr(node) => self.enter_expr(node)?,
                State::ContinueBinExpr(op) => self.continue_bin_expr(op)?,
                State::EndBinExpr => self.end_bin_expr()?
            };
        };

        Ok(())
    }

    fn enter_stat(&mut self, node: &Stat) -> Result<()> {
        match node {
            Stat::Expr(expr) => {
                let expr = self.get_expression(*expr)?;
                self.push_state(State::EnterExpr(expr));
                Ok(())
            }
        }
    }

    // fn exit_stat(&mut self, _node: &Stat) -> Result<()> {
    //     writeln!(self)?;
    //     Ok(())
    // }

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