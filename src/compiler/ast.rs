use std::fmt::{Display, Error, Formatter, Write};

use crate::pretty_formatter::PrettyFormatter;

#[derive(Debug)]
pub struct Ast<S> {
    pub nodes: Box<[Node<S>]>,
    pub refs: Box<[NodeRef]>,
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
    Compound(RefLen),
    VarDecl(S, Option<NodeRef>),
    Expr(NodeRef),
}

#[derive(Debug)]
pub enum Expr<S> {
    Integer(i64),
    Float(f64),
    Var(S),
    Return(Option<NodeRef>),
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

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct NodeRef(pub u32);

#[derive(Copy, Clone, Debug)]
pub struct RefLen(pub u32);

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
        &self.nodes[index.0 as usize]
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
            };
        }

        Ok(())
    }

    fn enter_stat(&mut self, node: NodeRef) -> Result<()> {
        writeln!(self)?;

        let statement = self.get_statement(node)?;
        match statement {
            Stat::Compound(len) => {
                write!(self, "{{")?;
                self.indent();
                self.push_state(State::ContinueCompoundStat(*len));
            }
            Stat::VarDecl(name, def) => {
                write!(self, "var {}", name)?;
                if let Some(def) = def {
                    write!(self, " = ")?;
                    self.push_state(State::EnterExpr(*def));
                }
            }
            Stat::Expr(expr) => {
                self.push_state(State::EnterExpr(*expr));
            }
        };

        Ok(())
    }

    fn continue_compound_stat(&mut self, len: RefLen) -> Result<()> {
        match len.0 {
            0 => {
                self.unindent();
                writeln!(self)?;
                write!(self, "}}")?;
            }
            _ => {
                let new_len = len.0 - 1;
                self.push_state(State::ContinueCompoundStat(RefLen(new_len)));
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
            Expr::Var(ref v) => write!(self, "{}", v)?,
            Expr::Return(right) => {
                write!(self, "return")?;
                if let Some(right) = right {
                    write!(self, " ")?;
                    self.push_state(State::EnterExpr(right));
                }
            }
            Expr::UnOp(op, right) => {
                match op {
                    UnOp::Neg => write!(self, "-")?,
                }
                self.push_state(State::EnterExpr(right));
            }
            Expr::BinOp(op, left, right) => {
                write!(self, "(")?;
                self.push_state(State::EndBinExpr);
                self.push_state(State::EnterExpr(right));
                self.push_state(State::ContinueBinExpr(op));
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
