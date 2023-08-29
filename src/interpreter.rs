use std::fmt::{Display, Formatter};

use crate::ast::{NodeRef, RefLen};
use crate::compiler::ast::{Ast, BinOp, Expr, Node, Stat, UnOp};
use crate::static_assert_size;

#[derive(Copy, Clone, Debug)]
pub enum Value {
    Integer(i64),
    Float(f64),
}

static_assert_size!(Value, 16);

impl std::ops::Neg for Value {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            Value::Integer(v) => Value::Integer(-v),
            Value::Float(v) => Value::Float(-v),
        }
    }
}

impl std::ops::Add for Value {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Integer(a), Value::Integer(b)) => Value::Integer(a.wrapping_add(b)),
            (Value::Integer(a), Value::Float(b)) => Value::Float((a as f64) + b),
            (Value::Float(a), Value::Integer(b)) => Value::Float(a + (b as f64)),
            (Value::Float(a), Value::Float(b)) => Value::Float(a + b),
        }
    }
}

impl std::ops::Sub for Value {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Integer(a), Value::Integer(b)) => Value::Integer(a.wrapping_sub(b)),
            (Value::Integer(a), Value::Float(b)) => Value::Float((a as f64) - b),
            (Value::Float(a), Value::Integer(b)) => Value::Float(a - (b as f64)),
            (Value::Float(a), Value::Float(b)) => Value::Float(a - b),
        }
    }
}

impl std::ops::Mul for Value {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Integer(a), Value::Integer(b)) => Value::Integer(a.wrapping_mul(b)),
            (Value::Integer(a), Value::Float(b)) => Value::Float((a as f64) * b),
            (Value::Float(a), Value::Integer(b)) => Value::Float(a * (b as f64)),
            (Value::Float(a), Value::Float(b)) => Value::Float(a * b),
        }
    }
}

impl std::ops::Div for Value {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Integer(a), Value::Integer(b)) => Value::Integer(a.checked_div(b).unwrap_or(0)),
            (Value::Integer(a), Value::Float(b)) => Value::Float((a as f64) / b),
            (Value::Float(a), Value::Integer(b)) => Value::Float(a / (b as f64)),
            (Value::Float(a), Value::Float(b)) => Value::Float(a / b),
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Integer(v) => write!(f, "{v}"),
            Value::Float(v) => write!(f, "{v}"),
        }
    }
}

pub fn interpret(ast: Ast) -> Value {
    assert!(!ast.is_empty());

    let mut state = vec![Value::Integer(0); ast.len()];
    let refs = ast.refs();
    let mut ref_cursor: usize = 0;

    fn get(state: &mut [Value], index: impl Into<usize>) -> Value {
        state[index.into()]
    }

    fn get_last_ref(cursor: &mut usize, refs: &[NodeRef], state: &mut [Value], len: RefLen) -> Value {
        let len: usize = len.into();
        *cursor += len;
        let index = refs[cursor.saturating_sub(1)];
        get(state, index)
    }

    for (i, node) in ast.nodes().iter().enumerate() {
        let result = match node {
            // statements
            Node::Stat(Stat::Compound(len)) => {
                get_last_ref(&mut ref_cursor, refs, &mut state, *len)
            }
            Node::Stat(Stat::Expr(expression)) => {
                get(&mut state, *expression)
            }

            // expression
            Node::Expr(Expr::Integer(v)) => Value::Integer(*v),
            Node::Expr(Expr::Float(v)) => Value::Float(*v),
            Node::Expr(Expr::UnOp(op, right)) => {
                let right = get(&mut state, *right);
                match op {
                    UnOp::Neg => -right,
                }
            }
            Node::Expr(Expr::BinOp(op, left, right)) => {
                let left = get(&mut state, *left);
                let right = get(&mut state, *right);
                match op {
                    BinOp::Add => left + right,
                    BinOp::Sub => left - right,
                    BinOp::Mul => left * right,
                    BinOp::Div => left / right,
                }
            }
        };
        state[i] = result;
    }

    let last_index = state.len().saturating_sub(1);
    get(&mut state, last_index)
}