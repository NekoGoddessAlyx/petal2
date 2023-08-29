use crate::ast::{NodeRef, RefLen};
use crate::compiler::ast::{Ast, BinOp, Expr, Node, Stat, UnOp};
use crate::value::Value;

pub fn interpret(ast: Ast) -> Value {
    assert!(!ast.is_empty());

    let mut state = vec![Value::Integer(0); ast.len()];
    let refs = ast.refs();
    let mut ref_cursor: usize = 0;

    fn get(state: &mut [Value], index: impl Into<usize>) -> Value {
        state[index.into()]
    }

    fn get_last_ref(
        cursor: &mut usize,
        refs: &[NodeRef],
        state: &mut [Value],
        len: RefLen,
    ) -> Value {
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
            Node::Stat(Stat::Expr(expression)) => get(&mut state, *expression),

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
