use std::ops::Index;

#[derive(Debug)]
pub struct Ast(Vec<Node>);

impl Ast {
    pub(super) fn new(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub(super) fn push(&mut self, node: Node) -> NodeRef {
        let index = self.0.len();
        self.0.push(node);
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

#[derive(Debug)]
pub enum Node {
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