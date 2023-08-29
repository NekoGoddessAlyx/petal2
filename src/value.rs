use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};

#[derive(Copy, Clone, Debug)]
pub enum Value {
    Integer(i64),
    Float(f64),
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => a == b,
            (Value::Integer(a), Value::Float(b)) => &(*a as f64) == b,
            (Value::Float(a), Value::Integer(b)) => a == &(*b as f64),
            (Value::Float(a), Value::Float(b)) => a == b,
        }
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Value::Integer(v) => {
                state.write_u8(1);
                v.hash(state);
            }
            Value::Float(v) => {
                state.write_u8(2);
                v.to_bits().hash(state);
            }
        }
    }
}

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

static_assert_size!(Value, 16);
