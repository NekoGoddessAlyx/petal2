use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};

#[derive(Copy, Clone, Debug)]
pub enum Value {
    Null,
    Boolean(bool),
    Integer(i64),
    Float(f64),
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Null, Value::Null) => true,
            (Value::Boolean(a), Value::Boolean(b)) => a == b,
            (Value::Integer(a), Value::Integer(b)) => a == b,
            (Value::Integer(a), Value::Float(b)) => &(*a as f64) == b,
            (Value::Float(a), Value::Integer(b)) => a == &(*b as f64),
            (Value::Float(a), Value::Float(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Value::Null => {
                state.write_u8(1);
            }
            Value::Boolean(v) => {
                state.write_u8(2);
                v.hash(state);
            }
            Value::Integer(v) => {
                state.write_u8(3);
                v.hash(state);
            }
            Value::Float(v) => {
                state.write_u8(4);
                v.to_bits().hash(state);
            }
        }
    }
}

#[derive(Debug)]
pub struct TypeError;

impl std::ops::Neg for Value {
    type Output = Result<Self, TypeError>;

    fn neg(self) -> Self::Output {
        Ok(match self {
            Value::Integer(v) => Value::Integer(-v),
            Value::Float(v) => Value::Float(-v),
            _ => return Err(TypeError),
        })
    }
}

impl std::ops::Not for Value {
    type Output = Result<Self, TypeError>;

    fn not(self) -> Self::Output {
        Ok(match self {
            Value::Null => Value::Boolean(true),
            Value::Boolean(v) => Value::Boolean(!v),
            _ => Value::Boolean(false),
        })
    }
}

impl std::ops::Add for Value {
    type Output = Result<Self, TypeError>;

    fn add(self, rhs: Self) -> Self::Output {
        Ok(match (self, rhs) {
            (Value::Integer(a), Value::Integer(b)) => Value::Integer(a.wrapping_add(b)),
            (Value::Integer(a), Value::Float(b)) => Value::Float((a as f64) + b),
            (Value::Float(a), Value::Integer(b)) => Value::Float(a + (b as f64)),
            (Value::Float(a), Value::Float(b)) => Value::Float(a + b),
            _ => return Err(TypeError),
        })
    }
}

impl std::ops::Sub for Value {
    type Output = Result<Self, TypeError>;

    fn sub(self, rhs: Self) -> Self::Output {
        Ok(match (self, rhs) {
            (Value::Integer(a), Value::Integer(b)) => Value::Integer(a.wrapping_sub(b)),
            (Value::Integer(a), Value::Float(b)) => Value::Float((a as f64) - b),
            (Value::Float(a), Value::Integer(b)) => Value::Float(a - (b as f64)),
            (Value::Float(a), Value::Float(b)) => Value::Float(a - b),
            _ => return Err(TypeError),
        })
    }
}

impl std::ops::Mul for Value {
    type Output = Result<Self, TypeError>;

    fn mul(self, rhs: Self) -> Self::Output {
        Ok(match (self, rhs) {
            (Value::Integer(a), Value::Integer(b)) => Value::Integer(a.wrapping_mul(b)),
            (Value::Integer(a), Value::Float(b)) => Value::Float((a as f64) * b),
            (Value::Float(a), Value::Integer(b)) => Value::Float(a * (b as f64)),
            (Value::Float(a), Value::Float(b)) => Value::Float(a * b),
            _ => return Err(TypeError),
        })
    }
}

impl std::ops::Div for Value {
    type Output = Result<Self, TypeError>;

    fn div(self, rhs: Self) -> Self::Output {
        Ok(match (self, rhs) {
            (Value::Integer(a), Value::Integer(b)) => a
                .checked_div(b)
                .map(Value::Integer)
                .unwrap_or(Value::Float(f64::INFINITY)),
            (Value::Integer(a), Value::Float(b)) => Value::Float((a as f64) / b),
            (Value::Float(a), Value::Integer(b)) => Value::Float(a / (b as f64)),
            (Value::Float(a), Value::Float(b)) => Value::Float(a / b),
            _ => return Err(TypeError),
        })
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Null => write!(f, "null"),
            Value::Boolean(true) => write!(f, "true"),
            Value::Boolean(false) => write!(f, "false"),
            Value::Integer(v) => write!(f, "{v}"),
            Value::Float(v) => write!(f, "{v}"),
        }
    }
}

static_assert_size!(Value, 16);
