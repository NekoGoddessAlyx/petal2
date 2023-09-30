use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};

use gc_arena::{Collect, Mutation};
use thiserror::Error;

use crate::PString;

#[derive(Copy, Clone, Debug, Collect)]
#[collect(no_drop)]
pub enum Value<'gc> {
    Null,
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(PString<'gc>),
}

impl<'gc> Value<'gc> {
    pub fn type_name(self) -> &'static str {
        match self {
            Value::Null => "Null",
            Value::Boolean(_) => "Bool",
            Value::Integer(_) => "Int",
            Value::Float(_) => "Float",
            Value::String(_) => "String",
        }
    }

    pub fn to_bool(self) -> bool {
        match self {
            Value::Null | Value::Boolean(false) => false,
            Value::Boolean(true) | Value::Integer(_) | Value::Float(_) | Value::String(_) => true,
        }
    }

    pub fn add(self, mc: &Mutation<'gc>, right: Self) -> Result<Self, TypeError> {
        Ok(match (self, right) {
            (Value::Integer(a), Value::Integer(b)) => Value::Integer(a.wrapping_add(b)),
            (Value::Integer(a), Value::Float(b)) => Value::Float((a as f64) + b),
            (Value::Float(a), Value::Integer(b)) => Value::Float(a + (b as f64)),
            (Value::Float(a), Value::Float(b)) => Value::Float(a + b),
            (a @ Value::String(_), b) => Value::String(
                PString::concat_from_slice(mc, &[a, b]).map_err(|_| TypeError {
                    expected: "",
                    got: b.type_name(),
                })?,
            ),
            (a, b @ Value::String(_)) => Value::String(
                PString::concat_from_slice(mc, &[a, b]).map_err(|_| TypeError {
                    expected: "String",
                    got: a.type_name(),
                })?,
            ),
            _ => {
                return Err(TypeError {
                    expected: "Number",
                    got: self.type_name(),
                })
            }
        })
    }
}

impl PartialEq for Value<'_> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Null, Value::Null) => true,
            (Value::Boolean(a), Value::Boolean(b)) => a == b,
            (Value::Integer(a), Value::Integer(b)) => a == b,
            (Value::Integer(a), Value::Float(b)) => &(*a as f64) == b,
            (Value::Float(a), Value::Integer(b)) => a == &(*b as f64),
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for Value<'_> {}

impl Hash for Value<'_> {
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
            Value::String(v) => {
                state.write_u8(5);
                v.hash(state);
            }
        }
    }
}

#[derive(Debug, Error)]
#[error("Expected {} but got {}", .expected, .got)]
pub struct TypeError {
    expected: &'static str,
    got: &'static str,
}

impl std::ops::Neg for Value<'_> {
    type Output = Result<Self, TypeError>;

    fn neg(self) -> Self::Output {
        Ok(match self {
            Value::Integer(v) => Value::Integer(-v),
            Value::Float(v) => Value::Float(-v),
            _ => {
                return Err(TypeError {
                    expected: "Number",
                    got: self.type_name(),
                })
            }
        })
    }
}

impl std::ops::Not for Value<'_> {
    type Output = Result<Self, TypeError>;

    fn not(self) -> Self::Output {
        Ok(match self {
            Value::Null => Value::Boolean(true),
            Value::Boolean(v) => Value::Boolean(!v),
            _ => Value::Boolean(false),
        })
    }
}

// impl std::ops::Add for Value<'_> {
//     type Output = Result<Self, TypeError>;
//
//     fn add(self, rhs: Self) -> Self::Output {
//         Ok(match (self, rhs) {
//             (Value::Integer(a), Value::Integer(b)) => Value::Integer(a.wrapping_add(b)),
//             (Value::Integer(a), Value::Float(b)) => Value::Float((a as f64) + b),
//             (Value::Float(a), Value::Integer(b)) => Value::Float(a + (b as f64)),
//             (Value::Float(a), Value::Float(b)) => Value::Float(a + b),
//             _ => return Err(TypeError),
//         })
//     }
// }

impl std::ops::Sub for Value<'_> {
    type Output = Result<Self, TypeError>;

    fn sub(self, rhs: Self) -> Self::Output {
        Ok(match (self, rhs) {
            (Value::Integer(a), Value::Integer(b)) => Value::Integer(a.wrapping_sub(b)),
            (Value::Integer(a), Value::Float(b)) => Value::Float((a as f64) - b),
            (Value::Float(a), Value::Integer(b)) => Value::Float(a - (b as f64)),
            (Value::Float(a), Value::Float(b)) => Value::Float(a - b),
            (Value::Integer(_) | Value::Float(_), b) => {
                return Err(TypeError {
                    expected: "Number",
                    got: b.type_name(),
                })
            }
            (a, Value::Integer(_) | Value::Float(_)) => {
                return Err(TypeError {
                    expected: "Number",
                    got: a.type_name(),
                })
            }
            _ => {
                return Err(TypeError {
                    expected: "Number",
                    got: self.type_name(),
                })
            }
        })
    }
}

impl std::ops::Mul for Value<'_> {
    type Output = Result<Self, TypeError>;

    fn mul(self, rhs: Self) -> Self::Output {
        Ok(match (self, rhs) {
            (Value::Integer(a), Value::Integer(b)) => Value::Integer(a.wrapping_mul(b)),
            (Value::Integer(a), Value::Float(b)) => Value::Float((a as f64) * b),
            (Value::Float(a), Value::Integer(b)) => Value::Float(a * (b as f64)),
            (Value::Float(a), Value::Float(b)) => Value::Float(a * b),
            (Value::Integer(_) | Value::Float(_), b) => {
                return Err(TypeError {
                    expected: "Number",
                    got: b.type_name(),
                })
            }
            (a, Value::Integer(_) | Value::Float(_)) => {
                return Err(TypeError {
                    expected: "Number",
                    got: a.type_name(),
                })
            }
            _ => {
                return Err(TypeError {
                    expected: "Number",
                    got: self.type_name(),
                })
            }
        })
    }
}

impl std::ops::Div for Value<'_> {
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
            (Value::Integer(_) | Value::Float(_), b) => {
                return Err(TypeError {
                    expected: "Number",
                    got: b.type_name(),
                })
            }
            (a, Value::Integer(_) | Value::Float(_)) => {
                return Err(TypeError {
                    expected: "Number",
                    got: a.type_name(),
                })
            }
            _ => {
                return Err(TypeError {
                    expected: "Number",
                    got: self.type_name(),
                })
            }
        })
    }
}

impl Display for Value<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Null => write!(f, "null"),
            Value::Boolean(true) => write!(f, "true"),
            Value::Boolean(false) => write!(f, "false"),
            Value::Integer(v) => write!(f, "{v}"),
            Value::Float(v) => write!(f, "{v}"),
            Value::String(v) => write!(f, "{v}"),
        }
    }
}

static_assert_size!(Value, 16);

impl From<()> for Value<'_> {
    #[inline]
    fn from(_: ()) -> Self {
        Self::Null
    }
}

impl From<bool> for Value<'_> {
    #[inline]
    fn from(value: bool) -> Self {
        Self::Boolean(value)
    }
}

impl From<i64> for Value<'_> {
    #[inline]
    fn from(value: i64) -> Self {
        Self::Integer(value)
    }
}

impl From<f64> for Value<'_> {
    #[inline]
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}

impl<'gc> From<PString<'gc>> for Value<'gc> {
    #[inline]
    fn from(value: PString<'gc>) -> Self {
        Self::String(value)
    }
}

pub trait IntoValue<'gc> {
    fn into_value(self, mc: &Mutation<'gc>) -> Value<'gc>;
}

impl<'gc> IntoValue<'gc> for Value<'gc> {
    #[inline]
    fn into_value(self, _: &Mutation<'gc>) -> Value<'gc> {
        self
    }
}

macro_rules! impl_into {
    ($($t: ty),* $(,)?) => {
        $(
            impl<'gc> IntoValue<'gc> for $t {
                #[inline]
                fn into_value(self, _: &Mutation<'gc>) -> Value<'gc> {
                    self.into()
                }
            }
        )*
    };
}
impl_into!((), bool, i64, f64, PString<'gc>,);

macro_rules! impl_into_integer {
    ($($t: ty),* $(,)?) => {
        $(
            impl<'gc> IntoValue<'gc> for $t {
                #[inline]
                fn into_value(self, _: &Mutation<'gc>) -> Value<'gc> {
                    Value::Integer(self.into())
                }
            }
        )*
    };
}
impl_into_integer!(u8, i8, u16, i16, u32, i32);

impl<'gc> IntoValue<'gc> for f32 {
    #[inline]
    fn into_value(self, _: &Mutation<'gc>) -> Value<'gc> {
        Value::Float(self.into())
    }
}

macro_rules! impl_copy_into {
    ($($t: ty),* $(,)?) => {
        $(
            impl<'a, 'gc> IntoValue<'gc> for &'a $t {
                #[inline]
                fn into_value(self, mc: &Mutation<'gc>) -> Value<'gc> {
                    (*self).into_value(mc)
                }
            }
        )*
    };
}
impl_copy_into!(
    (),
    bool,
    u8,
    i8,
    u16,
    i16,
    u32,
    i32,
    i64,
    f32,
    f64,
    PString<'gc>,
);

impl<'gc> IntoValue<'gc> for &'static str {
    fn into_value(self, mc: &Mutation<'gc>) -> Value<'gc> {
        Value::String(PString::try_from_static(self).unwrap_or_else(|_| PString::new(mc, self)))
    }
}

impl<'gc> IntoValue<'gc> for &'static [u8] {
    fn into_value(self, mc: &Mutation<'gc>) -> Value<'gc> {
        Value::String(PString::try_from_static(self).unwrap_or_else(|_| PString::new(mc, self)))
    }
}

impl<'gc> IntoValue<'gc> for String {
    fn into_value(self, mc: &Mutation<'gc>) -> Value<'gc> {
        Value::String(PString::new(mc, &self))
    }
}

impl<'gc, T: IntoValue<'gc>> IntoValue<'gc> for Option<T> {
    fn into_value(self, mc: &Mutation<'gc>) -> Value<'gc> {
        match self {
            None => Value::Null,
            Some(v) => v.into_value(mc),
        }
    }
}

impl<'a, 'gc, T> IntoValue<'gc> for &'a Option<T>
where
    &'a T: IntoValue<'gc>,
{
    fn into_value(self, mc: &Mutation<'gc>) -> Value<'gc> {
        match self {
            None => Value::Null,
            Some(v) => v.into_value(mc),
        }
    }
}

pub trait FromValue<'gc>: Sized {
    fn from_value(value: Value<'gc>) -> Result<Self, TypeError>;
}

impl<'gc> FromValue<'gc> for Value<'gc> {
    fn from_value(value: Value<'gc>) -> Result<Self, TypeError> {
        Ok(value)
    }
}

macro_rules! impl_integer_from {
    ($($t: ty),* $(,)?) => {
        $(
            impl<'gc> FromValue<'gc> for $t {
                fn from_value(value: Value<'gc>) -> Result<Self, TypeError> {
                    match value {
                        Value::Integer(v) => v.try_into().map_err(|_| TypeError {
                            expected: "Int",
                            got: "Int",
                        }),
                        Value::Float(v) => {
                            let v_i = v as i64;
                            match (v_i as f64) == v {
                                true => v_i.try_into().map_err(|_| TypeError {
                                    expected: "Int",
                                    got: "Float",
                                }),
                                false => Err(TypeError {
                                    expected: "Int",
                                    got: "Float",
                                }),
                            }
                        }
                        Value::String(v) => {
                            let s = std::str::from_utf8(v.as_bytes()).map_err(|_| TypeError {
                                expected: "Int",
                                got: "String",
                            })?;
                            str::parse(s).map_err(|_| TypeError {
                                expected: "Int",
                                got: "String",
                            })
                        }
                        _ => Err(TypeError {
                            expected: "Int",
                            got: value.type_name(),
                        }),
                    }
                }
            }
        )*
    };
}
impl_integer_from!(u8, i8, u16, i16, u32, i32, u64, i64);

macro_rules! impl_float_from {
    ($($t: ty),* $(,)?) => {
        $(
            impl<'gc> FromValue<'gc> for $t {
                fn from_value(value: Value<'gc>) -> Result<Self, TypeError> {
                    match value {
                        Value::Integer(v) => Ok(v as $t),
                        Value::Float(v) => Ok(v as $t),
                        Value::String(v) => {
                            let s = std::str::from_utf8(v.as_bytes()).map_err(|_| TypeError {
                                expected: "Float",
                                got: "String",
                            })?;
                            str::parse(s).map_err(|_| TypeError {
                                expected: "Float",
                                got: "String",
                            })
                        }
                        _ => Err(TypeError {
                            expected: "Float",
                            got: value.type_name(),
                        }),
                    }
                }
            }
        )*
    };
}
impl_float_from!(f32, f64);

impl<'gc> FromValue<'gc> for () {
    fn from_value(value: Value<'gc>) -> Result<Self, TypeError> {
        match value {
            Value::Null => Ok(()),
            _ => Err(TypeError {
                expected: "Null",
                got: value.type_name(),
            }),
        }
    }
}

macro_rules! impl_from {
    ($($v: ident($ty_name: ident) -> $t: ty),* $(,)?) => {
        $(
            impl<'gc> FromValue<'gc> for $t {
                fn from_value(value: Value<'gc>) -> Result<Self, TypeError> {
                    match value {
                        Value::$v(v) => Ok(v),
                        _ => Err(TypeError {
                            expected: "$ty_name",
                            got: value.type_name(),
                        }),
                    }
                }
            }
        )*
    };
}
impl_from!(Boolean(Bool) -> bool, String(String) -> PString<'gc>);
