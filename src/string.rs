use std::borrow::Borrow;
use std::collections::HashSet;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::io::{Error, Write};
use std::ops::Deref;
use std::ptr::slice_from_raw_parts;
use std::str::{from_utf8, Utf8Error};

use gc_arena::{Collect, Gc, Mutation};
use smallvec::SmallVec;

use crate::value::Value;

pub trait StringInterner<'gc> {
    type String: AsRef<[u8]> + Clone;

    fn intern(&mut self, mc: &Mutation<'gc>, string: &[u8]) -> Self::String;
}

#[derive(Debug, Collect)]
#[collect(require_static)]
pub enum StringError {
    TooLongForShortString(usize),
    TooLongForStaticString(usize),
    WriteErr(Error),
}

impl From<Error> for StringError {
    fn from(value: Error) -> Self {
        Self::WriteErr(value)
    }
}

impl Display for StringError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            StringError::TooLongForShortString(len) => {
                write!(
                    f,
                    "Tried to create a short string with {} bytes ({} bytes max)",
                    len, SHORT_LEN
                )
            }
            StringError::TooLongForStaticString(len) => {
                write!(
                    f,
                    "Tried to create a static string with {} bytes ({} bytes max)",
                    len,
                    u32::MAX
                )
            }
            StringError::WriteErr(err) => write!(f, "{}", err),
        }
    }
}

#[derive(Copy, Clone, Collect)]
#[collect(no_drop)]
#[repr(transparent)]
pub struct PString<'gc>(StringData<'gc>);

/// Private data
///
/// Data must be kept at <= 14 bytes so that the discriminant + [Value] discriminant is <= 16 bytes
#[derive(Copy, Clone, Collect)]
#[collect(no_drop)]
enum StringData<'gc> {
    /// Short string that doesn't require allocation
    Short(u8, [u8; SHORT_LEN]),
    /// Allocated string
    Long(Gc<'gc, Box<[u8]>>),
    /// Essentially a [std::slice] but the len component is a u32
    Static(u32, &'static u8),
}

// make sure documentation stays in sync with this
/// Length of a short string
const SHORT_LEN: usize = 14;

static_assert_size!(PString, 16);

impl<'gc> PString<'gc> {
    /// Creates a new string that may be allocated if necessary
    pub fn new<B: AsRef<[u8]> + ?Sized>(mc: &Mutation<'gc>, bytes: &B) -> Self {
        let bytes = bytes.as_ref();
        Self::try_new_short(bytes).unwrap_or_else(|_| {
            Self(StringData::Long(Gc::new(
                mc,
                bytes.to_vec().into_boxed_slice(),
            )))
        })
    }

    /// Tries to create a new String that does not require allocation.
    ///
    /// Panics if the length is >= 14
    ///
    /// This is the panicking variant of [`String::try_new_short`]
    pub fn new_short<B: AsRef<[u8]> + ?Sized>(bytes: &B) -> Self {
        Self::try_new_short(bytes).unwrap()
    }

    /// Tries to create a new String that does not require allocation.
    ///
    /// The length must be <= 14
    pub fn try_new_short<B: AsRef<[u8]> + ?Sized>(bytes: &B) -> Result<Self, StringError> {
        let bytes = bytes.as_ref();
        match bytes.len() {
            len @ 0..=SHORT_LEN => {
                let mut b = [0; SHORT_LEN];
                b[..len].copy_from_slice(bytes);
                Ok(Self(StringData::Short(len as u8, b)))
            }
            len => Err(StringError::TooLongForShortString(len)),
        }
    }

    /// Creates a new string from an iterator of bytes that may be allocated if necessary
    pub fn from_iter<I: IntoIterator<Item = u8>>(mc: &Mutation<'gc>, bytes: I) -> Self {
        let bytes: SmallVec<[u8; SHORT_LEN]> = bytes.into_iter().collect();
        Self::new(mc, &bytes)
    }

    /// Tries to create a new String that does not require allocation from a static slice.
    ///
    /// Panics if the length is >= [`u32::MAX`]
    ///
    /// This is the panicking variant of [`String::try_from_static`]
    pub fn from_static<B: AsRef<[u8]> + ?Sized>(bytes: &'static B) -> Self {
        Self::try_from_static(bytes).unwrap()
    }

    /// Tries to create a new String that does not require allocation from a static slice.
    ///
    /// The length must be <= [`u32::MAX`]
    pub fn try_from_static<B: AsRef<[u8]> + ?Sized>(
        bytes: &'static B,
    ) -> Result<Self, StringError> {
        let bytes = bytes.as_ref();
        const MAX: usize = u32::MAX as usize;
        match bytes.len() {
            0..=SHORT_LEN => Self::try_new_short(bytes),
            len @ ..=MAX => {
                // SAFETY: const pointer to static byte buffer
                let ptr = unsafe { &*bytes.as_ptr() };
                Ok(Self(StringData::Static(len as u32, ptr)))
            }
            len => Err(StringError::TooLongForStaticString(len)),
        }
    }

    /// Creates a new empty String.
    /// The String is not allocated.
    pub fn empty() -> Self {
        Self(StringData::Short(0, [0; SHORT_LEN]))
    }

    pub fn try_concat<I: IntoIterator<Item = V>, V: Into<Value<'gc>>>(
        mc: &Mutation<'gc>,
        values: I,
    ) -> Result<Self, StringError> {
        let mut bytes: SmallVec<[u8; SHORT_LEN]> = SmallVec::new();
        for value in values.into_iter() {
            match value.into() {
                Value::Null => write!(&mut bytes, "null")?,
                Value::Boolean(true) => write!(&mut bytes, "true")?,
                Value::Boolean(false) => write!(&mut bytes, "false")?,
                Value::Integer(v) => write!(&mut bytes, "{}", v)?,
                Value::Float(v) => write!(&mut bytes, "{}", v)?,
                Value::String(v) => bytes.extend_from_slice(v.as_bytes()),
            }
        }
        Ok(match bytes.len() {
            len @ 0..=SHORT_LEN => {
                bytes.resize(SHORT_LEN, 0);
                let bytes = bytes.into_inner().unwrap();
                Self(StringData::Short(len as u8, bytes))
            }
            _ => Self(StringData::Long(Gc::new(mc, bytes.into_boxed_slice()))),
        })
    }

    pub fn try_concat_from_slice(
        mc: &Mutation<'gc>,
        values: &[Value<'gc>],
    ) -> Result<Self, StringError> {
        Self::try_concat(mc, values.iter().copied())
    }

    pub fn len(&self) -> usize {
        match self.0 {
            StringData::Short(len, _) => len as usize,
            StringData::Long(str) => str.len(),
            StringData::Static(len, _) => len as usize,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_bytes(&self) -> &[u8] {
        match &self.0 {
            StringData::Short(len, bytes) => &bytes[..*len as usize],
            StringData::Long(str) => str,
            &StringData::Static(len, bytes) => {
                // SAFETY: same as a &[u8] but the len component is u32
                unsafe { &*slice_from_raw_parts(bytes, len as usize) }
            }
        }
    }

    pub fn as_str(&self) -> Result<&str, Utf8Error> {
        from_utf8(self.as_bytes())
    }
}

impl Borrow<[u8]> for PString<'_> {
    fn borrow(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl<T> AsRef<T> for PString<'_>
where
    T: ?Sized,
    for<'gc> <PString<'gc> as Deref>::Target: AsRef<T>,
{
    fn as_ref(&self) -> &T {
        self.deref().as_ref()
    }
}

impl Deref for PString<'_> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_bytes()
    }
}

impl Display for PString<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let bytes = self.as_ref();
        match from_utf8(bytes) {
            Ok(string) => Display::fmt(string, f),
            Err(_) => write!(f, "{:?}", bytes),
        }
    }
}

impl Debug for PString<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            match &self.0 {
                StringData::Short(_, _) => write!(f, "Short")?,
                StringData::Long(_) => write!(f, "Long")?,
                StringData::Static(_, _) => write!(f, "Static")?,
            }
            write!(f, "(")?;
        }
        let bytes = self.as_bytes();
        match from_utf8(bytes) {
            Ok(string) => write!(f, "{:?}", string)?,
            Err(_) => write!(f, "{:?}", bytes)?,
        }
        if f.alternate() {
            write!(f, ")")?;
        }
        Ok(())
    }
}

impl<T: AsRef<[u8]>> PartialEq<T> for PString<'_> {
    fn eq(&self, other: &T) -> bool {
        self.as_bytes() == other.as_ref()
    }
}

impl Eq for PString<'_> {}

impl Hash for PString<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_bytes().hash(state);
    }
}

impl Default for PString<'_> {
    fn default() -> Self {
        Self::empty()
    }
}

impl TryFrom<&'static str> for PString<'_> {
    type Error = StringError;
    fn try_from(value: &'static str) -> Result<Self, Self::Error> {
        Self::try_from_static(value)
    }
}

impl TryFrom<&'static [u8]> for PString<'_> {
    type Error = StringError;
    fn try_from(value: &'static [u8]) -> Result<Self, Self::Error> {
        Self::try_from_static(value)
    }
}

#[derive(Default)]
pub struct PStringInterner<'gc> {
    strings: HashSet<PString<'gc>>,
}

impl<'gc> StringInterner<'gc> for PStringInterner<'gc> {
    type String = PString<'gc>;

    fn intern(&mut self, mc: &Mutation<'gc>, string: &[u8]) -> Self::String {
        match self.strings.get(string) {
            Some(string) => *string,
            None => {
                let string = PString::new(mc, string);
                self.strings.insert(string);
                string
            }
        }
    }
}
