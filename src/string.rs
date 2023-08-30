use std::borrow::Borrow;
use std::collections::HashSet;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::ops::Deref;
use std::rc::Rc;
use std::str::from_utf8;

pub trait StringInterner {
    type String: AsRef<[u8]> + Clone;

    fn intern(&mut self, string: &[u8]) -> Self::String;
}

#[derive(Clone, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct PString(Rc<[u8]>);

impl Borrow<[u8]> for PString {
    fn borrow(&self) -> &[u8] {
        &self.0
    }
}

impl AsRef<[u8]> for PString {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl Deref for PString {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Display for PString {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let bytes = self.as_ref();
        match from_utf8(bytes) {
            Ok(string) => write!(f, "{}", string),
            Err(_) => write!(f, "{:?}", bytes),
        }
    }
}

impl Debug for PString {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let bytes = self.as_ref();
        match from_utf8(bytes) {
            Ok(string) => write!(f, "{:?}", string),
            Err(_) => write!(f, "{:?}", bytes),
        }
    }
}

#[derive(Default)]
pub struct PStringInterner {
    strings: HashSet<PString>,
}

impl StringInterner for PStringInterner {
    type String = PString;

    fn intern(&mut self, string: &[u8]) -> Self::String {
        match self.strings.get(string) {
            Some(string) => string.clone(),
            None => {
                let string = PString(Rc::from(Box::from(string)));
                self.strings.insert(string.clone());
                string
            }
        }
    }
}
