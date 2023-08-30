pub use compiler::*;
pub use interpreter::*;
pub use string::*;

macro_rules! static_assert_size {
    ($ty: ty, $size: expr) => {
        const _: [(); $size] = [(); ::std::mem::size_of::<$ty>()];
    };
}

pub(crate) use static_assert_size;

mod compiler;
mod interpreter;
mod pretty_formatter;
mod prototype;
mod string;
mod value;
