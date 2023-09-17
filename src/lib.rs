#![warn(
    clippy::empty_structs_with_brackets,
    clippy::inconsistent_struct_constructor,
    clippy::unneeded_field_pattern,
    clippy::large_digit_groups,
    clippy::unreadable_literal,
    clippy::manual_string_new,
    clippy::dbg_macro,
    clippy::undocumented_unsafe_blocks,
    clippy::default_trait_access,
    clippy::semicolon_if_nothing_returned
)]

use std::time::{Duration, Instant};

pub use compiler::*;
pub use interpreter::*;
pub use string::*;

macro_rules! static_assert_size {
    ($ty: ty, $size: expr) => {
        const _: [(); $size] = [(); ::std::mem::size_of::<$ty>()];
    };
}

mod compiler;
mod instruction;
mod interpreter;
mod pretty_formatter;
mod prototype;
mod string;
mod value;

/// Temporary
#[inline]
pub fn timed<F: FnOnce() -> R, R>(f: F) -> (R, Duration) {
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed();
    (result, duration)
}
