use std::fmt::{Result, Write};

pub struct PrettyFormatter<'a, W> {
    w: &'a mut W,
    indent: &'a str,
    was_nl: bool,
    indent_level: usize,
}

impl<'a, W: Write> PrettyFormatter<'a, W> {
    pub fn new(w: &'a mut W) -> Self {
        Self::new_with_indent(w, "    ")
    }

    pub fn new_with_indent(w: &'a mut W, indent: &'a str) -> Self {
        Self {
            w,
            indent,
            was_nl: false,
            indent_level: 0,
        }
    }

    pub fn indent(&mut self) {
        assert!(
            self.indent_level < usize::MAX,
            "Cannot indent an indent level of {}",
            usize::MAX
        );
        self.indent_level += 1;
    }

    pub fn unindent(&mut self) {
        assert!(
            self.indent_level > 0,
            "Cannot unindent an indent level of 0"
        );
        self.indent_level -= 1;
    }
}

impl<W: Write> Write for PrettyFormatter<'_, W> {
    fn write_str(&mut self, s: &str) -> Result {
        for s in s.split_inclusive('\n') {
            if self.was_nl {
                for _ in 0..self.indent_level {
                    self.w.write_str(self.indent)?;
                }
            }

            self.was_nl = s.ends_with('\n');
            self.w.write_str(s)?;
        }

        Ok(())
    }
}
