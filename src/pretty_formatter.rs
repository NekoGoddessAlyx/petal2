use std::fmt::{Result, Write};

pub struct PrettyFormatter<'a> {
    w: &'a mut dyn Write,
    indent: &'a str,
    was_nl: bool,
    indent_level: usize,
}

#[allow(unused)]
impl<'a> PrettyFormatter<'a> {
    pub fn new(w: &'a mut dyn Write) -> Self {
        Self::new_with_indent(w, "    ")
    }

    pub fn new_with_indent(w: &'a mut dyn Write, indent: &'a str) -> Self {
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

impl Write for PrettyFormatter<'_> {
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
