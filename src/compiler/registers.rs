use crate::compiler::sem_check::Local;

#[derive(Copy, Clone, Default, PartialEq, Debug)]
pub struct Register(u8);

impl From<Register> for crate::prototype::Register {
    fn from(value: Register) -> Self {
        value.0
    }
}

pub struct Registers {
    /// Register state: in use?
    registers: [bool; 256],
    /// Maps local indices to registers
    locals: [u8; 256],
    /// Location of first free register (or 256 if no free registers)
    first_free: u16,
    /// The location after the largest register in use (or 256 if no free registers)
    stack_top: u16,
    /// The largest register that has been allocated so far
    stack_size: u16,
}

impl Registers {
    pub fn new() -> Self {
        Self {
            registers: [false; 256],
            locals: [0; 256],
            first_free: 0,
            stack_top: 0,
            stack_size: 0,
        }
    }

    pub fn stack_top(&self) -> u16 {
        self.stack_top
    }

    pub fn stack_size(&self) -> u16 {
        self.stack_size
    }

    /// Assigns a local to a register.
    ///
    /// The register must already be allocated.
    pub fn assign_local(&mut self, local: Local, register: Register) {
        assert!(
            self.registers[register.0 as usize],
            "Register {} must be allocated",
            register.0
        );
        self.locals[local.0 as usize] = register.0
    }

    /// Returns the register assigned to a given local.
    ///
    /// The register must still be allocated or None will be returned.
    #[must_use]
    pub fn address_of_local(&mut self, local: Local) -> Option<Register> {
        let register = self.locals[local.0 as usize];
        if !self.registers[register as usize] {
            return None;
        }
        Some(Register(register))
    }

    /// Allocates any free register.
    ///
    /// Returns None if there are no free registers remaining.
    #[must_use]
    pub fn allocate_any(&mut self) -> Option<Register> {
        if self.first_free >= 256 {
            return None;
        }

        let register = self.first_free as u8;
        self.registers[register as usize] = true;

        if self.first_free == self.stack_top {
            self.stack_top += 1;
        }

        let mut i = self.first_free;
        self.first_free = loop {
            if i >= 256 || !self.registers[i as usize] {
                break i;
            }
            i += 1;
        };
        self.stack_size = self.stack_size.max(self.stack_top);

        Some(Register(register))
    }

    /// Frees a single register
    pub fn free(&mut self, register: Register) {
        assert!(
            self.registers[register.0 as usize],
            "Cannot free unallocated register {}",
            register.0
        );

        let register = register.0 as u16;
        self.registers[register as usize] = false;
        self.first_free = self.first_free.min(register);
        if register + 1 == self.stack_top {
            self.stack_top = register;

            for i in (self.first_free..self.stack_top).rev() {
                if self.registers[i as usize] {
                    break;
                }
                self.stack_top = i;
            }
        }
    }

    /// Frees all registers allocated above the given register
    pub fn pop_to(&mut self, new_top: u16) {
        if self.stack_top > new_top {
            for i in new_top..self.stack_top {
                self.registers[i as usize] = false;
            }
            self.stack_top = new_top;

            self.first_free = self.first_free.min(self.stack_top);

            for i in (self.first_free..self.stack_top).rev() {
                if self.registers[i as usize] {
                    break;
                }
                self.stack_top = i;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::compiler::registers::{Register, Registers};

    #[test]
    fn test() {
        let mut registers = Registers::new();

        let register_1 = registers.allocate_any().unwrap();
        let register_2 = registers.allocate_any().unwrap();
        let register_3 = registers.allocate_any().unwrap();

        assert_eq!(register_1.0, 0);
        assert_eq!(register_2.0, 1);
        assert_eq!(register_3.0, 2);
        assert_eq!(registers.first_free, 3);
        assert_eq!(registers.stack_top, 3);

        registers.free(register_2);
        assert_eq!(registers.first_free, 1);
        assert_eq!(registers.stack_top, 3);

        let register_2 = registers.allocate_any().unwrap();
        assert_eq!(register_2.0, 1);
        assert_eq!(registers.first_free, 3);
        assert_eq!(registers.stack_top, 3);

        registers.free(register_2);
        registers.free(register_3);
        assert_eq!(registers.first_free, 1);
        assert_eq!(registers.stack_top, 1);

        let register_2 = registers.allocate_any().unwrap();
        assert_eq!(register_2.0, 1);
        assert_eq!(registers.first_free, 2);
        assert_eq!(registers.stack_top, 2);

        registers.free(register_2);
        registers.free(register_1);
        assert_eq!(registers.first_free, 0);
        assert_eq!(registers.stack_top, 0);

        for i in 0_u8..12 {
            let register = registers.allocate_any().unwrap();
            assert_eq!(register.0, i);
        }
        assert_eq!(registers.first_free, 12);
        assert_eq!(registers.stack_top, 12);

        for i in 3_u8..9 {
            let register = Register(i);
            registers.free(register);
        }
        assert_eq!(registers.first_free, 3);
        assert_eq!(registers.stack_top, 12);

        let register = registers.allocate_any().unwrap();
        assert_eq!(register.0, 3);
        assert_eq!(registers.first_free, 4);
        assert_eq!(registers.stack_top, 12);
    }
}
