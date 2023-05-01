use std::{
    cmp::Ordering,
    fmt::{Display, LowerHex},
    sync::atomic::AtomicBool,
};

use crate::ops;

macro_rules! regs {
    ($reg:ident, $abi:ident;) => {
        pub const $reg: usize = 0;
        pub const $abi: usize = 0;
    };
    ($reg:ident, $abi:ident;$($body:tt)+) => {
        pub const $reg: usize = 0;
        pub const $abi: usize = 0;

        regs!{ 1, $($body)* }
    };
    ($val:expr, $reg:ident, $abi:ident;) => {
        pub const $reg: usize = $val;
        pub const $abi: usize = $val;
    };
    ($val:expr, $reg:ident, $abi:ident;$($body:tt)+) => {
        pub const $reg: usize = $val;
        pub const $abi: usize = $val;

        regs!{ $val + 1, $($body)* }
    };
    () => {

    };
}

pub mod regs {
    #![allow(dead_code)]
    #![allow(non_upper_case_globals)]

    regs! {
        x0, zero;
        x1, ra;
        x2, sp;
        x3, gp;
        x4, tp;
        x5, t0;
        x6, t1;
        x7, t2;
        x8, s0;
        x9, s1;
        x10, a0;
        x11, a1;
        x12, a2;
        x13, a3;
        x14, a4;
        x15, a5;
        x16, a6;
        x17, a7;
        x18, s2;
        x19, s3;
        x20, s4;
        x21, s5;
        x22, s6;
        x23, s7;
        x24, s8;
        x25, s9;
        x26, s10;
        x27, s11;
        x28, t3;
        x29, t4;
        x30, t5;
        x31, t6;
        pc, pc_abi;
    }

    pub const fp: usize = s0;
}

pub struct CPU<'a, T: CPUBitSize> {
    registers: [T; 32],
    pc: T,

    run: AtomicBool,

    memory: &'a mut [u8],
}

impl<'a, T: CPUBitSize + std::fmt::Debug> std::fmt::Debug for CPU<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, reg) in self.registers.iter().enumerate() {
            let reg_name = match i {
                regs::x0 => "x0 (zero)",
                regs::x1 => "x1 (ra)",
                regs::x2 => "x2 (sp)",
                regs::x3 => "x3 (gp)",
                regs::x4 => "x4 (tp)",
                regs::x5 => "x5 (t0)",
                regs::x6 => "x6 (t1)",
                regs::x7 => "x7 (t2)",
                regs::x8 => "x8 (s0/fp)",
                regs::x9 => "x9 (s1)",
                regs::x10 => "x10 (a0)",
                regs::x11 => "x11 (a1)",
                regs::x12 => "x12 (a2)",
                regs::x13 => "x13 (a3)",
                regs::x14 => "x14 (a4)",
                regs::x15 => "x15 (a5)",
                regs::x16 => "x16 (a6)",
                regs::x17 => "x17 (a7)",
                regs::x18 => "x18 (s2)",
                regs::x19 => "x19 (s3)",
                regs::x20 => "x20 (s4)",
                regs::x21 => "x21 (s5)",
                regs::x22 => "x22 (s6)",
                regs::x23 => "x23 (s7)",
                regs::x24 => "x24 (s8)",
                regs::x25 => "x25 (s9)",
                regs::x26 => "x26 (s10)",
                regs::x27 => "x27 (s11)",
                regs::x28 => "x28 (t3)",
                regs::x29 => "x29 (t4)",
                regs::x30 => "x30 (t5)",
                regs::x31 => "x31 (t6)",
                _ => panic!(),
            };
            write!(f, "{} = {:x}\n", reg_name, reg)?;
        }
        write!(f, "pc = {:x}", self.pc)?;
        Ok(())
    }
}

impl<'a, T: CPUBitSize> CPU<'a, T> {
    pub fn new(memory: &'a mut [u8], regs: &[(usize, T)]) -> CPU<'a, T> {
        let mut registers = [T::zero(); 32];
        let mut pc = T::zero();
        for reg in regs {
            if reg.0 < registers.len() {
                registers[reg.0] = reg.1;
            } else if reg.0 == regs::pc {
                pc = reg.1;
            }
        }
        CPU {
            registers,
            pc,

            run: AtomicBool::new(true),

            memory,
        }
    }

    pub fn get_register(&self, idx: u8) -> T {
        if idx == 32 {
            self.pc
        } else {
            let idx = (idx & 0x1F) as usize;
            self.registers[idx]
        }
    }

    pub fn set_register(&mut self, idx: u8, value: T) {
        if idx == 32 {
            self.pc = value;
        } else {
            let idx = (idx & 0x1F) as usize;
            self.registers[idx] = value;
        }
    }

    pub fn get_memory(&self, addr: usize, size: OpWidth) -> T {
        match size {
            OpWidth::Byte => {
                let value = self.memory[addr];
                if value & 0x80 > 1 {
                    T::MAX & !T::from_32(0xFF) | T::from_32(value as u32)
                } else {
                    T::from_32(value as u32)
                }
            }
            OpWidth::ByteUnsigned => {
                let value = self.memory[addr];
                T::from_32(value as u32)
            }
            OpWidth::Half => {
                let value = u16::from_le_bytes([self.memory[addr], self.memory[addr + 1]]);
                if value & 0x8000 > 1 {
                    T::MAX & !T::from_32(0xFFFF) | T::from_32(value as u32)
                } else {
                    T::from_32(value as u32)
                }
            }
            OpWidth::HalfUnsigned => {
                let value = u16::from_le_bytes([self.memory[addr], self.memory[addr + 1]]);
                T::from_32(value as u32)
            }
            OpWidth::Word => {
                let value = u32::from_le_bytes([
                    self.memory[addr],
                    self.memory[addr + 1],
                    self.memory[addr + 2],
                    self.memory[addr + 3],
                ]);
                T::from_32(value as u32)
            }
            OpWidth::WordUnsigned => {
                let value = u32::from_le_bytes([
                    self.memory[addr],
                    self.memory[addr + 1],
                    self.memory[addr + 2],
                    self.memory[addr + 3],
                ]);
                if value & 0x80000000 > 1 {
                    T::MAX & !T::from_32(0xFFFFFFFF) | T::from_32(value as u32)
                } else {
                    T::from_32(value as u32)
                }
            }
            OpWidth::Double => {
                let value = u64::from_le_bytes([
                    self.memory[addr],
                    self.memory[addr + 1],
                    self.memory[addr + 2],
                    self.memory[addr + 3],
                    self.memory[addr + 4],
                    self.memory[addr + 5],
                    self.memory[addr + 6],
                    self.memory[addr + 7],
                ]);
                T::from_64(value)
            }
        }
    }

    pub fn set_memory(&mut self, addr: usize, size: OpWidth, value: T) {
        match size {
            OpWidth::Byte | OpWidth::ByteUnsigned => {
                value.write_le(&mut self.memory[addr..addr + 1]);
            }
            OpWidth::Half | OpWidth::HalfUnsigned => {
                value.write_le(&mut self.memory[addr..addr + 2]);
            }
            OpWidth::Word | OpWidth::WordUnsigned => {
                value.write_le(&mut self.memory[addr..addr + 4]);
            }
            OpWidth::Double => {
                value.write_le(&mut self.memory[addr..addr + 8]);
            }
        }
    }

    pub fn continute(&mut self) {
        while self.run.load(std::sync::atomic::Ordering::SeqCst) {
            self.step();
            println!("CPU:\n{:?}\n", self);
        }
    }

    pub fn step(&mut self) {
        self.execute();
        self.pc = self.pc + T::from_32(4);
    }

    pub fn execute(&mut self) {
        let inst = self.current_instruction();

        let ty = inst.opcode().get_type().unwrap();
        println!("{:x?}  --  {:?}", inst.opcode(), ty);

        let _: Option<()> = try {
            match ty {
                InstructionType::R => match (inst.opcode(), inst.funct3(), inst.funct7(), 0) {
                    ops::ADD => {
                        self.set_register(
                            inst.destination(),
                            self.get_register(inst.source1()) + self.get_register(inst.source2()),
                        );
                    }
                    ops::SUB => {
                        self.set_register(
                            inst.destination(),
                            self.get_register(inst.source2()) - self.get_register(inst.source1()),
                        );
                    }
                    ops::SLT => {
                        if self.get_register(inst.source1()) < self.get_register(inst.source2()) {
                            self.set_register(inst.destination(), T::one());
                        } else {
                            self.set_register(inst.destination(), T::zero());
                        }
                    }
                    ops::SLTU => {
                        if let Some(Ordering::Less) = self
                            .get_register(inst.source1())
                            .cmp_signed(self.get_register(inst.source2()))
                        {
                            self.set_register(inst.destination(), T::one());
                        } else {
                            self.set_register(inst.destination(), T::zero());
                        }
                    }
                    ops::AND => {
                        self.set_register(
                            inst.destination(),
                            self.get_register(inst.source1()) & self.get_register(inst.source2()),
                        );
                    }
                    ops::OR => {
                        self.set_register(
                            inst.destination(),
                            self.get_register(inst.source1()) | self.get_register(inst.source2()),
                        );
                    }
                    ops::XOR => {
                        self.set_register(
                            inst.destination(),
                            self.get_register(inst.source1()) ^ self.get_register(inst.source2()),
                        );
                    }
                    ops::SLL => {
                        self.set_register(
                            inst.destination(),
                            self.get_register(inst.source1())
                                << (self.get_register(inst.source2()) & T::from_64(0x1F)),
                        );
                    }
                    ops::SRL => {
                        let imm = self.get_register(inst.source2());
                        self.set_register(
                            inst.destination(),
                            self.get_register(inst.source1()) << (imm & T::from_64(0x1FF)),
                        );
                    }
                    ops::SRA => {
                        let imm = self.get_register(inst.source2());
                        let val = self.get_register(inst.source1()) >> (imm & T::from_64(0x1F));
                        if val.is_signed() {
                            let val = val | !(T::MAX >> (imm & T::from_64(0x1F)));
                            self.set_register(inst.destination(), val);
                        } else {
                            self.set_register(inst.destination(), val);
                        }
                    }
                    _ => (),
                },
                InstructionType::I => {
                    match inst.opcode() {
                        Opcode::Load => {
                            println!(
                                "{:x} {:x} {:x}",
                                self.get_register(inst.source1()),
                                inst.immediate::<T>().unwrap(),
                                self.get_register(inst.source1())
                                    .add_overflow(inst.immediate::<T>().unwrap())
                            );
                            let addr = self
                                .get_register(inst.source1())
                                .add_overflow(inst.immediate().unwrap());

                            let value = self.get_memory(addr.as_idx(), inst.width());
                            println!("{} {:x}", inst.destination(), value);
                            self.set_register(inst.destination(), value);

                            None?
                        }
                        Opcode::System if inst.funct7() == ops::ECALL.2 => {
                            self.run.store(false, std::sync::atomic::Ordering::SeqCst);
                        }
                        _ => (),
                    }
                    match (inst.opcode(), inst.funct3(), 0, 0) {
                        ops::ADDI => {
                            self.set_register(
                                inst.destination(),
                                self.get_register(inst.source1())
                                    .add_overflow(inst.immediate::<T>().unwrap()),
                            );
                        }
                        ops::SLTI => {
                            if self.get_register(inst.source1()) < inst.immediate().unwrap() {
                                self.set_register(inst.destination(), T::one());
                            } else {
                                self.set_register(inst.destination(), T::zero());
                            }
                        }
                        ops::SLTIU => {
                            if let Some(Ordering::Less) = self
                                .get_register(inst.source1())
                                .cmp_signed(inst.immediate().unwrap())
                            {
                                self.set_register(inst.destination(), T::one());
                            } else {
                                self.set_register(inst.destination(), T::zero());
                            }
                        }
                        ops::ANDI => {
                            self.set_register(
                                inst.destination(),
                                self.get_register(inst.source1()) & inst.immediate().unwrap(),
                            );
                        }
                        ops::ORI => {
                            self.set_register(
                                inst.destination(),
                                self.get_register(inst.source1()) | inst.immediate().unwrap(),
                            );
                        }
                        ops::XORI => {
                            self.set_register(
                                inst.destination(),
                                self.get_register(inst.source1()) ^ inst.immediate().unwrap(),
                            );
                        }
                        ops::SLLI => {
                            self.set_register(
                                inst.destination(),
                                self.get_register(inst.source1())
                                    << (inst.immediate::<T>().unwrap() & T::from_64(0x1F)),
                            );
                        }
                        ops::SRI => {
                            let imm = inst.immediate::<T>().unwrap();
                            if imm & T::from_64(0x0b111111100000) > T::from_64(0) {
                                // SRAI
                                let val =
                                    self.get_register(inst.source1()) >> (imm & T::from_64(0x1F));
                                if val.is_signed() {
                                    let val = val | !(T::MAX >> (imm & T::from_64(0x1F)));
                                    self.set_register(inst.destination(), val);
                                } else {
                                    self.set_register(inst.destination(), val);
                                }
                            } else {
                                // SRLI
                                self.set_register(
                                    inst.destination(),
                                    self.get_register(inst.source1()) << (imm & T::from_64(0x1FF)),
                                );
                            }
                        }
                        ops::JALR => {
                            // push onto RAS
                            self.set_register(inst.destination(), self.pc + T::from_64(4));
                            self.pc = self.pc.add_overflow(
                                (inst
                                    .immediate::<T>()
                                    .unwrap()
                                    .add_overflow(self.get_register(inst.source1())))
                                    & !T::from_32(1) - T::from_32(4),
                            );
                        }

                        _ => (),
                    };
                }
                InstructionType::U => match (inst.opcode(), 0, 0, 0) {
                    ops::LUI => {
                        self.set_register(inst.destination(), inst.immediate::<T>().unwrap());
                    }
                    ops::AUIPC => {
                        self.set_register(
                            inst.destination(),
                            inst.immediate::<T>().unwrap() + self.pc,
                        );
                    }
                    _ => (),
                },
                InstructionType::S => {
                    match inst.opcode() {
                        Opcode::Store => {
                            let addr = self
                                .get_register(inst.source1())
                                .add_overflow(inst.immediate().unwrap());
                            self.set_memory(
                                addr.as_idx(),
                                inst.width(),
                                self.get_register(inst.source2()),
                            );
                            None?
                        }
                        _ => (),
                    };
                }
                InstructionType::J => match (inst.opcode(), 0, 0, 0) {
                    ops::JAL => {
                        // push onto RAS
                        self.set_register(inst.destination(), self.pc + T::from_64(4));
                        self.pc =
                            self.pc.add_overflow(inst.immediate::<T>().unwrap()) - T::from_32(4);
                    }
                    _ => (),
                },
                InstructionType::B => match (inst.opcode(), inst.funct3(), 0, 0) {
                    ops::BEQ => {
                        if self.get_register(inst.source1()) == self.get_register(inst.source2()) {
                            self.pc = self.pc + inst.immediate::<T>().unwrap() - T::from_32(4);
                        }
                    }
                    ops::BNE => {
                        if self.get_register(inst.source1()) != self.get_register(inst.source2()) {
                            self.pc = self.pc + inst.immediate::<T>().unwrap() - T::from_32(4);
                        }
                    }
                    ops::BLT => {
                        if self.get_register(inst.source1()) < self.get_register(inst.source2()) {
                            self.pc = self.pc + inst.immediate::<T>().unwrap() - T::from_32(4);
                        }
                    }
                    ops::BLTU => {
                        if let Some(Ordering::Less) = self
                            .get_register(inst.source1())
                            .cmp_signed(self.get_register(inst.source2()))
                        {
                            self.pc = self.pc + inst.immediate::<T>().unwrap() - T::from_32(4);
                        }
                    }
                    ops::BGE => {
                        if self.get_register(inst.source1()) >= self.get_register(inst.source2()) {
                            self.pc = self.pc + inst.immediate::<T>().unwrap() - T::from_32(4);
                        }
                    }
                    ops::BGEU => {
                        if let Some(Ordering::Greater | Ordering::Equal) = self
                            .get_register(inst.source1())
                            .cmp_signed(self.get_register(inst.source2()))
                        {
                            self.pc = self.pc + inst.immediate::<T>().unwrap() - T::from_32(4);
                        }
                    }
                    _ => (),
                },
                _ => (),
            }

            ()
        };
    }

    pub fn current_instruction(&self) -> Instruction {
        u32::from_le_bytes(
            self.memory[self.pc.as_idx()..self.pc.as_idx() + 4]
                .try_into()
                .unwrap(),
        )
        .into()
    }
}

#[rustfmt::skip]
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
#[repr(u8)]
pub enum Opcode {
    Load    = 0b00 << 5 | 0b000 << 2 | 0b11,
    Store   = 0b01 << 5 | 0b000 << 2 | 0b11,
    MAdd    = 0b10 << 5 | 0b000 << 2 | 0b11,
    Branch  = 0b11 << 5 | 0b000 << 2 | 0b11,

    LoadFP  = 0b00 << 5 | 0b001 << 2 | 0b11,
    StoreFP = 0b01 << 5 | 0b001 << 2 | 0b11,
    MSub    = 0b10 << 5 | 0b001 << 2 | 0b11,
    JALR    = 0b11 << 5 | 0b001 << 2 | 0b11,

    Custom0 = 0b00 << 5 | 0b010 << 2 | 0b11,
    Custom1 = 0b01 << 5 | 0b010 << 2 | 0b11,
    NMSub   = 0b10 << 5 | 0b010 << 2 | 0b11,
    Res1    = 0b11 << 5 | 0b010 << 2 | 0b11,

    MiscMem = 0b00 << 5 | 0b011 << 2 | 0b11,
    Amo     = 0b01 << 5 | 0b011 << 2 | 0b11,
    NMAdd   = 0b10 << 5 | 0b011 << 2 | 0b11,
    JAL     = 0b11 << 5 | 0b011 << 2 | 0b11,

    OpImm   = 0b00 << 5 | 0b100 << 2 | 0b11,
    Op      = 0b01 << 5 | 0b100 << 2 | 0b11,
    OpFP    = 0b10 << 5 | 0b100 << 2 | 0b11,
    System  = 0b11 << 5 | 0b100 << 2 | 0b11,

    AUIPC   = 0b00 << 5 | 0b101 << 2 | 0b11,
    LUI     = 0b01 << 5 | 0b101 << 2 | 0b11,
    Res2    = 0b10 << 5 | 0b101 << 2 | 0b11,
    Res3    = 0b11 << 5 | 0b101 << 2 | 0b11,

    OpImm32 = 0b00 << 5 | 0b110 << 2 | 0b11,
    Op32    = 0b01 << 5 | 0b110 << 2 | 0b11,
    Custom2 = 0b10 << 5 | 0b110 << 2 | 0b11,
    Custom3 = 0b11 << 5 | 0b110 << 2 | 0b11,

    Unknown,
}

impl Opcode {
    pub fn get_type(&self) -> Option<InstructionType> {
        use Opcode::*;
        let ty = match self {
            LUI | AUIPC => InstructionType::U,
            JAL => InstructionType::J,
            JALR | Load | OpImm | MiscMem | System | Op32 => InstructionType::I,
            Branch => InstructionType::B,
            Store => InstructionType::S,
            Op | Amo | OpFP => InstructionType::R,
            MAdd | MSub | NMSub | NMAdd => InstructionType::R4,
            _ => return None,
        };

        Some(ty)
    }
}

impl From<u8> for Opcode {
    fn from(value: u8) -> Self {
        use Opcode::*;
        // Hope this is right :/
        match value {
            0b0000011 => Load,
            0b0100011 => Store,
            0b1000011 => MAdd,
            0b1100011 => Branch,

            0b0000111 => LoadFP,
            0b0100111 => StoreFP,
            0b1000111 => MSub,
            0b1100111 => JALR,

            0b0001011 => Custom0,
            0b0101011 => Custom1,
            0b1001011 => NMSub,
            0b1101011 => Res1,

            0b0001111 => MiscMem,
            0b0101111 => Amo,
            0b1001111 => NMAdd,
            0b1101111 => JAL,

            0b0010011 => OpImm,
            0b0110011 => Op,
            0b1010011 => OpFP,
            0b1110011 => System,

            0b0010111 => AUIPC,
            0b0110111 => LUI,
            0b1010111 => Res2,
            0b1110111 => Res3,

            0b0011011 => OpImm32,
            0b0111011 => Op32,
            0b1011011 => Custom2,
            0b1111011 => Custom3,
            _ => Unknown,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum OpWidth {
    Byte = 0b000,
    ByteUnsigned = 0b100,
    Half = 0b001,
    HalfUnsigned = 0b101,
    Word = 0b010,
    WordUnsigned = 0b110,
    Double = 0b011,
    // Quad = 0b100,
}

impl From<u8> for OpWidth {
    fn from(value: u8) -> Self {
        use OpWidth::*;
        match value {
            0b000 => Byte,
            0b100 => ByteUnsigned,
            0b001 => Half,
            0b101 => HalfUnsigned,
            0b010 => Word,
            0b110 => WordUnsigned,
            0b011 => Double,
            _ => panic!("Unkonwn size!"),
        }
    }
}

pub struct Instruction(u32);

impl std::fmt::Debug for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", (self.0 & 0x7f) as u8)
    }
}

impl Instruction {
    pub fn opcode(&self) -> Opcode {
        match (self.0, self.0) {
            _ => (),
        }
        ((self.0 & 0x7F) as u8).into()
    }

    pub fn destination(&self) -> u8 {
        (self.0 >> 7) as u8 & 0x1F
    }

    pub fn source1(&self) -> u8 {
        (self.0 >> 15) as u8 & 0x1F
    }

    pub fn source2(&self) -> u8 {
        (self.0 >> 20) as u8 & 0x1F
    }

    pub fn source3(&self) -> u8 {
        (self.0 >> 27) as u8 & 0x7
    }

    pub fn funct3(&self) -> u8 {
        (self.0 >> 12) as u8 & 0x7
    }

    pub fn funct2(&self) -> u8 {
        (self.0 >> 25) as u8 & 0x3
    }

    pub fn funct7(&self) -> u8 {
        (self.0 >> 25) as u8 & 0x7F
    }

    pub fn width(&self) -> OpWidth {
        self.funct3().into()
    }

    pub fn immediate<T: CPUBitSize>(&self) -> Option<T> {
        let ty = self.opcode().get_type()?;

        match ty {
            InstructionType::I => Some(
                T::from_32((self.0 >> 20) & 0xFFF)
                    | if self.0 >> 31 > 0 {
                        !T::from_64(0x7FF)
                    } else {
                        T::zero()
                    },
            ),
            InstructionType::S => Some(
                T::from_32((self.0 >> 7) & 0x1F)
                    | T::from_32(((self.0 >> 25) & 0x3F) << 5)
                    | if self.0 >> 31 > 0 {
                        !T::from_64(0x7FF)
                    } else {
                        T::zero()
                    },
            ),
            InstructionType::U => Some(
                T::from_32((self.0 >> 12) & 0xFFFFF) << T::from_32(12)
                    | if self.0 >> 31 > 0 {
                        !T::from_64(0xFFF)
                    } else {
                        T::zero()
                    },
            ),
            InstructionType::B => Some(
                T::from_32(
                    ((self.0 >> 8) & 0xF) << 1
                        | ((self.0 >> 25) & 0x3F) << 6
                        | ((self.0 >> 7) & 1) << 11,
                ) | if self.0 >> 31 > 0 {
                    !T::from_64(0x7FFFFFFF)
                } else {
                    T::zero()
                },
            ),
            InstructionType::J => Some(
                T::from_32(
                    ((self.0 >> 21) & 0x7FF) << 1
                        | ((self.0 >> 20) & 1) << 10
                        | ((self.0 >> 12) & 0xFF) << 11
                        | ((self.0 >> 31) & 1) << 19,
                ) | if self.0 >> 31 > 0 {
                    !T::from_64(0xFFFFF)
                } else {
                    T::zero()
                },
            ),
            _ => None,
        }
    }
}

impl From<u32> for Instruction {
    fn from(value: u32) -> Self {
        Instruction(value)
    }
}

#[derive(Debug)]
pub enum InstructionType {
    R,
    R4,
    I,
    S,
    U,
    B,
    J,
}

pub trait CPUBitSize:
    Clone
    + Copy
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::BitOr<Output = Self>
    + std::ops::BitAnd<Output = Self>
    + std::ops::BitXor<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Not<Output = Self>
    + std::ops::Rem<Output = Self>
    + std::ops::Shl<Output = Self>
    + std::ops::Shr<Output = Self>
    + From<u32>
    + PartialEq
    + Eq
    + PartialOrd
    + Ord
    + Display
    + std::fmt::Debug
    + LowerHex
{
    type Signed;
    const MAX: Self;

    fn zero() -> Self;
    fn one() -> Self;
    fn max() -> Self;
    fn cmp_signed(&self, other: Self) -> Option<Ordering>;
    fn from_64(value: u64) -> Self;
    fn from_32(value: u32) -> Self;
    fn as_idx(&self) -> usize;
    fn is_signed(&self) -> bool;
    fn write_le(&self, to: &mut [u8]);

    fn add_overflow(&self, other: Self) -> Self;
    fn sub_underflow(&self, other: Self) -> Self;
}

impl CPUBitSize for u32 {
    type Signed = i32;
    const MAX: u32 = u32::MAX;

    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }

    fn max() -> Self {
        u32::MAX
    }

    fn as_idx(&self) -> usize {
        *self as usize
    }

    fn from_32(value: u32) -> Self {
        value
    }

    fn from_64(value: u64) -> u32 {
        value as u32
    }

    fn cmp_signed(&self, other: Self) -> Option<Ordering> {
        (*self as i32).partial_cmp(&(other as i32))
    }

    fn is_signed(&self) -> bool {
        *self & 0x8FFFFFFF > 0
    }

    fn write_le(&self, to: &mut [u8]) {
        to.copy_from_slice(&self.to_le_bytes());
    }

    fn add_overflow(&self, other: Self) -> Self {
        self.wrapping_add(other)
    }

    fn sub_underflow(&self, other: Self) -> Self {
        self.wrapping_sub(other)
    }
}

impl CPUBitSize for u64 {
    type Signed = i64;
    const MAX: u64 = u64::MAX;

    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }

    fn max() -> Self {
        u64::MAX
    }

    fn as_idx(&self) -> usize {
        *self as usize
    }

    fn from_32(value: u32) -> Self {
        value as u64
    }

    fn from_64(value: u64) -> u64 {
        value
    }

    fn cmp_signed(&self, other: Self) -> Option<Ordering> {
        (*self as i64).partial_cmp(&(other as i64))
    }

    fn is_signed(&self) -> bool {
        *self & 0x8FFFFFFFFFFFFFFF > 0
    }

    fn write_le(&self, to: &mut [u8]) {
        to.copy_from_slice(&self.to_le_bytes());
    }

    fn add_overflow(&self, other: Self) -> Self {
        self.wrapping_add(other)
    }

    fn sub_underflow(&self, other: Self) -> Self {
        self.wrapping_sub(other)
    }
}
