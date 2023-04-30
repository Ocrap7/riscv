use crate::ops;

pub struct CPU<'a> {
    registers: [u32; 32],
    pc: u32,

    memory: &'a mut [u8],
}

impl CPU<'_> {
    pub fn new(memory: &mut [u8]) -> CPU {
        CPU {
            registers: [0; 32],
            pc: 0,
            memory,
        }
    }

    pub fn step(&mut self) {}

    pub fn execute(&mut self) {
        let inst = self.current_instruction();

        let ty = inst.opcode().get_type().unwrap();

        match ty {
            InstructionType::I => {
                
            }
            _ => ()
        }

        match (inst.opcode(), inst.funct3(), inst.funct7(), inst.funct2()) {
            ops::ADDI => {

            }
            _ => ()
        }
    }

    pub fn current_instruction(&self) -> Instruction {
        u32::from_le_bytes(
            self.memory[self.pc as usize..self.pc as usize + 4]
                .try_into()
                .unwrap(),
        )
        .into()
    }
}

#[rustfmt::skip]
#[derive(PartialEq, Eq)]
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
        match value {
            _ => Self::Unknown,
        }
    }
}

pub struct Instruction(u32);

impl Instruction {
    pub fn opcode(&self) -> Opcode {
        match (self.0, self.0) {
            _ => (),
        }
        (self.0 as u8 & 0x7F).into()
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

    pub fn immediate(&self) -> Option<u32> {
        let ty = self.opcode().get_type()?;

        match ty {
            InstructionType::I => Some(
                (self.0 >> 20) & 0xFFF
                    | if self.0 >> 31 > 0 {
                        u32::MAX & 0xFFFFF800
                    } else {
                        0
                    },
            ),
            InstructionType::S => Some(
                (self.0 >> 7) & 0x1F
                    | ((self.0 >> 25) & 0x3F) << 5
                    | if self.0 >> 31 > 0 {
                        u32::MAX & 0xFFFFF800
                    } else {
                        0
                    },
            ),
            InstructionType::U => Some(
                (self.0 >> 12) & 0xFFFFF
                    | if self.0 >> 31 > 0 {
                        u32::MAX & 0xFFFFF000
                    } else {
                        0
                    },
            ),
            InstructionType::B => Some(
                ((self.0 >> 8) & 0xF) << 1
                    | ((self.0 >> 25) & 0x3F) << 6
                    | ((self.0 >> 7) & 1) << 11
                    | ((self.0 >> 31) & 1) << 12,
            ),
            InstructionType::J => Some(
                ((self.0 >> 21) & 0x7FF) << 1
                    | ((self.0 >> 20) & 1) << 10
                    | ((self.0 >> 12) & 0xFF) << 11
                    | ((self.0 >> 31) & 1) << 19
                    | if self.0 >> 31 > 0 {
                        u32::MAX & 0xFFF00000
                    } else {
                        0
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

pub enum InstructionType {
    R,
    R4,
    I,
    S,
    U,
    B,
    J,
}
