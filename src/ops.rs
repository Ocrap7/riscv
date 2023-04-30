use crate::cpu::Opcode;

macro_rules! op {
    ($name:ident, $opcode:ident;$($body:tt)*) => {
        pub const $name: (Opcode, u8, u8, u8) = (Opcode::$opcode, 0, 0, 0);
        op!{$($body)*}
    };
    ($name:ident, $opcode:ident, f3:$f32:expr;$($body:tt)*) => {
        pub const $name: (Opcode, u8, u8, u8) = (Opcode::$opcode, $f32, 0, 0);
        op!{$($body)*}
    };
    ($name:ident, $opcode:ident, f3:$f32:expr, f7:$f7:expr;$($body:tt)*) => {
        pub const $name: (Opcode, u8, u8, u8) = (Opcode::$opcode, $f32, $f7, 0);
        op!{$($body)*}
    };
    ($name:ident, $opcode:ident, f3:$f32:expr, f2:$f2:expr;$($body:tt)*) => {
        pub const $name: (Opcode, u8, u8, u8) = (Opcode::$opcode, $f32, 0, $f2);
        op!{$($body)*}
    };
    () => {};
}

op! {
    /* RV32I Base Instruction Set */
    LUI, LUI;
    AUIPC, AUIPC;
    JAL, JAL;
    JALR, JALR, f3:0b000;

    BEQ, Branch, f3:0b000;
    BNE, Branch, f3:0b001;
    BLT, Branch, f3:0b100;
    BGE, Branch, f3:0b101;
    BLTU, Branch, f3:0b110;
    BGUE, Branch, f3:0b111;

    LB, Load, f3:0b000;
    LH, Load, f3:0b001;
    LW, Load, f3:0b010;
    LBU, Load, f3:0b100;
    LHU, Load, f3:0b101;

    SB, Store, f3:0b000;
    SH, Store, f3:0b000;
    SW, Store, f3:0b000;

    ADDI, OpImm, f3:0b000;
    SLTI, OpImm, f3:0b010;
    SLTIU, OpImm, f3:0b011;
    XORI, OpImm, f3:0b100;
    ORI, OpImm, f3:0b110;
    ANDI, OpImm, f3:0b111;

    SLLI, OpImm, f3:0b001, f7:0b0000000;
    SRLI, OpImm, f3:0b101, f7:0b0000000;
    SRAI, OpImm, f3:0b101, f7:0b0100000;
    ADD, OpImm, f3:0b000, f7:0b0000000;
    SUB, OpImm, f3:0b000, f7:0b0100000;
    SLL, OpImm, f3:0b001, f7:0b0000000;
    SLT, OpImm, f3:0b010, f7:0b0000000;
    SLTU, OpImm, f3:0b011, f7:0b0000000;
    XOR, OpImm, f3:0b100, f7:0b0000000;
    SRL, OpImm, f3:0b101, f7:0b0000000;
    SRA, OpImm, f3:0b101, f7:0b0100000;
    OR, OpImm, f3:0b110, f7:0b0000000;
    AND, OpImm, f3:0b111, f7:0b0000000;

    FENCE, MiscMem, f3:0b000, f7:0b0000000;
    ECALL, System, f3:0b000, f7:0b0000000;
    EBREAK, System, f3:0b000, f7:0b0100000;


    /* RV64I Base Instruction Set */
    LWU, Load, f3:0b110;
    LD, Load, f3:0b011;

    SD, Store, f3:0b011;

    ADDIW, OpImm32, f3:0b000;
    SLLIW, OpImm32, f3:0b001, f7:0b0000000;
    SRLIW, OpImm32, f3:0b101, f7:0b0000000;
    SRAIW, OpImm32, f3:0b101, f7:0b0100000;

    ADDW, Op, f3:0b000, f7:0b0000000;
    SUBW, Op, f3:0b000, f7:0b0100000;
    SLLW, Op, f3:0b001, f7:0b0000000;
    SRLW, Op, f3:0b101, f7:0b0000000;
    SRAW, Op, f3:0b101, f7:0b0100000;

    /* RV32/RV64 Zifencei Standard Extension */
    FENCEI, MiscMem, f3:0b001, f7:0b0000000;

    /* RV32/RV64 Zicsr Standard Extension */
    CSRRW, System, f3:0b001, f7:0b0000000;
    CSRRS, System, f3:0b010, f7:0b0000000;
    CSRRC, System, f3:0b011, f7:0b0000000;
    CSRRWI, System, f3:0b101, f7:0b0000000;
    CSRRSI, System, f3:0b110, f7:0b0000000;
    CSRRCI, System, f3:0b111, f7:0b0000000;

}

op! {
    /* RV32M Standard Extension */
    MUL, Op, f3:0b000, f7:0b0000001;
    MULH, Op, f3:0b001, f7:0b0000001;
    MULHSU, Op, f3:0b010, f7:0b0000001;
    MULHU, Op, f3:0b011, f7:0b0000001;
    DIV, Op, f3:0b100, f7:0b0000001;
    DIVU, Op, f3:0b101, f7:0b0000001;
    REM, Op, f3:0b110, f7:0b0000001;
    REMU, Op, f3:0b111, f7:0b0000001;

    /* RV64M Standard Extension */
    MULW, Op32, f3:0b000, f7:0b0000001;
    DIVW, Op32, f3:0b100, f7:0b0000001;
    DIVUW, Op32, f3:0b101, f7:0b0000001;
    REMW, Op32, f3:0b110, f7:0b0000001;
    REMUW, Op32, f3:0b111, f7:0b0000001;
}

op! {
    /* RV32A Standard Extension */
    LRW, Amo, f3:0b010, f7:0b0001000;
    SCW, Amo, f3:0b010, f7:0b0011000;
    AMOSWAPW, Amo, f3:0b010, f7:0b0000100;
    AMOADDW, Amo, f3:0b010, f7:0b0000000;
    AMOXORW, Amo, f3:0b010, f7:0b0010000;
    AMOORW, Amo, f3:0b010, f7:0b0110000;
    AMOMINW, Amo, f3:0b010, f7:0b1000000;
    AMOMAXW, Amo, f3:0b010, f7:0b1010000;
    AMOMINUW, Amo, f3:0b010, f7:0b1100000;
    AMOMAXUW, Amo, f3:0b010, f7:0b1110000;

    /* RV64A Standard Extension */
    LRD, Amo, f3:0b011, f7:0b0001000;
    SCD, Amo, f3:0b011, f7:0b0011000;
    AMOSWAPD, Amo, f3:0b011, f7:0b0000100;
    AMOADDD, Amo, f3:0b011, f7:0b0000000;
    AMOXORD, Amo, f3:0b011, f7:0b0010000;
    AMOORD, Amo, f3:0b011, f7:0b0110000;
    AMOMIND, Amo, f3:0b011, f7:0b1000000;
    AMOMAXD, Amo, f3:0b011, f7:0b1010000;
    AMOMINUD, Amo, f3:0b011, f7:0b1100000;
    AMOMAXUD, Amo, f3:0b011, f7:0b1110000;
}

op! {
    /* RV32F Standard Extension */
    FLW, Load, f3:0b010;
    FSW, Store, f3:0b010;
    FMADDS, MAdd, f3:0b000, f2:0b00;
    FMSUBS, MSub, f3:0b000, f2:0b00;
    FNMADDS, NMAdd, f3:0b000, f2:0b00;
    FNMSUBS, NMSub, f3:0b000, f2:0b00;
    FADDS, OpFP, f3:0b000, f7:0b0000000;
    FSUBS, OpFP, f3:0b000, f7:0b0000100;
    FMULS, OpFP, f3:0b000, f7:0b0001000;
    FDIVS, OpFP, f3:0b000, f7:0b0001100;
    FSQRTS, OpFP, f3:0b000, f7:0b0101100;
    FSGNJS, OpFP, f3:0b000, f7:0b0010000;
    FSGNJNS, OpFP, f3:0b001, f7:0b0010000;
    FSGNJXS, OpFP, f3:0b010, f7:0b0010000;
    FMINS, OpFP, f3:0b000, f7:0b0010100;
    FMAXS, OpFP, f3:0b001, f7:0b0010100;
    FCVTWS, OpFP, f3:0b000, f7:0b1100000;
    FCVTWUS, OpFP, f3:0b000, f7:0b1100000;
    FMVXW, OpFP, f3:0b000, f7:0b1110000;
    FEQS, OpFP, f3:0b010, f7:0b1010000;
    FLTS, OpFP, f3:0b001, f7:0b1010000;
    FLES, OpFP, f3:0b000, f7:0b1010000;
    FCLASSS, OpFP, f3:0b001, f7:0b1110000;
    FCVTSW, OpFP, f3:0b000, f7:0b1101000;
    FCVTSWU, OpFP, f3:0b000, f7:0b1101000;
    FMVWX, OpFP, f3:0b000, f7:0b1111000;


    /* RV64F Standard Extension */
    FCVTLS, OpFP, f3:0, f7:0b1100000;
    FCVTLUS, OpFP, f3:0, f7:0b1100000;
    FCVTSL, OpFP, f3:0, f7:0b1101000;
    FCVTSLU, OpFP, f3:0, f7:0b1101000;
}

op! {
    /* RV32D Standard Extension */
    FLD, Load, f3:0b011;
    FSD, Store, f3:0b011;
    FMADDD, MAdd, f3:0b000, f2:0b01;
    FMSUBD, MSub, f3:0b000, f2:0b01;
    FNMADDD, NMAdd, f3:0b000, f2:0b01;
    FNMSUBD, NMSub, f3:0b000, f2:0b01;
    FADDD, OpFP, f3:0b000, f7:0b0000001;
    FSUBD, OpFP, f3:0b000, f7:0b0000101;
    FMULD, OpFP, f3:0b000, f7:0b0001001;
    FDIVD, OpFP, f3:0b000, f7:0b0001101;
    FSQRTD, OpFP, f3:0b000, f7:0b0101101;
    FSGNJD, OpFP, f3:0b000, f7:0b0010001;
    FSGNJND, OpFP, f3:0b001, f7:0b0010001;
    FSGNJXD, OpFP, f3:0b010, f7:0b0010001;
    FMIND, OpFP, f3:0b000, f7:0b0010101;
    FMAXD, OpFP, f3:0b001, f7:0b0010101;
    FCVTSD, OpFP, f3:0b000, f7:0b0100000;
    FCVTDS, OpFP, f3:0b000, f7:0b0100001;
    FEQD, OpFP, f3:0b010, f7:0b1010001;
    FLTD, OpFP, f3:0b001, f7:0b1010001;
    FLED, OpFP, f3:0b000, f7:0b1010001;
    FCLASSD, OpFP, f3:0b001, f7:0b1110001;
    FCVTWD, OpFP, f3:0b000, f7:0b1101001;
    FCVTWUD, OpFP, f3:0b000, f7:0b1101001;
    FCVTDW, OpFP, f3:0b000, f7:0b1101001;
    FCVTDWU, OpFP, f3:0b000, f7:0b1101001;

    /* RV64D Standard Extension */
    FCVTLD, OpFP, f3:0b000, f7:0b1100001;
    FCVTLUD, OpFP, f3:0b000, f7:0b1100001;
    FMVXD, OpFP, f3:0b000, f7:0b1110001;
    FCVTDL, OpFP, f3:0b000, f7:0b1101001;
    FCVTDLU, OpFP, f3:0b000, f7:0b1101001;
    FMVXDX, OpFP, f3:0b000, f7:0b1111001;
}

op! {
    /* RV32Q Standard Extension */
    FLQ, Load, f3:0b011;
    FSQ, Store, f3:0b011;
    FMADDQ, MAdd, f3:0b000, f2:0b11;
    FMSUBQ, MSub, f3:0b000, f2:0b11;
    FNMADDQ, NMAdd, f3:0b000, f2:0b11;
    FNMSUBQ, NMSub, f3:0b000, f2:0b11;
    FADDQ, OpFP, f3:0b000, f7:0b0000011;
    FSUBQ, OpFP, f3:0b000, f7:0b0000111;
    FMULQ, OpFP, f3:0b000, f7:0b0001011;
    FDIVQ, OpFP, f3:0b000, f7:0b0001111;
    FSQRTQ, OpFP, f3:0b000, f7:0b0101111;
    FSGNJQ, OpFP, f3:0b000, f7:0b0010011;
    FSGNJNQ, OpFP, f3:0b001, f7:0b0010011;
    FSGNJXQ, OpFP, f3:0b010, f7:0b0010011;
    FMINQ, OpFP, f3:0b000, f7:0b0010111;
    FMAXQ, OpFP, f3:0b001, f7:0b0010111;
    FCVTSQ, OpFP, f3:0b000, f7:0b0100000;
    FCVTQS, OpFP, f3:0b000, f7:0b0100011;
    FCVTDQ, OpFP, f3:0b000, f7:0b0100001;
    FCVTQD, OpFP, f3:0b000, f7:0b0100011;
    FEQQ, OpFP, f3:0b010, f7:0b1010011;
    FLTQ, OpFP, f3:0b001, f7:0b1010011;
    FLEQ, OpFP, f3:0b000, f7:0b1010011;
    FCLASSQ, OpFP, f3:0b001, f7:0b1110011;
    FCVTWQ, OpFP, f3:0b000, f7:0b1101011;
    FCVTWUQ, OpFP, f3:0b000, f7:0b1101011;
    FCVTQW, OpFP, f3:0b000, f7:0b1101011;
    FCVTQWU, OpFP, f3:0b000, f7:0b1101011;

    /* RV64Q Standard Extension */
    FCVTLQ, OpFP, f3:0, f7:0b1100011;
    FCVTLUQ, OpFP, f3:0, f7:0b1100011;
    FCVTQL, OpFP, f3:0, f7:0b1101011;
    FCVTQLU, OpFP, f3:0, f7:0b1101011;
}
