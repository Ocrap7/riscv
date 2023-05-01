#![feature(try_blocks)]

use std::io::Write;

use crate::cpu::regs;

mod cpu;
mod ops;

const DEFAULT_SIZE: usize = 1024 * 1024;
const STACK_SIZE: usize = 1024 * 1024;

fn main() {
    let path = std::path::PathBuf::from("test_files/main");
    let file_data = std::fs::read(path).expect("Could not read file.");
    let slice = file_data.as_slice();
    let file = elf::ElfBytes::<elf::endian::AnyEndian>::minimal_parse(slice).expect("Open test1");

    let segs = file.segments().expect("Nosegs");
    let max_size = segs
        .iter()
        .filter(|seg| seg.p_type == 1)
        .fold(0usize, |max, seg| {
            if (seg.p_vaddr + seg.p_memsz) as usize > max {
                (seg.p_vaddr + seg.p_memsz) as usize
            } else {
                max
            }
        });

    println!("Entry: {:x}", file.ehdr.e_entry);

    let (memory_size, sp) = {
        let mut size = DEFAULT_SIZE;

        while max_size > size {
            size *= 16;
        }

        size += STACK_SIZE * 2;
        let sp = size / 4 * 4;

        (size, sp)
    };

    println!("Memory Size: {:x}b, Stack Pointer: {:x}", memory_size, sp);

    let mut memory = vec![0u8; memory_size];

    segs.iter().filter(|seg| seg.p_type == 1).for_each(|seg| {
        let data = file.segment_data(&seg).unwrap();
        memory[seg.p_vaddr as usize..seg.p_vaddr as usize + data.len()].copy_from_slice(data);
    });

    // let mut file = std::fs::File::create("out.hex").unwrap();
    // file.write_all(&memory).unwrap();

    let mut cpu = cpu::CPU::<u32>::new(
        memory.as_mut_slice(),
        &[(regs::sp, sp as u32), (regs::pc, file.ehdr.e_entry as u32)],
    );
    cpu.continute();
    println!("{:?}", cpu);
}
