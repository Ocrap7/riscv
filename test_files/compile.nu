clang --target=riscv32 -march=rv32i .\main.c -c -o main.o
ld.lld main.o -o main -eentry

llvm-objdump -D -x main
llvm-objcopy -O binary -j .text -j .data -j .rodata main main.raw