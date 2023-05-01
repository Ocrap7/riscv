
const int CONSTANT = 69;

void entry() {
    int i = 37;
    int *p = &CONSTANT;
    // while (1) {
        i += *p;
    // }
    asm volatile ("ebreak");
}