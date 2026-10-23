    .text
    .balign 4

    .globl xtile_run_test_execute
    .type xtile_run_test_execute, @function

xtile_run_test_execute:
    .globl xtile_run_test_insn
xtile_run_test_insn:
    .insn r 0x0b, 4, 0, x0, x0, x0
    ret

    .size xtile_run_test_execute, .-xtile_run_test_execute

    .section .note.GNU-stack,"",@progbits