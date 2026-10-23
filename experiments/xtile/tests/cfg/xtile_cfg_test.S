    .text
    .balign 4

    .globl xtile_cfg_test_execute
    .type xtile_cfg_test_execute, @function

xtile_cfg_test_execute:
    .globl xtile_cfg_test_insn
xtile_cfg_test_insn:
    .insn r 0x0b, 0, 0, x0, a0, x0
    ret

    .size xtile_cfg_test_execute, .-xtile_cfg_test_execute

    .section .note.GNU-stack,"",@progbits