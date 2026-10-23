    .text
    .balign 4
    .globl xrequant_cfg_test_execute
    .type xrequant_cfg_test_execute, @function

xrequant_cfg_test_execute:
    .globl xrequant_cfg_test_insn
xrequant_cfg_test_insn:
    .insn r 0x2b, 0, 0, x0, a0, x0
    ret

    .size xrequant_cfg_test_execute, .-xrequant_cfg_test_execute
    .section .note.GNU-stack,"",@progbits