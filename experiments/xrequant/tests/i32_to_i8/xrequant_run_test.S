    .text
    .balign 4
    .globl xrequant_run_test_execute
    .type xrequant_run_test_execute, @function

xrequant_run_test_execute:
    .globl xrequant_run_test_insn
xrequant_run_test_insn:
    .insn r 0x2b, 4, 0, x0, x0, x0
    ret

    .size xrequant_run_test_execute, .-xrequant_run_test_execute
    .section .note.GNU-stack,"",@progbits