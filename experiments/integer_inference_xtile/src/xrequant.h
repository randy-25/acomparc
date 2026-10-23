#ifndef INTEGER_INFERENCE_XREQUANT_H
#define INTEGER_INFERENCE_XREQUANT_H

#include <stdint.h>

#if !defined(__riscv) || !defined(__riscv_xlen)
#error "XREQUANT wrappers require a RISC-V target"
#elif __riscv_xlen != 64
#error "XREQUANT wrappers require RV64"
#endif

static inline void xrequant_cfg(uint64_t cfg)
{
    __asm__ volatile(
        ".insn r 0x2b, 0, 0, x0, %0, x0"
        : : "r"(cfg)
    );
}

static inline void xrequant_set_io(
    const void *src,
    void *dst
)
{
    __asm__ volatile(
        ".insn r 0x2b, 1, 0, x0, %0, %1"
        : : "r"((uintptr_t)src), "r"((uintptr_t)dst)
    );
}

static inline void xrequant_set_mul(int64_t multiplier)
{
    __asm__ volatile(
        ".insn r 0x2b, 2, 0, x0, %0, x0"
        : : "r"(multiplier)
    );
}

static inline void xrequant_set_zp(
    int32_t input_zp,
    int32_t output_zp
)
{
    __asm__ volatile(
        ".insn r 0x2b, 3, 0, x0, %0, %1"
        : : "r"((intptr_t)input_zp),
            "r"((intptr_t)output_zp)
    );
}

static inline void xrequant_run(void)
{
    __asm__ volatile(
        ".insn r 0x2b, 4, 0, x0, x0, x0"
        : : : "memory"
    );
}

#endif