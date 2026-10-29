#ifndef INTEGER_INFERENCE_XTILE_H
#define INTEGER_INFERENCE_XTILE_H

#include <stdint.h>

/*
 * Include this header only in the RV64 implementation.
 * Host builds will use the separate scalar path.
 */
#if !defined(__riscv) || !defined(__riscv_xlen)
#error "XTILE instruction wrappers require a RISC-V target"
#elif __riscv_xlen != 64
#error "XTILE instruction wrappers require RV64"
#endif

/*
 * Retain the packed configuration.
 * The caller must validate fields before packing them.
 */
static inline void xtile_cfg(uint64_t cfg)
{
    __asm__ volatile(
        ".insn r 0x0b, 0, 0, x0, %0, x0"
        :
        : "r"(cfg)
    );
}

/* Retain addresses of A[M,K] and BT[N,K]. */
static inline void xtile_set_ab(const void *a, const void *bt)
{
    __asm__ volatile(
        ".insn r 0x0b, 1, 0, x0, %0, %1"
        :
        : "r"((uintptr_t)a), "r"((uintptr_t)bt)
    );
}

/* Retain the address of C[M,N]. */
static inline void xtile_set_c(void *c)
{
    __asm__ volatile(
        ".insn r 0x0b, 2, 0, x0, %0, x0"
        :
        : "r"((uintptr_t)c)
    );
}

/* Retain signed zero points from the low 32 bits. */
static inline void xtile_set_zp(int32_t a_zp, int32_t b_zp)
{
    __asm__ volatile(
        ".insn r 0x0b, 3, 0, x0, %0, %1"
        :
        : "r"((intptr_t)a_zp), "r"((intptr_t)b_zp)
    );
}

/*
 * Compute the whole configured multiplication.
 *
 * INT8/INT8 produces INT32 output.
 * Any mode involving INT32 produces INT64 output.
 *
 * C must not overlap A or BT.
 * Bias and requantization are handled separately.
 */
static inline void xtile_run(void)
{
    __asm__ volatile(
        ".insn r 0x0b, 4, 0, x0, x0, x0"
        :
        :
        : "memory"
    );
}

#endif