# Retired-instruction analysis

Measured comparison of the scalar and combined XTILE/XREQUANT backends for the eight-block transformer chain.

The MAC estimates and interpretation were prepared with ChatGPT using measured Spike counts and compiled disassembly.

## Main result

For the single benchmark case across blocks 0–7, the whole-chain raw count falls from **461,087,565** to **36,302,539** retired instructions: **92.1268% fewer**, or a **12.701× scalar/custom instruction ratio**. Both uploaded profile logs report successful chain, block and stage output checks. These files establish one measured benchmark case; they do not establish an eight-case profiling run.

No counter-overhead correction is applied. Adjacent reads report 1 instruction and the eight-NOP sanity check reports 9 in each log. Chain, full-block and individual-stage intervals are separate measurements, so their sums need not be identical.

The full stage table and exact per-block values are in `../results/instruction_comparison_benchmark/report.md` and `dynamic_regions.csv`.

## What the assembly shows

The scalar disassembly contains no XTILE/XREQUANT opcodes. The custom disassembly contains five XTILE instruction sites (one for each operation) and 25 XREQUANT sites (five operations across five buffer wrappers). These are static assembly sites, not dynamic execution counts.

The scalar dense accumulation inner loop is:

```asm
80000a86: lb    t3,0(a0)
80000a8a: lb    t1,0(t4)
80000a8e: addi  a0,a0,1
80000a90: subw  t3,t3,a7
80000a94: mulw  t1,t1,t3
80000a98: addi  t4,t4,1
80000a9a: addw  t5,t1,t5
80000a9e: bne   t6,a0,80000a86
```

This compiled loop retires eight instructions per multiply-accumulate, including loads, input-zero-point subtraction, pointer advances and the loop branch. From the fixed matrix shapes, six dense projections perform 6,291,456 MACs per block, or 50,331,648 across eight blocks. The loop therefore accounts for 402,653,184 instructions, approximately 87.33% of the scalar whole-chain count. This is a calculation from the observed loop and dimensions, not an instruction-category trace.

The custom dense path instead contains these encoded instructions (names annotated from opcode/funct3):

```asm
80000a78: 0007800b  xtile_cfg
80000a7c: 00b5100b  xtile_set_ab
80000a80: 0006a00b  xtile_set_c
80000a84: 01c8b00b  xtile_set_zp
80000a88: 0000400b  xtile_run
```

Configuration preparation and software bias handling still surround these sites. One RUN performs the configured matrix operation inside Spike; internal tile work is not retired as separate scalar instructions. Similarly, XREQUANT replaces per-element multiplication, rounding, shifting, zero-point addition, clamping and stores with buffer operations. The static wrappers are shown in the supplied `scalar_kernels.dis` and `custom_kernels.dis`.

This explains why QKV falls from 102,736,208 to 1,636 retired instructions. It does not imply that the hardware completes those matrix operations in 1,636 cycles.

## Remaining work indicated by the profile

Attention contributes **63.3868%** of custom stage intervals; MLP expansion plus GELU contributes **26.7336%**. Together they account for **90.1204%**. Dense projections and buffer requantization have shifted the software bottleneck toward those stages.

The MLP/GELU stage still includes both expansion and activation processing. A substage measurement would be required to quantify GELU alone. The modest attention reduction is consistent with buffer requantization at its output while attention computation remains scalar.

The current result measures the combined XTILE/XREQUANT backend. It does not separately measure each extension's contribution, physical accelerator latency, memory-system costs or RTL speedup.

## Analysis verification

Six local analyzer tests passed, covering encoded custom operations, compressed instructions, normal/joined headers, completeness and aggregate checks, missing/extra/duplicate records, declared case counts and CLI output generation. Test fixtures are explicitly synthetic and are not used for the reported performance numbers.

The corrected analyzer was also run against the actual uploaded logs/disassemblies. Their SHA-256 digests and the supplied source-file digests are recorded in `../results/instruction_comparison_benchmark/input_provenance.json`.
