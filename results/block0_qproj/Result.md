# Block-0 q-projection results (B0)

This page records the scalar RV64IMC baseline for the block-0 q-projection kernel
of TinyStories-3M / GPT-Neo. It covers one projection and one fixed input case;
it does not measure the complete Transformer block or full model.

## Run and evidence

Baseline reference: `20261007-imported-b0`.

- [Run metadata and hashes](runs/20261007-imported-b0/run.json)
- [Instruction profile](runs/20261007-imported-b0/profile.json)
- [Measured ELF](runs/20261007-imported-b0/qproj_test.elf)
- [Disassembly](runs/20261007-imported-b0/qproj_test.dis)
- [Experiment source and run instructions](../../experiments/block0_qproj/README.md)

This record was imported from the existing experiment during project cleanup.
The original execution time and compiler/Spike versions were not recorded.
Host and Spike correctness results and counters come from the supplied terminal
output; instruction counts come from the matching ELF and trace.

## Workload and correctness

| Item | Result |
|---|---|
| Baseline | B0, scalar implementation |
| Spike ISA | `rv64imc_zicsr_zifencei` |
| Input | INT8, `[1, 32, 128]`, zero point -5 |
| Weights | INT8, `[128, 128]`, zero point 0 |
| Output | INT8, `[1, 32, 128]`, zero point -17 |
| Bias | None |
| MACs | 32 × 128 × 128 = 524,288 |
| Output elements | 32 × 128 = 4,096 |
| Observed scalar MACs | 524,288; matches expected |
| Observed requantizations | 4,096; matches expected |
| Host correctness | 4,096 outputs checked; 0 mismatches; PASS |
| Spike correctness | 4,096 outputs checked; 0 mismatches; PASS |

Requantization uses multiplier `1205592262`, shift `39`, a rounding offset of
`1 << 38`, output zero point `-17`, and saturation to INT8. Correctness is
established for this fixed case.

## Instruction breakdown

| Part | Executions | Executed instructions | Share of kernel |
|---|---:|---:|---:|
| Inner MAC loop | 524,288 MACs | 4,194,304 | 98.742% |
| Requantization and saturation checks, excluding stores | 4,096 outputs | 24,576 | 0.579% |
| Output stores | 4,096 | 4,096 | 0.096% |
| Other setup and loop control | — | 24,781 | 0.583% |
| **Total kernel** | — | **4,247,757** | **100.000%** |

The profiler directly reports the inner loop and total kernel counts. The finer
breakdown follows the disassembly and executed mnemonic counts: six instructions
per output perform the requantization and saturation comparisons, followed by
one store. The two clamp-value assignments did not execute for this input case.
The remaining 24,781 instructions are calculated by subtraction.

Each inner-loop MAC executes eight instructions:

| Work inside one MAC iteration | Instructions per MAC |
|---|---:|
| Load input and weight | 2 |
| Increment input and weight pointers | 2 |
| Correct the input zero point | 1 |
| Multiply (`mulw`) | 1 |
| Accumulate (`addw`, compressed encoding) | 1 |
| Loop branch | 1 |
| **Total** | **8** |

## Instruction mix

| Category | Executed instructions | Share of kernel |
|---|---:|---:|
| integer ALU | 2,130,092 | 50.146% |
| load | 1,048,576 | 24.685% |
| branch/jump | 536,609 | 12.633% |
| multiply | 528,384 | 12.439% |
| store | 4,096 | 0.096% |

The 528,384 multiply instructions consist of 524,288 MAC multiplications and
4,096 requantization multiplications. Instruction mix and instruction breakdown
are two views of the same kernel; their totals must not be added together.

## Counters and derived metrics

| Metric | Value |
|---|---:|
| Spike `mcycle` | 4,247,768 |
| Spike `minstret` | 4,247,773 |
| Instructions counted inside `qproj_accumulate` | 4,247,757 |
| Inner-loop instructions per MAC | 8 |
| Whole-kernel instructions per MAC | 8.101954 |
| Whole-kernel instructions per output | 1037.050049 |
| Loads per MAC | 2 |

The counter interval includes 16 more retired instructions than the selected
function's trace count. These are different measurement scopes.

The Spike counter ratio `mcycle / minstret` is approximately 1.000. This is a
simulator counter ratio, not a measured CV-Wally hardware CPI. Separate cycles
for MACs, `mulw`, `addw`, loads, or requantization were not measured. The earlier
run's different `mcycle` value is not used in this baseline record.

## Analysis

- The inner MAC loop accounts for 98.742% of executed kernel instructions, so
  reducing work in that loop offers the largest instruction-count opportunity
  for this projection.
- A scalar MAC needs eight instructions. Multiplication and accumulation account
  for two; loads, pointer updates, zero-point correction, and branching account
  for six. A packed MAC design should be evaluated together with data layout and
  loop changes.
- Requantization and saturation checks account for about 0.579% of this kernel's
  instructions. Their benefit may differ for other kernels and shapes.
- These counts do not identify hardware stalls, cache behavior, memory bandwidth,
  or accelerator speedup. Hardware performance needs separate RTL measurements.
