# Block-0 q-projection results — B0

Scalar INT8 q-projection baseline for TinyStories-3M / GPT-Neo on bare-metal RV64IMC.

This experiment measures block-0 q-projection for one fixed input case. It does not measure the complete Transformer block or full model.

## Latest verified run

| Item | Value |
|---|---|
| Experiment | `block0_qproj` |
| Variant | B0 — scalar implementation |
| Run ID | `b0-refresh-01` |
| Execution date | 2026-10-07 |
| Recorded status | `passed` |
| Recorded Git commit, abbreviated | `13ccb84770ff` |
| Earlier reference | `20261007-imported-b0` |

Saved evidence:

- [Run metadata, environment, snapshots, and hashes](runs/b0-refresh-01/run.json)
- [Instruction profile](runs/b0-refresh-01/profile.json)
- [Build and correctness log](runs/b0-refresh-01/01-command.log)
- [Disassembly and trace-generation log](runs/b0-refresh-01/02-command.log)
- [Profiling log](runs/b0-refresh-01/03-command.log)
- [Measured ELF](runs/b0-refresh-01/qproj_test.elf)
- [Disassembly](runs/b0-refresh-01/qproj_test.dis)
- [Source and input snapshots](runs/b0-refresh-01/snapshot/)
- [Experiment implementation and instructions](../../experiments/block0_qproj/README.md)



The raw trace remains in
`experiments/block0_qproj/build/runs/b0-refresh-01/qproj_trace.log`
relative to the project root. Its hash is recorded in the run metadata.

## Workload and arithmetic

| Item | Value |
|---|---|
| Input tensor | T45, INT8 `[1, 32, 128]` |
| Weight tensor | T27, INT8 `[128, 128]` |
| Output tensor | T46, INT8 `[1, 32, 128]` |
| Input zero point | -5 |
| Weight zero point | 0 |
| Output zero point | -17 |
| Bias | None |
| Accumulator | INT32 |
| Expected MACs | 32 × 128 × 128 = 524,288 |
| Expected output elements | 32 × 128 = 4,096 |
| Requantization multiplier | 1,205,592,262 |
| Requantization shift | 39 |

For each token `t` and output feature `o`:

```text
acc[t,o] = sum_k ((x_q[t,k] - (-5)) * w_q[o,k])

scaled = ((int64_t)acc[t,o] * 1205592262 + (1LL << 38)) >> 39
y_q[t,o] = clamp(scaled - 17, -128, 127)
```

The input fixture is `artifacts/exports/q_proj_case0.npz`. 
Its matching model is `artifacts/models/block0_w8a8_baseline.tflite`.

The C executable consumes the frozen headers at runtime. Correctness is established for this fixed case; general equivalence to LiteRT rounding for every possible accumulator has not been established.

## Correctness results

| Check | Expected | Observed | Result |
|---|---:|---:|---|
| Host INT8 output checks | 4,096 | 4,096; zero mismatches | PASS |
| Spike INT8 output checks | 4,096 | 4,096; zero mismatches | PASS |
| Scalar MAC executions | 524,288 | 524,288 | MATCH |
| Requantization executions | 4,096 | 4,096 | MATCH |

The Spike execution used to generate the trace also passed its correctness test.

## Instruction breakdown

The measured function is `qproj_accumulate`, covering
`0x80000136` through `0x800001c2`, with the end address excluded.

| Part | Executions | Executed instructions | Share of kernel |
|---|---:|---:|---:|
| Inner MAC loop | 524,288 MACs | 4,194,304 | 98.742% |
| Requantization and saturation checks, excluding stores | 4,096 outputs | 24,576 | 0.579% |
| Output stores | 4,096 | 4,096 | 0.096% |
| Other setup and loop control | — | 24,781 | 0.583% |
| **Total kernel** | — | **4,247,757** | **100.000%** |

The profiler directly reports:

- Inner-loop instructions: 4,194,304.
- Outside-inner-loop instructions: 53,453.
- Total kernel instructions: 4,247,757.

The finer outside-loop breakdown follows the disassembly and executed mnemonic counts. Each output executes six instructions for requantization and saturation comparisons, followed by one store.

The six instructions are multiplication, rounding-offset addition, arithmetic shift, output-zero-point addition, and two saturation comparisons. Neither clamp-value assignment executed for this case.

The remaining setup/control count is calculated as:

```text
4,247,757 - 4,194,304 - 24,576 - 4,096 = 24,781
```

### Instructions inside one MAC iteration

The inner loop covers `0x80000178` through `0x80000190`, excluding the end.

| Work | Instruction | Instructions per MAC |
|---|---|---:|
| Load input and weight | Two `lb` instructions | 2 |
| Advance input and weight pointers | Two `c.addi` instructions | 2 |
| Correct input zero point | `c.addiw` | 1 |
| Multiply | `mulw` | 1 |
| Accumulate | `c.addw` | 1 |
| Loop branch | `bne` | 1 |
| **Total** | | **8** |

Therefore:

```text
524,288 MACs × 8 instructions/MAC = 4,194,304 instructions
```

## Instruction mix

| Category | Executed instructions | Share of kernel |
|---|---:|---:|
| Integer ALU | 2,130,092 | 50.146% |
| Load | 1,048,576 | 24.685% |
| Branch/jump | 536,609 | 12.633% |
| Multiply | 528,384 | 12.439% |
| Store | 4,096 | 0.096% |
| **Total** | **4,247,757** | **100.000%** |

The multiply count contains:

- 524,288 `mulw` executions for MACs.
- 4,096 `mul` executions for requantization.

The instruction mix and functional breakdown describe the same kernel.
Their totals must not be added together.

## Counters and derived metrics

| Metric | Value |
|---|---:|
| Spike `mcycle` | 4,247,768 |
| Spike `minstret` | 4,247,773 |
| Instructions inside the measured function | 4,247,757 |
| Inner-loop instructions per MAC | 8 |
| Whole-kernel instructions per MAC | 8.101954 |
| Whole-kernel instructions per output | 1,037.050049 |
| Loads per MAC | 2 |
| Stores per MAC | 0.0078125 |
| Memory instructions per MAC | 2.0078125 |
| Spike counter ratio: `mcycle / minstret` | Approximately 1.000 |

Derived instruction metrics use:

```text
Instructions per MAC = kernel instructions / MACs
Instructions per output = kernel instructions / output elements
Memory instructions per MAC = (loads + stores) / MACs
```

The counter interval includes 16 more retired instructions than the selected function's trace count. These are different measurement scopes.

The Spike counter ratio is not a measured CV-Wally hardware CPI. Separate cycles for MACs, loads, `mulw`, `addw`, or requantization were not measured. Instruction counts do not establish hardware latency.

## ELF section sizes

| Section | Bytes |
|---|---:|
| Text, including read-only data | 27,408 |
| Data | 72 |
| BSS | 4,128 |

The measured function itself occupies 140 bytes.

These section sizes do not represent complete runtime memory usage, including stack requirements.

## Comparison with the imported baseline

| Measurement | `20261007-imported-b0` | `b0-refresh-01` |
|---|---:|---:|
| Kernel instructions | 4,247,757 | 4,247,757 |
| Inner-loop instructions | 4,194,304 | 4,194,304 |
| Observed MACs | 524,288 | 524,288 |
| Requantizations | 4,096 | 4,096 |
| Spike `mcycle` | 4,247,768 | 4,247,768 |
| Spike `minstret` | 4,247,773 | 4,247,773 |
| Recorded Git commit, abbreviated | `6b5d84ba4a25` | `13ccb84770ff` |


The refreshed run reproduces the imported baseline's instruction measurements and Spike counters. It provides a newly executed run with captured metadata and logs; it is not a performance optimization.

## Analysis

1. **The inner MAC loop dominates this projection's instruction count.**
   It accounts for 98.742% of executed kernel instructions. Reducing work in that loop offers the largest instruction-count opportunity here.

2. **A scalar MAC requires more than multiplication and accumulation.**
   Those two operations account for two of the eight instructions. Loads, pointer updates, zero-point correction, and branching account for the other six. Evaluate packed MAC instructions together with data layout and loop changes.

3. **Requantization has a small instruction share for this workload.**
   Requantization and saturation checks account for approximately 0.579%. Other kernel shapes may have different proportions.

4. **Hardware bottlenecks remain to be measured.**
   This experiment does not determine cache behavior, load stalls, memory bandwidth, physical latency, or accelerator speedup. CV-Wally RTL measurements are needed for hardware cycle analysis.

## Reproduce another run

From the project root, with the existing virtual environment active, choose an unused run ID:

```bash
python3 scripts/run_experiment.py block0_qproj \
  --run-id b0-refresh-02 --dry-run

python3 scripts/run_experiment.py block0_qproj \
  --run-id b0-refresh-02

python3 scripts/list_runs.py
```

After verification, update this parent README with the latest complete results and evidence links. Keep earlier run directories for comparison.

Use separate experiment IDs for software or ISA variants, and identify the specific B0 run used as their comparison reference.