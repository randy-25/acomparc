# Block 0 Profiling Results

## Measurement

- ISA: `rv64imc_zicsr`
- Simulator: Spike
- Counter: `minstret` (retired instructions)
- Workload: one block-0 inference, sequence length 32, hidden size 128.
- Cases: one benchmark and seven calibration sequences.
- Counter sanity check: adjacent reads = 1; eight NOPs between reads = 9.

Counts in the measurement tables below are raw counter differences, including small measurement and call overhead. The analysis section separately labels corrected stage-body counts. They are instruction counts, not hardware cycle counts.

## Full Block

| Case | block0_forward | Sum of seven stage measurements |
|---|---:|---:|
| 0 | 57,181,526 | 57,181,517 |
| 1 | 57,181,531 | 57,181,522 |
| 2 | 57,181,638 | 57,181,629 |
| 3 | 57,181,373 | 57,181,364 |
| 4 | 57,181,491 | 57,181,482 |
| 5 | 57,181,396 | 57,181,387 |
| 6 | 57,181,509 | 57,181,500 |
| 7 | 57,181,316 | 57,181,307 |
| **Mean** | **57,181,472.5** | **57,181,463.5** |

The full measurement exceeds the stage sum by 9 instructions in every case because the wrapper and measurement call sites execute different surrounding instructions.

## Individual Stages

| Case | P1: LN1 | P2: QKV | P3: Attention | P4: Out projection + residual | P5: LN2 | P6: MLP expansion + GELU | P7: MLP output + residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 256,445 | 12,842,003 | 2,922,571 | 4,366,724 | 256,384 | 19,595,945 | 16,941,445 |
| 1 | 256,447 | 12,842,005 | 2,922,550 | 4,366,724 | 256,406 | 19,595,945 | 16,941,445 |
| 2 | 256,458 | 12,842,004 | 2,922,686 | 4,366,724 | 256,371 | 19,595,943 | 16,941,443 |
| 3 | 256,488 | 12,842,002 | 2,922,376 | 4,366,725 | 256,385 | 19,595,945 | 16,941,443 |
| 4 | 256,485 | 12,842,002 | 2,922,492 | 4,366,724 | 256,393 | 19,595,943 | 16,941,443 |
| 5 | 256,453 | 12,842,005 | 2,922,449 | 4,366,725 | 256,364 | 19,595,948 | 16,941,443 |
| 6 | 256,484 | 12,842,004 | 2,922,523 | 4,366,724 | 256,380 | 19,595,942 | 16,941,443 |
| 7 | 256,449 | 12,842,003 | 2,922,370 | 4,366,724 | 256,375 | 19,595,942 | 16,941,444 |
| **Mean** | **256,463.625** | **12,842,003.5** | **2,922,502.125** | **4,366,724.25** | **256,382.25** | **19,595,944.125** | **16,941,443.625** |

Means are arithmetic averages across these eight fixtures.

## Instruction Distribution by Stage

The percentages below use the mean raw instruction count across eight cases, divided by the mean full-block count of **57,181,472.5 instructions**.

| Part | Stage | Share of full-block instructions |
|---|---|---:|
| 1 | First layer normalization | 0.45% |
| 2 | Q/K/V projections | 22.46% |
| 3 | Attention | 5.11% |
| 4 | Attention output projection + residual | 7.64% |
| 5 | Second layer normalization | 0.45% |
| 6 | MLP expansion + GELU | 34.27% |
| 7 | MLP output projection + residual | 29.63% |

Percentages are rounded and include the small overhead present in the raw measurements.

Parts **2, 6, and 7** together account for approximately **86.36%** of the full-block instruction count.

## Analysis Findings

The following table uses **case 0, stage bodies including their callees**, with measurement/caller instructions excluded. `Other` is everything except the listed multiply and register-add opcodes. Register-add totals include address calculations as well as numerical additions; mathematical MAC counts appear separately.

| Part and detailed analysis | Matrix MACs | `mul` + `mulw` | `add` + `addw` | Other instructions | Stage-body total |
|---|---|---|---|---|---|
| [1: First Layer Normalization](ln1_forward.md) | 0 | 28,672 | 30,158 | 197,610 | 256,440 |
| [2: QKV Projection](qkv_forward.md) | 1,572,864 | 1,585,152 | 1,622,409 | 9,634,435 | 12,841,996 |
| [3: Attention](attention_forward.md) | 262,144 | 283,143 | 485,939 | 2,153,481 | 2,922,563 |
| [4: Attention Output Projection and First Residual](out_residual_forward.md) | 524,288 | 536,576 | 553,093 | 3,277,048 | 4,366,717 |
| [5: Second Layer Normalization](ln2_forward.md) | 0 | 28,672 | 30,017 | 197,690 | 256,379 |
| [6: MLP Expansion and GELU](mlp_gelu_forward.md) | 2,097,152 | 2,359,296 | 2,490,536 | 14,746,108 | 19,595,940 |
| [7: MLP Output Projection and Second Residual](mlp_out_residual_forward.md) | 2,097,152 | 2,105,344 | 2,125,957 | 12,710,136 | 16,941,437 |
| **Seven stage bodies** | 6,553,600 | 6,926,855 | 7,338,109 | 42,916,508 | 57,181,472 |

The full wrapper adds **48 instructions** to those stage bodies; the full measurement adds **6 caller/counter instructions**. Thus the case-0 raw full count is **57,181,526**, and the sum of raw stage measurements is **57,181,517**. The constant difference of nine is fully explained by these boundaries.

### Projection accumulation across the block

| Purpose | Executed instructions |
|---|---|
| MAC multiplications | 6,291,456 |
| MAC accumulation additions | 6,291,456 |
| Input zero-point subtractions | 6,291,456 |
| Inner-loop input and weight loads | 12,582,912 |
| Inner-loop pointer updates | 12,582,912 |
| Inner-loop branches | 6,291,456 |
| **All six projection inner loops** | **50,331,648** |
| Remaining projection work: setup, bias, stores, outer loops | 419,664 |
| **Complete projection accumulation kernels** | **50,751,312** |

The six projection calls perform **6,291,456 mathematical MACs** and execute **50,751,312 instructions**, fixed across these fixtures. They account for **88.75%** of the mean full-function body count (57,181,466.5). This percentage uses corrected function bodies; the earlier stage-distribution table uses raw measurements.

The complete scalar projection loop is the dominant instruction-count region. Multiplication plus accumulation is 25% of that loop's instructions; loads, centering, pointer updates, and branching make up 75%. This does not prove a hardware memory bottleneck or predict an accelerator cycle speedup.

### Findings by stage

| Stage | Main finding |
|---|---|
| LN1 and LN2 | About 0.45% each. Each executes 28,672 multiplications and 32 unsigned divisions; inverse-square-root branches cause most count variation. |
| QKV | Three 128 → 128 projections perform 1,572,864 MACs. Accumulation is 98.95% of this stage; ordinary conversion is about 1.05%. |
| Attention | 262,144 MACs: 131,072 scores and 131,072 contexts. Context uses INT32 probabilities × INT8 values. Softmax executes 8,448 divisions. |
| Out projection + residual | 524,288 projection MACs. Accumulation is 97.18% of this stage; bias, conversions, and T0 alignment remain explicit. |
| MLP expansion + GELU | 2,097,152 projection MACs. Expansion accumulation alone is 86.62% of this stage. GELU has separate elementwise and conversion passes. |
| MLP output + residual | 2,097,152 projection MACs. Accumulation is 99.32% of this stage. T73 must stay live through the final residual. |

Across all projection and attention matrix products, the block performs **6,553,600 mathematical MACs**. GELU and LN contain additional multiplication instructions that are not matrix MACs. The required two-stage GELU Op 70 rounding remains part of the baseline.

### Evidence and limits

All six new stage analyses follow the QKV structure and contain the actual C, annotated assembly, loop formulas, complete opcode counts with purposes, and reconciliation for all eight cases. The local semantic replay reproduces **32,768 final outputs with zero mismatches**, all 56 supplied stage counts, and all eight supplied full counts after their exact caller overheads. A separate host build of the uploaded snapshot also passes both original output-check paths.

These are **instruction-count findings**. No new Spike run, CV-Wally cycle speedup, or accelerator performance result is claimed. Functional validation covers the supplied integer fixtures, not unseen-data quality or exact equivalence to the original mixed TFLite graph.

See [counting method and validation](analysis_method.md) for boundaries, provenance, and reproducibility. The `evidence/` folder contains the machine-readable counts.
