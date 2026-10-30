# Scalar Matrix Multiplication Cycle Analysis

## Experiment locations

Experiments are stored under `experiments/cycle_analysis/`:

| Folder | Computation |
|---|---|
| `multiply/` | One centered product: `(a - zero_point) * b` |
| `qproj/` | Q projection: 32×128 input multiplied by 128×128 weights |
| `mlp_expand/` | MLP expansion: 32×128 input multiplied by 128×512 weights |
| `mlp_out/` | MLP output projection: 32×512 input multiplied by 512×128 weights |

Each folder contains the benchmark C source and a Makefile. Generated data and binaries are stored under `build/cycle_analysis/<experiment>/`.

Measurements use CV-Wally's `rv64gc` configuration with Verilator, running executables compiled for `rv64imc_zicsr` at `-O2`. Wally revision: `ff84ec10824bcbb904838efac415975eed717e66`.

## Computation and measured results

The matrix kernels compute INT32 accumulators from signed INT8 activations and weights:

```text
acc[t, o] = sum over k of (input[t, k] - input_zero_point) * weights[o, k]
```

The measured interval includes the kernel's memory accesses, zero-point subtraction, multiplication, accumulation, loop control, and output stores. Bias addition, requantization, and correctness checking are outside the measured computation. No explicit warm-up call precedes the matrix measurements.

| Kernel | MAC operations | Raw measured cycles | Cycles/MAC | Golden-reference check |
|---|---:|---:|---:|---|
| Q projection | 524,288 | 4,254,689 | 8.12 | 4,096 INT32 outputs; zero mismatches |
| MLP expansion | 2,097,152 | 17,272,290 | 8.24 | 16,384 INT32 outputs; zero mismatches |
| MLP output | 2,097,152 | 17,147,067 | 8.18 | 4,096 INT32 outputs; zero mismatches |

All three matrix kernels compile to eight instructions per MAC in their inner loops.

The earlier `multiply/` experiment returned the expected result, `(12 - 2) * (-7) = -70`. Its raw measurements were 31 cycles before adding a warm-up call and 9 cycles after adding one. These include call and measurement effects and do not measure the isolated multiply-instruction latency.

## Estimated cycles across eight blocks

For batch 1 and sequence length 32, each block contains four 128→128 projections (Q, K, V, and attention output), one 128→512 MLP expansion, and one 512→128 MLP output projection.

Assume the Q measurement represents each 128→128 projection and reuse the measured costs for every block:

```text
Dense MACs per block = 4 × 524,288 + 2 × 2,097,152
                    = 6,291,456

Dense MACs across 8 blocks = 50,331,648

Estimated cycles per block = 4 × 4,254,689 + 17,272,290 + 17,147,067
                           = 51,438,113

Estimated cycles across 8 blocks = 8 × 51,438,113
                                = 411,504,904 cycles
```

**Approximately 411.5 million cycles for dense projection accumulation across all eight blocks.** This extrapolates separate kernel measurements; cache state and block-specific code can change actual costs.

This is a partial model estimate. Complete model cycles also include attention matrix products, normalization, softmax, GELU, bias addition, requantization, residual operations, embedding, and the final output head.
