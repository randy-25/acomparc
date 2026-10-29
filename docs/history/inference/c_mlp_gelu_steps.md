# Part 6: MLP expansion and GELU for blocks 0–7

Prerequisites: Parts 1–5 have passed on the combined custom Spike for all eight blocks. LN2 reported 104 cases per path, 425984 outputs per path, zero mismatches and checked checkpoints and guards.

This package ports the existing staged block-0 MLP/GELU calculation to each block's exported parameters. It retains the saved expansion INT8 boundary, all GELU conversion points, the per-block tanh table and the final two-round conversion.

## Changes to review before extraction

| Path, relative to this experiment | Change and purpose |
|---|---|
| `src/gelu.c`, `src/gelu.h` | Exact copies of the uploaded working GELU arithmetic kernels |
| `src/mlp_gelu.h` | Common `MlpGeluWorkspace` layout for all eight blocks |
| `src/blocks/block0.c/.h` through `block7.c/.h` | Add expansion-only, GELU-only and complete Part-6 entry points; existing Part 1–5 function bodies remain unchanged |
| Earlier five test Makefiles | Link `gelu.c`, because the shared block source files now also define Part 6 |
| `python/export_mlp_gelu_fixtures.py` | Export expansion/MLP fixtures plus exhaustive GELU-code checkpoints |
| `tests/mlp_gelu/test_allblocks_mlp_gelu.c`, `Makefile` | Verify full Part 6, the upstream C chain and all 256 GELU INT8 input codes |

The archive contains complete updated versions of the 16 block files and five earlier Makefiles. `docs/c_mlp_gelu_changes.patch` shows the exact changes relative to Part 5; it is for review and does not need to be applied after extraction.

`c_mlp_gelu_package.sha256` checks this delivery snapshot. Earlier checksum lists retain their original snapshot values and differ for subsequently extended files.

No model-parameter, ISA or instruction-wrapper changes are made. No Spike rebuild is needed. The existing scalar and XTILE block-0 experiment directories remain intact.

## Arithmetic and custom instructions

Input is the block's saved LN2 INT8 `[32,128]` output. Expansion produces `[32,512]`, or 16384 elements, and GELU retains this shape.

| Stage | Arithmetic | Conversion |
|---|---|---|
| Dense expansion | XTILE INT8 accumulation, then saved INT32 bias added once in software | XREQUANT mode 2 produces the saved INT8 expansion boundary |
| Half-input branch | Centered INT8 times the saved centered scalar -> INT32 | Mode 2 -> INT8 |
| Cubic branch | Signed cube of the centered expansion code -> INT32 | Mode 2 -> INT8 |
| Weighted cubic branch | Centered cubic code times its saved scalar -> INT32 | Mode 2 -> INT8 |
| Branch sum | Two scaled centered INT8 branches combined in INT64 | Mode 1 with multiplier 1; one rounded conversion after the sum |
| Tanh argument | Saved scalar multiplication -> INT32 | Mode 2 -> INT8 |
| Tanh and one-plus-tanh | Existing 256-entry tanh table -> INT32 at 2^-24; add integer `1<<24` | Mode 2 -> saved INT8 factor |
| Final product | Centered half-branch code times centered factor code -> INT32 | Mode 4 -> final INT8 GELU output |

The final conversion keeps its first Q31 rounding toward positive infinity on ties and its second rounded right shift away from zero on ties. It adds the saved output zero point and clamps to INT8. The branch sum retains a single rounding after addition. All coefficients and tables come from the validated exported profile.

The tanh table and integer GELU arithmetic are the existing implementation. This step does not change the GELU approximation policy.

## Entry points and workspace

Each block now exposes:

```c
void blockN_mlp_expand_forward(
    const int8_t *input, int8_t *output, int32_t *accumulator
);

void blockN_gelu_forward(
    const int8_t *input, int8_t *output, MlpGeluWorkspace *workspace
);

void blockN_mlp_gelu_forward(
    const int8_t *input, int8_t *output, MlpGeluWorkspace *workspace
);
```

The expansion-only and complete Part-6 functions read 4096 INT8 LN2 values. Expansion output, accumulator and GELU output each contain 16384 elements of their declared types. GELU-only reads 16384 INT8 expansion codes. The combined function calls expansion followed by GELU; computation and conversions remain separate operations.

`MlpGeluWorkspace` has the same backing-buffer roles used in the working block-0 implementation:

| Field | Capacity | Role |
|---|---:|---|
| `t89` | 16384 INT8 | Retained expansion output |
| `half` | 16384 INT8 | Retained half-input branch |
| `branch` | 16384 INT8 | Reused through cube, weighted cube, sum, argument and factor |
| `product_i32` | 16384 INT32 | Expansion accumulators, scalar products, lookup values and final products |
| `wide_i64` | 4096 INT64 | Chunked branch sum before its rounded conversion |

The caller supplies all buffers. Complete Part-6 input, output and workspace buffers must be mutually disjoint. GELU-only permits its input to equal `workspace->t89`, which remains read-only throughout GELU. No allocation occurs inside these functions.

The test reuses the 4096-element wide allocation for LayerNorm's fine workspace in earlier phases. Those phases finish before GELU starts; buffers within each active workspace remain disjoint.

## Tests

The MLP test has two paths with 13 cases per block:

- **Standalone:** eight actual Python LN2 outputs plus five directly supplied INT8 expansion-input patterns (centered zero, minimum, maximum, alternating extrema and a single feature outlier).
- **Parts1-5-chained:** eight saved block inputs plus five original block-input edge patterns pass through C LN1, Q/K/V, attention, residual and LN2. The resulting LN2 output is checked before expansion.

The first eight inputs match across paths; the extra five exercise different stages. Each path checks the biased INT32 expansion accumulator, the saved INT8 expansion, the full Part-6 function, separate expansion/GELU calls, input preservation and guards.

Totals: **104 cases per path and 1703936 final MLP/GELU INT8 outputs per path**.

Additionally, the GELU-only test covers **every INT8 input code -128 through 127 for each block**, repeated across a full 16384-element buffer to exercise every branch-sum chunk. It compares all raw and quantized stages against integer Python: half, cube, weighted cube, combined INT64 sum, quantized sum, argument, tanh, one-plus-tanh, factor, final product and final output. The periodic reference stores only 256 elements per checkpoint.

These tests require exact agreement with the validated integer Python implementation. Intentional edge inputs can saturate intermediate INT8 boundaries; their saturated codes are part of the expected result.

## Step 1: install the Part-6 files

From the project root, inspect the archive. Adjust the download location if needed.

```bash
cd "$HOME/Documents/ACompArc/model/project"
```

```bash
tar -tzf "$HOME/Downloads/tinystories-allblocks-mlp-gelu-c.tar.gz"
```

After reviewing the changes above, extract:

```bash
tar -xzf "$HOME/Downloads/tinystories-allblocks-mlp-gelu-c.tar.gz"
```

Optional checksum check:

```bash
cd experiments/integer_inference_allblocks
```

```bash
sha256sum -c c_mlp_gelu_package.sha256
```

```bash
cd "$HOME/Documents/ACompArc/model/project"
```

## Step 2: export the Part-6 checkpoints

Use the same virtual environment and existing parameter archive. NumPy and the earlier Python package/fixture helpers are sufficient; LiteRT and FP32-model execution are not needed.

```bash
python experiments/integer_inference_allblocks/python/export_mlp_gelu_fixtures.py \
    --params experiments/integer_inference_allblocks/build/parameter_export \
    --out-dir experiments/integer_inference_allblocks/build/fixtures/mlp_gelu
```

```bash
printf 'MLP/GELU fixture export exit status: %s\n' "$?"
```

Expected final message:

```text
PASS: 8 blocks, 104 cases per path, 1703936 MLP/GELU INT8 outputs per path; 256 GELU codes per block; C parameters verified
```

The exporter checks the profile, reference-vector checksum and all 17 C parameter files before generating complete per-block files atomically. Existing outputs require `--overwrite` or a new directory.

Generated outputs under the ignored build directory are `mlp_gelu_golden.h`, `mlp_gelu_case_names.c`, eight block fixture `.c` files and `manifest.json`. Each block fixture contains full-path expansion/GELU data and the small exhaustive GELU checkpoint table. They are test data, separate from the runtime parameters.

## Step 3: compile the MLP/GELU test

```bash
make -C experiments/integer_inference_allblocks/tests/mlp_gelu \
    CVW_ROOT="$HOME/Documents/ACompArc/cvw" \
    riscv
```

```bash
printf 'MLP/GELU build exit status: %s\n' "$?"
```

The configuration remains `rv64imc_zicsr`, ABI lp64, code model medany, using the existing CV-Wally runtime. Expected ELF:

```text
experiments/integer_inference_allblocks/build/tests/mlp_gelu/test_allblocks_mlp_gelu.elf
```

## Step 4: run on the combined custom Spike

After a successful build:

```bash
make -C experiments/integer_inference_allblocks/tests/mlp_gelu \
    CVW_ROOT="$HOME/Documents/ACompArc/cvw" \
    SPIKE="$HOME/Documents/ACompArc/spike-xtile-install/bin/spike" \
    run
```

```bash
printf 'All-block MLP/GELU Spike exit status: %s\n' "$?"
```

For each block, expect:

```text
Block 0: all 256 GELU input codes and checkpoints PASS
Block 0: 13 standalone and Parts1-5-chained MLP/GELU cases PASS
```

The same two messages should appear for blocks 1 through 7. Expected final line:

```text
PASS: 8 blocks, 104 cases per path, 1703936 MLP/GELU INT8 outputs per path, 0 mismatches; all 256 GELU codes per block and guards checked
```

Require status 0. A failure identifies the block, path/case or exhaustive code test, checkpoint, index and expected/actual value. Earlier Makefiles were updated to remain buildable; this test's chained path also exercises Parts 1–5 and verifies their resulting LN2 output.

## Verification status

On local copies, all eight blocks passed the full standalone/chained MLP tests, biased accumulator checks, expansion boundaries, exhaustive 256-code GELU checkpoints, input-preservation checks and guards. Compilation used warnings as errors and undefined-behavior sanitization.

The GELU arithmetic kernels match the uploaded originals byte for byte. The GELU orchestration was extracted from the uploaded working block-0 implementation, with explicit input and the shared workspace type. Earlier Part 1–5 block function bodies are unchanged. All six Makefiles passed dependency/build-command dry runs with placeholder runtime files.

Local execution used the real software kernels with separate temporary XTILE/XREQUANT emulators outside the package. Actual RV64 compilation and custom-Spike execution remain laptop checks.

After Part 6 passes, continue with Part 7: 512-to-128 output projection, the retained residual addition, and the final INT8 block-boundary conversion.
