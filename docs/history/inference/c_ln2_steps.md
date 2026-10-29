# Part 5: LN2 for blocks 0–7

Prerequisites: Parts 1–4 have passed on the combined custom Spike for all eight blocks. Part 4 reported 104 cases per path, 425984 residual INT32 outputs per path, zero mismatches, and checked checkpoints, buffer reuse and guards.

This package adds LN2 using the existing INT32-input LayerNorm arithmetic and XREQUANT backend. It preserves the validated Python conversion points and per-block constants.

## Changes to review before extraction

| Path, relative to this experiment | Change and purpose |
|---|---|
| `src/blocks/block0.c/.h` through `block7.c/.h` | Add `blockN_ln2_forward()`; existing Part 1–4 function bodies remain unchanged |
| `python/export_ln2_fixtures.py` | Export complete LN2 integer checkpoints for standalone and chained tests |
| `tests/ln2/test_allblocks_ln2.c`, `Makefile` | Check LN2 alone and after the C Parts-1-through-4 chain |

The archive contains complete updated versions of the 16 block files. `docs/c_ln2_changes.patch` shows the exact additions relative to Part 4; it is for review and does not need to be applied after extraction.

No shared-kernel, parameter, instruction-wrapper or earlier-Makefile changes are needed. The existing LayerNorm source already contains the required INT32-input functions. No Spike rebuild is required.

`c_ln2_package.sha256` checks this delivery snapshot. Earlier checksum lists retain their original snapshot values and differ for block files extended in later stages. All code remains under `experiments/integer_inference_allblocks/`.

## LN2 formats and conversions

All blocks receive the Part-4 INT32 residual at scale 2^-16 and zero point 0. This makes the LN2 internal scales common to every block. Each block retains its own gamma, beta, final multiplier and output zero point.

| Stage | Stored format | Scale / conversion |
|---|---|---|
| Residual input | INT32 `[32,128]` | 2^-16 |
| Feature-sum mean | INT32 `[32]` | 2^-23 when interpreted as a mean |
| Fine squared difference | INT64 `[32,128]` | 2^-46 |
| Stored squared difference | INT32 `[32,128]` | 2^-23; XREQUANT mode 0, shift 23 |
| Variance plus epsilon | INT32 `[32]` | 2^-30; integer epsilon 10737 |
| Inverse standard deviation | INT32 `[32]` | 2^-16; integer square root of `2^62 / variance_plus_epsilon` |
| Fine normalized values | INT64 `[32,128]` | 2^-39 |
| Stored normalized values | INT32 `[32,128]` | 2^-16; XREQUANT mode 0, shift 23 |
| Gamma/beta affine values | INT64 `[32,128]` | 2^-32 |
| LN2 output | INT8 `[32,128]` | Saved per-block quantization; XREQUANT mode 1, saved multiplier and shift 47 |

The arithmetic kernels and requantization calls remain separate. Reciprocal square root uses floor integer division followed by floor integer square root. The existing rounding and saturation rules are unchanged.

The API is:

```c
void blockN_ln2_forward(
    const int32_t *input,
    int8_t *output,
    LayerNormI8Workspace *workspace
);
```

Input/output each contain 4096 values. Row workspace arrays contain 32 INT32 values; feature workspaces contain 4096 INT32 and 4096 INT64 values. The existing workspace type is reused.

The INT32 input must remain separate from all workspace buffers: the pipeline reads the input again during normalization. Input, output and workspace buffers must be valid and mutually disjoint. Intermediates must satisfy the integer bounds established in Python. The function performs no allocation and preserves its input.

## Test coverage

Each block has 13 cases per path:

- **Standalone:** eight actual Python Part-4 residual outputs plus five direct INT32 LN2 patterns. At scale 2^-16, these are zero, constant -1, constant +1, alternating -1/+1, and one +1 feature among zeros. The constant inputs directly exercise zero variance and epsilon handling.
- **Parts1-4-chained:** eight actual saved block inputs plus five original INT8 block-input edge patterns pass through the C LN1, Q/K/V, attention and first residual functions. The resulting INT32 residual is checked before entering LN2.

The first eight standalone and chained inputs match. The five additional patterns exercise different stages, so each path has its own expected LN2 checkpoints.

The test compares every feature sum, fine/stored square, variance before/after epsilon, inverse standard deviation, fine/stored normalized value, affine value and final INT8 output. It checks both the actual block LN2 function and the individual stages, along with input preservation and guards around all writable arrays.

Totals: **104 cases per path and 425984 LN2 INT8 outputs per path**. Intentional edge inputs can saturate the LN2 INT8 output; the test requires exact agreement with Python's saturated values. Saturation totals are recorded in the fixture manifest.

## Step 1: install the Part-5 files

From the project root, inspect the archive. Adjust the download location if needed.

```bash
cd "$HOME/Documents/ACompArc/model/project"
```

```bash
tar -tzf "$HOME/Downloads/tinystories-allblocks-ln2-c.tar.gz"
```

After reviewing the changes above, extract:

```bash
tar -xzf "$HOME/Downloads/tinystories-allblocks-ln2-c.tar.gz"
```

Optional delivery checksum check:

```bash
cd experiments/integer_inference_allblocks
```

```bash
sha256sum -c c_ln2_package.sha256
```

```bash
cd "$HOME/Documents/ACompArc/model/project"
```

## Step 2: generate LN2 checkpoints

Use the same virtual environment and existing parameter archive. This command needs NumPy and the earlier Python package and fixture-export helpers. It does not load the FP32 model or execute LiteRT.

```bash
python experiments/integer_inference_allblocks/python/export_ln2_fixtures.py \
    --params experiments/integer_inference_allblocks/build/parameter_export \
    --out-dir experiments/integer_inference_allblocks/build/fixtures/ln2
```

```bash
printf 'LN2 fixture export exit status: %s\n' "$?"
```

Expected final message:

```text
PASS: 8 blocks, 104 cases per path, 425984 LN2 INT8 outputs per path; C parameters verified
```

The exporter checks the profile, reference-vector checksum and all 17 C parameter files, then writes complete per-block fixtures atomically. Existing outputs require `--overwrite` or a new output directory.

Generated files under the ignored build directory are `ln2_golden.h`, `ln2_case_names.c`, eight block fixture `.c` files and `manifest.json`. Full INT64 checkpoints and separate edge cases for each path make these relatively large test files. They are separate from inference parameters.

## Step 3: compile LN2

```bash
make -C experiments/integer_inference_allblocks/tests/ln2 \
    CVW_ROOT="$HOME/Documents/ACompArc/cvw" \
    riscv
```

```bash
printf 'LN2 build exit status: %s\n' "$?"
```

The configuration remains `rv64imc_zicsr`, ABI lp64, code model medany, using the existing CV-Wally runtime. Expected ELF:

```text
experiments/integer_inference_allblocks/build/tests/ln2/test_allblocks_ln2.elf
```

## Step 4: run LN2 on custom Spike

After a successful build:

```bash
make -C experiments/integer_inference_allblocks/tests/ln2 \
    CVW_ROOT="$HOME/Documents/ACompArc/cvw" \
    SPIKE="$HOME/Documents/ACompArc/spike-xtile-install/bin/spike" \
    run
```

```bash
printf 'All-block LN2 Spike exit status: %s\n' "$?"
```

Expected output:

```text
Block 0: 13 standalone and Parts1-4-chained LN2 cases PASS
Block 1: 13 standalone and Parts1-4-chained LN2 cases PASS
Block 2: 13 standalone and Parts1-4-chained LN2 cases PASS
Block 3: 13 standalone and Parts1-4-chained LN2 cases PASS
Block 4: 13 standalone and Parts1-4-chained LN2 cases PASS
Block 5: 13 standalone and Parts1-4-chained LN2 cases PASS
Block 6: 13 standalone and Parts1-4-chained LN2 cases PASS
Block 7: 13 standalone and Parts1-4-chained LN2 cases PASS
PASS: 8 blocks, 104 cases per path, 425984 LN2 INT8 outputs per path, 0 mismatches; checkpoints and guards checked
```

Require status 0. A mismatch reports the block, case, path, checkpoint, index, actual and expected values. The chained path also checks the complete C Part-4 residual against Python before LN2.

## Verification status

On local copies, the actual INT32-input LayerNorm kernels and new block LN2 functions passed all standalone and chained comparisons, all checkpoints, input-preservation checks and guards. Compilation used warnings as errors and undefined-behavior sanitization. The direct zero-variance and ±1 edge patterns also passed.

Every existing Part 1–4 block function is unchanged. The new LN2 Makefile and existing Part-4 Makefile passed dependency/build-command dry runs with placeholder runtime files. No earlier Makefiles needed modification.

Local execution used the real LayerNorm, attention and residual software kernels with separate temporary XTILE/XREQUANT emulators outside the delivered package. Actual RV64 compilation and execution on your combined custom Spike remain laptop checks.

After LN2 passes, continue with Part 6: MLP expansion through XTILE, followed by the existing integer GELU stages and their requantization rules.
