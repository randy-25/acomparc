# Part 3: software attention for blocks 0–7

Prerequisites: all-block LN1 and Q/K/V have passed on the combined custom Spike. The Q/K/V test reported 104 cases per path, 1277952 outputs per path, zero mismatches, and checked accumulators and guards.

This package extends that implementation through the complete attention stage. It reuses the existing block-0 software kernels. It does not introduce attention packing, new attention instructions, or a change to the Python arithmetic profile.

## Changes to review before extraction

| Path, relative to this experiment | Change and purpose |
|---|---|
| `src/attention.c`, `src/attention.h` | Exact copies of the uploaded working block-0 software attention kernels |
| `src/blocks/block0.c/.h` through `block7.c/.h` | Add score and complete-attention functions for each block; LN1 and Q/K/V function bodies remain unchanged |
| `tests/ln1/Makefile`, `tests/qkv/Makefile` | Link `attention.c`, because the shared block source files now also define attention functions |
| `python/export_attention_fixtures.py` | Generate integer Python attention checkpoints for both test paths |
| `tests/attention/test_allblocks_attention.c`, `Makefile` | Verify scores through final INT8 context output, including the upstream C chain |

The archive contains complete updated versions of the 16 block files and two existing Makefiles. `docs/c_attention_changes.patch` shows the additions relative to the supplied Q/K/V package. It is for review; do not apply it again after extracting the updated files.

`c_attention_package.sha256` checks this delivery snapshot. Earlier package checksum lists retain their original snapshot values and will differ for files extended by later stages.

The existing scalar and XTILE block-0 experiment directories, generated parameters and instruction wrappers are unchanged. No Spike rebuild is needed for this stage.

## Arithmetic and formats

Every block uses 32 tokens, 16 heads and 8 features per head. Q/K/V remain in token-major `[32,128]` INT8 layout. The software kernels index heads directly; no transpose or packing buffer is added.

| Stage | Stored format | Scale / rule |
|---|---|---|
| Q × K^T | INT32 `[16,32,32]` | Q scale × K scale, zero point 0 |
| Causal maximum subtraction | INT32, same shape | Only keys `j <= query_token` participate; masked placeholders are ignored downstream |
| Interpolated exponential LUT | INT32, same shape | Scale 2^-30; masked exponentials are zero |
| Normalized probabilities | INT32, same shape | Scale 2^-16; integer code for 1 is 65536 |
| Probability × V | INT32 `[32,128]` | Scale 2^-16 × V scale; output directly in merged token-major order |
| Final context conversion | INT8 `[32,128]` | Saved block-specific scale and zero point, through XREQUANT mode 2 |

The existing score definition is retained, with no additional `1/sqrt(8)` factor. Softmax retains the 4097-entry exponential LUT, coordinate conversion, interpolation, cutoff and rounded normalization from the validated notebook. Coordinate multipliers remain INT64; block 1 uses 5266217801, which exceeds signed INT32.

Score, softmax and context arithmetic execute in software. The final context INT32-to-INT8 conversion uses the existing XREQUANT backend. In the chained path, LN1 uses XREQUANT modes 0 and 1, and Q/K/V uses XTILE plus XREQUANT mode 2 as already tested.

Each block exposes:

- `blockN_attention_scores_forward(query, key, scores)` for the raw score matrix.
- `blockN_attention_forward(query, key, value, output, score_workspace, context_workspace)` for the full stage.

The score workspace has 16384 INT32 elements and is reused in place for shifted scores, exponentials and probabilities. The context workspace has 4096 INT32 elements. Inputs, output and workspace buffers must be disjoint. The functions perform no allocation.

## Test paths and coverage

Each block has 13 cases: the eight actual saved-chain inputs followed by five block-input edge patterns (zero, all minimum, all maximum, alternating extrema and one feature outlier).

The standalone path receives the Python-computed Q/K/V for those inputs. The chained path recomputes C LN1 and C Q/K/V and checks their outputs before entering attention. Both paths then compare with the same integer Python attention checkpoints. This avoids storing duplicate checkpoint matrices for identical expected attention inputs.

The test checks:

- Every raw score, shifted score, exponential and probability code.
- Every merged INT32 context accumulator and final INT8 context output.
- The actual block attention function and its individual stages.
- Probability bounds, exact zeros at masked positions, first-query probability 65536, and the existing rounded row-sum bound of ±16 steps.
- Input preservation and guards around all writable buffers.

Totals: **104 cases per path and 425984 final INT8 outputs per path**. Each of the four score-shaped checkpoints contains 1703936 INT32 values per path. The comparison target is the validated integer Python reference; this does not claim bit-exact agreement with floating-point TFLite.

## Step 1: install the Part 3 files

From the project root, inspect the archive contents. Adjust the download location if needed.

```bash
cd "$HOME/Documents/ACompArc/model/project"
```

```bash
tar -tzf "$HOME/Downloads/tinystories-allblocks-attention-c.tar.gz"
```

After reviewing the changes above, extract:

```bash
tar -xzf "$HOME/Downloads/tinystories-allblocks-attention-c.tar.gz"
```

Optional checksum check:

```bash
cd experiments/integer_inference_allblocks
```

```bash
sha256sum -c c_attention_package.sha256
```

```bash
cd "$HOME/Documents/ACompArc/model/project"
```

## Step 2: generate the attention checkpoints

Use the same virtual environment and existing parameter archive. This command needs NumPy and the preceding Python package, including `export_ln1_fixtures.py` and `export_qkv_fixtures.py`. It does not execute LiteRT or load the FP32 model.

```bash
python experiments/integer_inference_allblocks/python/export_attention_fixtures.py \
    --params experiments/integer_inference_allblocks/build/parameter_export \
    --out-dir experiments/integer_inference_allblocks/build/fixtures/attention
```

```bash
printf 'Attention fixture export exit status: %s\n' "$?"
```

Expected final message:

```text
PASS: 8 blocks, 104 cases per path, 425984 attention INT8 outputs per path; C parameters verified
```

The exporter verifies profile/checksum consistency and all 17 generated C parameter files before exporting checkpoints. It writes each large per-block fixture through a temporary file and atomically replaces the finished output, then checks its completeness. Existing fixtures require `--overwrite` or a new output directory.

Outputs are `attention_golden.h`, `attention_case_names.c`, eight block fixture `.c` files and `manifest.json`, under the ignored build directory. The full score checkpoints make these fixtures larger than Q/K/V fixtures. They are test data and are not part of a final inference executable.

## Step 3: check the earlier C stages after the linkage update

Rebuild and rerun LN1:

```bash
make -C experiments/integer_inference_allblocks/tests/ln1 \
    CVW_ROOT="$HOME/Documents/ACompArc/cvw" \
    SPIKE="$HOME/Documents/ACompArc/spike-xtile-install/bin/spike" \
    run
```

```bash
printf 'LN1 regression exit status: %s\n' "$?"
```

Then rebuild and rerun Q/K/V:

```bash
make -C experiments/integer_inference_allblocks/tests/qkv \
    CVW_ROOT="$HOME/Documents/ACompArc/cvw" \
    SPIKE="$HOME/Documents/ACompArc/spike-xtile-install/bin/spike" \
    run
```

```bash
printf 'QKV regression exit status: %s\n' "$?"
```

Both should retain their previous passing results and return status 0. The changed files and added attention source are rebuilt automatically.

## Step 4: compile the attention test

```bash
make -C experiments/integer_inference_allblocks/tests/attention \
    CVW_ROOT="$HOME/Documents/ACompArc/cvw" \
    riscv
```

```bash
printf 'Attention build exit status: %s\n' "$?"
```

The configuration remains `rv64imc_zicsr`, ABI lp64, code model medany, with the existing CV-Wally bare-metal runtime. Expected ELF:

```text
experiments/integer_inference_allblocks/build/tests/attention/test_allblocks_attention.elf
```

## Step 5: run the complete attention test

After a successful build:

```bash
make -C experiments/integer_inference_allblocks/tests/attention \
    CVW_ROOT="$HOME/Documents/ACompArc/cvw" \
    SPIKE="$HOME/Documents/ACompArc/spike-xtile-install/bin/spike" \
    run
```

```bash
printf 'All-block attention Spike exit status: %s\n' "$?"
```

Expected output:

```text
Block 0: 13 standalone and LN1-QKV-chained attention cases PASS
Block 1: 13 standalone and LN1-QKV-chained attention cases PASS
Block 2: 13 standalone and LN1-QKV-chained attention cases PASS
Block 3: 13 standalone and LN1-QKV-chained attention cases PASS
Block 4: 13 standalone and LN1-QKV-chained attention cases PASS
Block 5: 13 standalone and LN1-QKV-chained attention cases PASS
Block 6: 13 standalone and LN1-QKV-chained attention cases PASS
Block 7: 13 standalone and LN1-QKV-chained attention cases PASS
PASS: 8 blocks, 104 cases per path, 425984 attention INT8 outputs per path, 0 mismatches; checkpoints, probabilities and guards checked
```

Require status 0. A mismatch identifies the block, case, path, checkpoint and index. This stage executes the software attention loops, so its Spike run can take longer than the dense projection test.

## Verification status

On local copies, the actual software attention kernels and new block entry points passed all checkpoint comparisons, probability contracts, input-preservation checks and guards against Python. Compilation used warnings as errors and undefined-behavior sanitization. The updated block sources also retained the passing local LN1 and Q/K/V regression results.

The attention kernels match the uploaded originals byte for byte. All earlier LN1 and Q/K/V function bodies remain unchanged. The three Makefiles passed dependency/build-command dry runs using placeholder runtime files.

Local execution used separate temporary emulators for upstream XTILE and XREQUANT operations, outside the delivered package. The real software attention C code was executed. Actual RISC-V builds and custom-Spike execution of the updated package remain laptop checks.

After Part 3 passes on Spike, continue with Part 4: output projection through XTILE, its existing requantization, and the first residual addition.
