# Part 4: output projection and first residual for blocks 0–7

Prerequisites: Parts 1–3 have passed on the combined custom Spike for all eight blocks. The Part-3 attention test reported 104 cases per path, 425984 outputs per path, zero mismatches, and checked checkpoints, probabilities and guards.

This package extends the existing implementation through the first residual sum. It preserves the numerical boundaries in the validated Python reference and the custom-instruction roles used by the working block-0 implementation.

## Changes to review before extraction

| Path, relative to this experiment | Change and purpose |
|---|---|
| `src/residual.c`, `src/residual.h` | Exact copies of the uploaded working residual arithmetic kernels |
| `src/blocks/block0.c/.h` through `block7.c/.h` | Add `blockN_out_residual_forward()`; existing Part 1–3 function bodies remain unchanged |
| `tests/ln1/Makefile`, `tests/qkv/Makefile`, `tests/attention/Makefile` | Link `residual.c`, because the shared block files now also define Part 4 |
| `python/export_out_residual_fixtures.py` | Export biased accumulators, INT8 projection values, INT32 branch values and final residual values |
| `tests/out_residual/test_allblocks_out_residual.c`, `Makefile` | Test Part 4 alone, chained from Parts 1–3, and with the permitted output/accumulator reuse |

The archive contains complete updated versions of the 16 block files and three existing Makefiles. The exact additions relative to Part 3 are in `docs/c_out_residual_changes.patch`; this is for review and does not need to be applied after extraction.

`c_out_residual_package.sha256` describes this delivery snapshot. Earlier checksum lists retain their original snapshot values and differ for files subsequently extended.

The working block-0 experiment directories, model parameters, earlier Python files and instruction wrappers remain unchanged. No Spike rebuild is required.

## Operations, formats and conversion points

Each tensor has 4096 elements, arranged as 32 tokens × 128 features.

| Step | Operation | Result |
|---|---|---|
| 1 | XTILE INT8 dense accumulation; add the saved INT32 bias once in software | INT32 accumulator at attention-input scale × weight scale |
| 2 | XREQUANT mode 2, saved multiplier, shift 39, projection zero point and saturation | Saved INT8 output-projection boundary |
| 3 | XREQUANT mode 3: center the projection INT8 codes, then multiply and round with the saved branch coefficient/shift | INT32 branch at scale 2^-16, zero point 0 |
| 4 | Align the original INT8 block input exactly and add it to that branch | INT32 residual output at scale 2^-16, zero point 0 |

The original block input is retained for the residual path. Block 1 uses input scale 2^-7, so its alignment to 2^-16 is 512. The other blocks use 2^-8, so their alignment is 256. The addition applies no further rounding or saturation.

Projection multipliers remain INT64, including those exceeding signed INT32. The bias remains part of the accumulator before requantization. The saved INT8 projection boundary is retained, including its rounding and saturation; it is not combined with the later branch conversion.

Part 4 returns **INT32**, ready for LN2. This is an internal residual representation; the agreed INT8 input/output block boundaries are unchanged.

The API is:

```c
void blockN_out_residual_forward(
    const int8_t *attention_output,
    const int8_t *residual_input,
    int32_t *output,
    int8_t *projection_workspace,
    int32_t *accumulator_workspace
);
```

Each buffer contains 4096 elements of its declared type. `output` may equal `accumulator_workspace`. All other buffers must be disjoint and valid. The function allocates no memory.

## Test coverage

There are 13 cases per block: eight actual saved-chain inputs plus five block-input edge patterns (zero, all minimum, all maximum, alternating extrema and one feature outlier).

The standalone path receives the Python attention output and the original block input. The chained path recomputes C LN1, Q/K/V and attention, checking their outputs before Part 4. Both use the same integer Python Part-4 checkpoints.

For every case and path, the test checks:

- The INT32 accumulator including the saved bias.
- The saved INT8 projection output.
- The converted INT32 residual branch.
- The final INT32 residual sum.
- The block orchestration with distinct output and accumulator buffers.
- The permitted case where final output and accumulator share one buffer.
- Input preservation and guards around all writable arrays.

Totals: **104 cases per path, 425984 final residual INT32 outputs per path**. The same number of projection, accumulator and branch checkpoint values are checked per path. Comparisons are exact against the validated integer Python reference.

## Step 1: install the Part-4 files

From the project root, inspect the archive; adjust the download location if needed.

```bash
cd "$HOME/Documents/ACompArc/model/project"
```

```bash
tar -tzf "$HOME/Downloads/tinystories-allblocks-out-residual-c.tar.gz"
```

After reviewing the changes above, extract:

```bash
tar -xzf "$HOME/Downloads/tinystories-allblocks-out-residual-c.tar.gz"
```

Optional checksum check:

```bash
cd experiments/integer_inference_allblocks
```

```bash
sha256sum -c c_out_residual_package.sha256
```

```bash
cd "$HOME/Documents/ACompArc/model/project"
```

## Step 2: generate the Part-4 checkpoints

Use the same virtual environment and existing parameter archive. This script uses NumPy, the integer Python package and helpers from the earlier fixture exporters. No LiteRT or FP32-model execution is needed.

```bash
python experiments/integer_inference_allblocks/python/export_out_residual_fixtures.py \
    --params experiments/integer_inference_allblocks/build/parameter_export \
    --out-dir experiments/integer_inference_allblocks/build/fixtures/out_residual
```

```bash
printf 'Output/residual fixture export exit status: %s\n' "$?"
```

Expected final line:

```text
PASS: 8 blocks, 104 cases per path, 425984 residual INT32 outputs per path; C parameters verified
```

The exporter checks the profile, reference-vector checksum and all 17 generated C parameter files, then writes complete per-block fixtures atomically. Existing outputs require `--overwrite` or a new directory.

Outputs under the ignored build directory are `out_residual_golden.h`, `out_residual_case_names.c`, eight block fixture `.c` files and `manifest.json`. The manifest records the residual output range, projection saturation count and input alignment for each block. Fixtures are test data, separate from inference parameters.

## Step 3: compile Part 4

```bash
make -C experiments/integer_inference_allblocks/tests/out_residual \
    CVW_ROOT="$HOME/Documents/ACompArc/cvw" \
    riscv
```

```bash
printf 'Output/residual build exit status: %s\n' "$?"
```

The configuration remains `rv64imc_zicsr`, ABI lp64, code model medany, using the existing CV-Wally bare-metal runtime. Expected ELF:

```text
experiments/integer_inference_allblocks/build/tests/out_residual/test_allblocks_out_residual.elf
```

## Step 4: run Part 4 on custom Spike

After a successful build:

```bash
make -C experiments/integer_inference_allblocks/tests/out_residual \
    CVW_ROOT="$HOME/Documents/ACompArc/cvw" \
    SPIKE="$HOME/Documents/ACompArc/spike-xtile-install/bin/spike" \
    run
```

```bash
printf 'All-block output/residual Spike exit status: %s\n' "$?"
```

Expected output:

```text
Block 0: 13 standalone and Parts1-3-chained output/residual cases PASS
Block 1: 13 standalone and Parts1-3-chained output/residual cases PASS
Block 2: 13 standalone and Parts1-3-chained output/residual cases PASS
Block 3: 13 standalone and Parts1-3-chained output/residual cases PASS
Block 4: 13 standalone and Parts1-3-chained output/residual cases PASS
Block 5: 13 standalone and Parts1-3-chained output/residual cases PASS
Block 6: 13 standalone and Parts1-3-chained output/residual cases PASS
Block 7: 13 standalone and Parts1-3-chained output/residual cases PASS
PASS: 8 blocks, 104 cases per path, 425984 residual INT32 outputs per path, 0 mismatches; checkpoints, buffer reuse and guards checked
```

Require status 0. Any mismatch reports the block, case, path, checkpoint, index, actual and expected values. The chained path also verifies the outputs of Parts 1–3, so they are exercised by this test. Their standalone Makefiles were updated to remain buildable if a separate regression run is needed.

## Verification status

On local copies, the Part-4 functions passed all standalone and chained comparisons, bias and requantization checkpoints, distinct-buffer and reused-buffer cases, input preservation, and guards. Compilation used warnings as errors and undefined-behavior sanitization.

The reused residual kernels match the uploaded originals byte for byte, and all Part 1–3 function bodies remain unchanged. Existing LN1, Q/K/V and attention tests also passed locally with the updated sources. All four Makefiles passed dependency/build-command dry runs using placeholder runtime files.

Local execution used the real software LayerNorm, attention and residual kernels, with separate temporary emulators for XTILE and XREQUANT operations. Those emulators are outside the package. Actual RV64 compilation and execution on your combined custom Spike remain laptop checks.

After Part 4 passes, proceed to Part 5: LN2 consumes the INT32 residual at scale 2^-16 and produces the block's saved INT8 LN2 output.
