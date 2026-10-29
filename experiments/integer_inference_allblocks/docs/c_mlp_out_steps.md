# Part 7: MLP output projection, residual and INT8 block boundary

Prerequisites: Parts 1–6 have passed on the combined custom Spike for all eight blocks. Part 6 reported all 256 GELU input codes per block passing, plus 104 cases and 1703936 MLP/GELU outputs per path with zero mismatches and checked guards.

This package adds the final component functions for every block. It preserves the working block-0 INT32 projection/residual calculation and adds the INT8 boundary conversion defined in the validated all-block profile.

## Changes to review before extraction

| Path, relative to this experiment | Change and purpose |
|---|---|
| `src/blocks/block0.c/.h` through `block7.c/.h` | Add `blockN_mlp_out_residual_forward()` and `blockN_output_boundary_forward()`; existing Part 1–6 function bodies remain unchanged |
| `python/export_mlp_out_residual_fixtures.py` | Export projection/residual checkpoints, final INT8 outputs and rounding/saturation probes |
| `tests/mlp_out_residual/test_allblocks_mlp_out_residual.c`, `Makefile` | Check Part 7 alone, after C Parts 1–6, with output-buffer reuse and boundary probes |

The archive contains complete updated versions of the 16 block files. `docs/c_mlp_out_changes.patch` shows the additions relative to Part 6; it is for review and does not need to be applied after extraction.

The required projection, residual-add and requantization functions are already present. No shared-kernel, parameter, instruction-wrapper or earlier-Makefile changes are needed. No Spike rebuild is required.

`c_mlp_out_package.sha256` checks this snapshot. Earlier checksum lists retain their original values and differ for files extended by later packages. All files remain inside `experiments/integer_inference_allblocks/`.

## Arithmetic and conversion points

Input is 16384 INT8 GELU values in `[32,512]` layout. The retained Part-4 residual has 4096 INT32 values at scale 2^-16. It must remain available through Parts 5–6 until the final addition consumes it.

| Stage | Operation | Result |
|---|---|---|
| Output projection | XTILE 512-to-128 INT8 accumulation, then saved INT32 bias added once in software | INT32 `[32,128]` accumulator |
| Projection conversion | XREQUANT mode 2, saved multiplier/shift/zero point and saturation | Saved INT8 projection boundary |
| Residual-branch conversion | XREQUANT mode 3, centered projection codes and saved branch multiplier/shift | INT32 branch at scale 2^-16 |
| Residual addition | Add that branch to the retained Part-4 INT32 residual | INT32 final sum at scale 2^-16 |
| Block-boundary conversion | XREQUANT mode 2, multiplier 1 and zero point 0 | Final INT8 block output |

The last shift depends on the agreed boundary scale:

| Blocks | Final boundary shift | INT8 output scale |
|---|---:|---|
| 0 and 7 | 9 | 2^-7 |
| 1–6 | 8 | 2^-8 |

Ordinary requantization rounds to nearest with ties toward positive infinity. INT8 conversion then clamps to [-128,127]. The residual addition itself is exact integer addition within the validated INT32 range, with no additional rounding or saturation.

## Entry points and buffer rules

```c
void blockN_mlp_out_residual_forward(
    const int8_t *input,
    const int32_t *residual_input,
    int32_t *output,
    int8_t *projection_workspace,
    int32_t *accumulator_workspace
);

void blockN_output_boundary_forward(
    const int32_t *input,
    int8_t *output
);
```

The first function reads 16384 INT8 values; its other buffers each contain 4096 elements of their declared types. Input, retained residual and both workspaces must be mutually disjoint. INT32 output may equal the retained residual or the accumulator workspace; otherwise keep it separate.

The boundary function reads 4096 INT32 values and writes 4096 INT8 values into a disjoint buffer. It uses the already exported boundary macros. Both functions perform no allocation. Calculation and final encoding remain separate calls.

## Tests

There are 13 cases per block on each path:

- **Standalone:** eight actual Python GELU/retained-residual pairs, followed by five directly supplied edge pairs. The GELU inputs cover centered zero, INT8 endpoints, alternating extrema and one feature outlier. Corresponding direct retained-residual patterns include 0 and ±1 at scale 2^-16, exercising final saturation.
- **Parts1-6-chained:** eight saved original block inputs plus five original INT8 block-input edge patterns pass through all earlier C components. The test checks the GELU output and the retained Part-4 residual before Part 7.

For each path, the test checks the biased INT32 accumulator, INT8 projection, converted INT32 branch, final INT32 sum and final INT8 output. It tests distinct output buffers, output equal to the accumulator, and output equal to the retained residual. Inputs remain unchanged except when the retained residual is explicitly used as the output. All writable arrays have guards.

Totals: **104 cases per path and 425984 final INT8 outputs per path**, plus the corresponding INT32 checkpoints.

Each block also has 32 boundary probes repeated across a full 4096-element buffer. They cover positive/negative rounding ties and neighboring values, saturation thresholds and INT32 endpoints. The expected codes come from the integer Python reference.

The fixture exporter independently checks the first eight cases per block against the frozen whole-block vectors: both the INT32 pre-boundary sums and final INT8 outputs must match. Intentional extra edge cases can clip at the INT8 boundary; their clipped values are expected and counts are recorded in the manifest.

## Step 1: install the Part-7 files

From the project root, inspect the archive. Adjust the download location if needed.

```bash
cd "$HOME/Documents/ACompArc/model/project"
```

```bash
tar -tzf "$HOME/Downloads/tinystories-allblocks-mlp-out-residual-c.tar.gz"
```

After reviewing the changes above, extract:

```bash
tar -xzf "$HOME/Downloads/tinystories-allblocks-mlp-out-residual-c.tar.gz"
```

Optional checksum check:

```bash
cd experiments/integer_inference_allblocks
```

```bash
sha256sum -c c_mlp_out_package.sha256
```

```bash
cd "$HOME/Documents/ACompArc/model/project"
```

## Step 2: export Part-7 checkpoints

Use the existing virtual environment and parameter archive. NumPy and the earlier Python package/fixture helpers are sufficient; no FP32-model or LiteRT execution is required.

```bash
python experiments/integer_inference_allblocks/python/export_mlp_out_residual_fixtures.py \
    --params experiments/integer_inference_allblocks/build/parameter_export \
    --out-dir experiments/integer_inference_allblocks/build/fixtures/mlp_out_residual
```

```bash
printf 'MLP output/residual fixture export exit status: %s\n' "$?"
```

Expected final message:

```text
PASS: 8 blocks, 104 cases per path, 425984 final INT8 outputs per path; 32 boundary probes per block; C parameters verified
```

The exporter verifies profile/checksum consistency and all 17 C parameter files, then creates complete per-block fixtures atomically. Existing outputs require `--overwrite` or a new output directory.

Generated outputs are `mlp_out_residual_golden.h`, `mlp_out_residual_case_names.c`, eight block fixture `.c` files and `manifest.json` under the ignored build directory. Each block file also contains its small boundary-probe table.

## Step 3: compile the Part-7 test

```bash
make -C experiments/integer_inference_allblocks/tests/mlp_out_residual \
    CVW_ROOT="$HOME/Documents/ACompArc/cvw" \
    riscv
```

```bash
printf 'MLP output/residual build exit status: %s\n' "$?"
```

Configuration remains `rv64imc_zicsr`, ABI lp64, code model medany, using the existing CV-Wally runtime. Expected ELF:

```text
experiments/integer_inference_allblocks/build/tests/mlp_out_residual/test_allblocks_mlp_out_residual.elf
```

## Step 4: run Part 7 on custom Spike

After a successful build:

```bash
make -C experiments/integer_inference_allblocks/tests/mlp_out_residual \
    CVW_ROOT="$HOME/Documents/ACompArc/cvw" \
    SPIKE="$HOME/Documents/ACompArc/spike-xtile-install/bin/spike" \
    run
```

```bash
printf 'All-block MLP output/residual Spike exit status: %s\n' "$?"
```

For each block, expect:

```text
Block 0: boundary rounding and saturation PASS
Block 0: 13 standalone and Parts1-6-chained MLP output/residual cases PASS
```

The same messages should appear for blocks 1–7. Expected final line:

```text
PASS: 8 blocks, 104 cases per path, 425984 final INT8 outputs per path, 0 mismatches; INT32 checkpoints, boundary probes, buffer reuse and guards checked
```

Require status 0. A failure identifies the block, case/path or boundary probe, checkpoint, index and actual/expected value.

## Verification status and next step

On local copies, all eight blocks passed standalone and chained Part-7 checks, all intermediate comparisons, both permitted output aliases, boundary probes, input-preservation checks and guards. Compilation used warnings as errors and undefined-behavior sanitization. The exported first eight cases per block match the frozen INT32 and INT8 notebook vectors.

Existing Part 1–6 function bodies are unchanged. The Part-7 and existing Part-6 Makefiles passed dependency/build-command dry runs with placeholder runtime files. Earlier Makefiles require no modifications.

Local execution used the real software kernels with temporary external XTILE/XREQUANT emulators. Those emulators are not included. The RV64 build and actual custom-Spike run remain laptop checks.

After this passes, all seven component paths will have been checked on Spike. The next step is to add and test complete `blockN_forward()` entry points and the actual eight-block C chain, passing each C block's INT8 output directly into the next block. Current chained component tests use the saved reference input for each block; they do not yet run that cross-block C driver.
