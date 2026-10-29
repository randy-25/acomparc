# Part 2: Q/K/V with XTILE and XREQUANT for blocks 0–7

Prerequisite: Part 1 LN1 passed on the modified Spike for all eight blocks: 104 cases, 425984 outputs per path, zero mismatches and unchanged guards. The existing all-block Python parameter export and regression must also be present.

## Changes to review before extraction

This package stays under `experiments/integer_inference_allblocks/`. It extends the block files from the LN1 package and adds the Part 2 sources and tests.

| Path | Change and purpose |
|---|---|
| `src/projection.c`, `src/projection.h`, `src/xtile.h` | Exact copies from the uploaded working block-0 XTILE implementation |
| `src/blocks/block0.c/.h` through `block7.c/.h` | Add `blockN_qkv_forward()` and its declaration; the LN1 function bodies are unchanged |
| `tests/ln1/Makefile` | Link `projection.c`, because the shared block source files now also define Q/K/V functions |
| `python/export_qkv_fixtures.py` | Export standalone and LN1-chained INT32 accumulators and INT8 outputs |
| `tests/qkv/test_allblocks_qkv.c`, `Makefile` | Verify Part 2 arithmetic, block wiring and LN1 chaining on custom Spike |

The archive contains complete updated versions of the 16 block source/header files and the LN1 Makefile. If you edited those files after installing the LN1 package, compare your changes before replacing them. `docs/c_qkv_changes.patch` shows the exact additions relative to the supplied LN1 package; it is for review and does not need to be applied after extraction.

The earlier `c_ln1_package.sha256` describes the original delivery snapshot. Its entries for extended files will differ after this update. `c_qkv_package.sha256` checks the current Part 2 delivery files.

The working scalar and XTILE block-0 experiment directories are untouched. This package does not change the generated parameter files, the arithmetic profile, or the custom ISA.

## C operation and formats

Each block-specific function receives its LN1 INT8 output `[32,128]`, three disjoint INT8 output buffers of 4096 elements, and one disjoint INT32 accumulator workspace of 4096 elements. That accumulator is reused between Q, K and V.

For each projection:

1. `projection_accumulate_i8()` invokes XTILE for `A[32,128] × BT[128,128]`, producing INT32 `[32,128]`.
2. `requantize_i32_to_i8()` invokes XREQUANT mode 2 using that projection's INT64 multiplier, shift 39, output zero point and INT8 saturation.

All three projections use the block's exported LN1 input zero point. Weight zero points are zero. Q/K/V have no bias. Weights retain their existing `[output_feature,input_feature]` layout; no packing or attention change is introduced.

Some exported multipliers exceed signed INT32, for example block 1 V uses 2413993628. The C macros, APIs and test metadata retain INT64 multipliers. The exporter already validated their INT64 product bounds.

LN1 computation and its three conversion points remain as tested. Input, Q/K/V outputs, accumulator and parameter arrays must not overlap. Neither LN1 nor Q/K/V allocates memory internally.

## Test coverage

There are 13 cases per block on each of two paths:

- **Standalone:** directly supplies the expected INT8 projection input. The first eight inputs are the Python LN1 outputs for the saved chain cases. Five additional direct projection inputs cover centered zero, all minimum, all maximum, alternating extrema, and one feature outlier.
- **LN1-chained:** supplies the corresponding block input, calls the C LN1 function, checks its INT8 output, then calls C Q/K/V. The eight saved cases use the actual previous Python block's output as each block's input. The five additional block inputs use the same edge pattern categories before LN1.

The first eight standalone and chained inputs match. The extra five inputs intentionally exercise different stages: directly supplied post-LN1 codes versus codes produced by LN1 from block-input edge patterns.

For every path, the test checks the actual block Q/K/V function's outputs, each separately computed INT32 accumulator, each requantized INT8 output, input preservation, and guards around all writable arrays. The accumulator retains V after the block function and is checked there too.

Totals: **104 cases per path, 1277952 Q/K/V INT8 outputs per path**, with the same number of expected INT32 accumulator values per path. This is a Part 2 test, not a C attention or full eight-block inference test.

## Step 1: install the Part 2 files

From the project root, inspect the archive. Adjust the download location if needed.

```bash
cd "$HOME/Documents/ACompArc/model/project"
```

```bash
tar -tzf "$HOME/Downloads/tinystories-allblocks-qkv-c.tar.gz"
```

After reviewing the changes above, extract the updated files:

```bash
tar -xzf "$HOME/Downloads/tinystories-allblocks-qkv-c.tar.gz"
```

Optional delivery checksum check:

```bash
cd experiments/integer_inference_allblocks
```

```bash
sha256sum -c c_qkv_package.sha256
```

```bash
cd "$HOME/Documents/ACompArc/model/project"
```

## Step 2: export the Q/K/V fixtures

Use the same virtual environment and the already validated parameter archive. The script also imports `export_ln1_fixtures.py` from the previous package. NumPy is sufficient; no LiteRT or FP32-model execution is needed.

```bash
python experiments/integer_inference_allblocks/python/export_qkv_fixtures.py \
    --params experiments/integer_inference_allblocks/build/parameter_export \
    --out-dir experiments/integer_inference_allblocks/build/fixtures/qkv
```

```bash
printf 'QKV fixture export exit status: %s\n' "$?"
```

Expected final line:

```text
PASS: 8 blocks, 104 cases per path, 1277952 Q/K/V outputs per path; C parameters verified
```

The exporter verifies the reference-vector profile and checksum and compares all 17 C parameter files with a fresh temporary rendering of the archive. It then creates `qkv_golden.h`, `qkv_case_names.c`, eight block fixture `.c` files and `manifest.json`. They are generated test data under the ignored build directory. Existing outputs require `--overwrite` or a new output directory.

## Step 3: rebuild and rerun LN1 after extending the block files

This checks the updated linkage and keeps the already passing Part 1 as a regression check. The Makefile rebuilds changed source files automatically.

```bash
make -C experiments/integer_inference_allblocks/tests/ln1 \
    CVW_ROOT="$HOME/Documents/ACompArc/cvw" \
    SPIKE="$HOME/Documents/ACompArc/spike-xtile-install/bin/spike" \
    run
```

```bash
printf 'LN1 regression exit status: %s\n' "$?"
```

Require the same LN1 result as before: 104 cases, 425984 outputs per path, zero mismatches, unchanged guards and status 0.

## Step 4: compile the Q/K/V test

```bash
make -C experiments/integer_inference_allblocks/tests/qkv \
    CVW_ROOT="$HOME/Documents/ACompArc/cvw" \
    riscv
```

```bash
printf 'QKV build exit status: %s\n' "$?"
```

The target is RV64IMC with Zicsr, ABI lp64, code model medany, using the existing CV-Wally bare-metal runtime. Expected ELF:

```text
experiments/integer_inference_allblocks/build/tests/qkv/test_allblocks_qkv.elf
```

## Step 5: run Q/K/V on the combined custom Spike

After a successful build:

```bash
make -C experiments/integer_inference_allblocks/tests/qkv \
    CVW_ROOT="$HOME/Documents/ACompArc/cvw" \
    SPIKE="$HOME/Documents/ACompArc/spike-xtile-install/bin/spike" \
    run
```

```bash
printf 'All-block QKV Spike exit status: %s\n' "$?"
```

Expected output:

```text
Block 0: 13 standalone and LN1-chained QKV cases PASS
Block 1: 13 standalone and LN1-chained QKV cases PASS
Block 2: 13 standalone and LN1-chained QKV cases PASS
Block 3: 13 standalone and LN1-chained QKV cases PASS
Block 4: 13 standalone and LN1-chained QKV cases PASS
Block 5: 13 standalone and LN1-chained QKV cases PASS
Block 6: 13 standalone and LN1-chained QKV cases PASS
Block 7: 13 standalone and LN1-chained QKV cases PASS
PASS: 8 blocks, 104 cases per path, 1277952 Q/K/V outputs per path, 0 mismatches; accumulators and guards checked
```

Require status 0. A mismatch reports the block, case, path, projection/checkpoint, element index, actual value and expected value.

## Verification status

On local copies, the new C orchestration, standalone paths, LN1-chained paths, accumulators and guards passed against integer Python fixtures with warnings treated as errors and undefined-behavior sanitization enabled. The existing LN1 function bodies are unchanged. The reused projection and XTILE wrapper files match the uploaded originals byte for byte.

Local C verification used temporary scalar emulators for the XTILE and XREQUANT operations, outside the package. The package's production backend uses the real wrappers. Both Makefiles passed dependency/build-command dry runs using placeholder runtime files. Actual RISC-V compilation and execution on the custom Spike remain laptop checks.

After both LN1 regression and Q/K/V pass on Spike, proceed to Part 3 using the existing software attention calculation and each block's exported coefficients and tables.
