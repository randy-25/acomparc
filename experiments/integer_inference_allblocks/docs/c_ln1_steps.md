# Part 1: C LN1 for blocks 0–7

This is the first C step after the all-block Python export and extraction regression passed on your laptop. It adds files only under `experiments/integer_inference_allblocks/`. It does not change the working scalar or XTILE block-0 experiments, the previously delivered Python files, or the generated parameters.

## What changes and why

| New path, relative to this experiment | Purpose |
|---|---|
| `src/layernorm.c` | Exact copy of the uploaded working arithmetic kernels |
| `src/layernorm.h` | Workspace and kernel declarations; imports the generated `LayerNormI8Params` definition |
| `src/requantization.c`, `.h`, `src/xrequant.h` | Exact copies of the uploaded custom-instruction backend and wrappers |
| `src/blocks/block0.c/.h` through `block7.c/.h` | Separate block entry points; each currently implements LN1 only |
| `python/export_ln1_fixtures.py` | Generate integer Python checkpoints and check C parameters against the archive |
| `tests/ln1/test_allblocks_ln1.c`, `Makefile` | Compare all blocks' LN1 checkpoints and forward outputs on modified Spike |

Each block has its own `blockN_ln1_forward()` function. The functions share arithmetic kernels and use their own exported `blockN_ln1_params` object. Computation and requantization remain separate calls.

The generated `integer_parameter_types.h` already defines `LayerNormI8Params`. The new `layernorm.h` includes that header instead of defining the type again. The Makefile supplies its include directory, so no copy of the generated type or parameter data is needed under `src/`.

## Arithmetic and instruction use

Every input and output contains 4096 INT8 values, arranged as 32 tokens × 128 features. Input zero point is 0. Output scales and zero points are the saved per-block LN1 values.

| Blocks | Input scale | Square shift | Normalize shift | Integer epsilon | RSQRT numerator |
|---|---|---:|---:|---:|---|
| 0 and 2–7 | 2^-8 | 15 | 15 | 42 | 2^54 |
| 1 | 2^-7 | 14 | 14 | 21 | 2^53 |

For each token, the C pipeline performs:

1. Sum its 128 feature codes and calculate aligned squared differences.
2. Use **XREQUANT mode 0** to store rounded squared differences as INT32.
3. Sum those values, add integer epsilon, and calculate the integer reciprocal square root in software.
4. Multiply centered features by the inverse standard deviation.
5. Use **XREQUANT mode 0** to store normalized INT32 features at scale 2^-16.
6. Apply the INT32 gamma/beta affine transform, producing INT64 values at scale 2^-32.
7. Use **XREQUANT mode 1** to produce the saved LN1 INT8 output.

Each conversion call configures and runs the existing XREQUANT engine. The encoding and rounding rules are unchanged. XTILE enters with Part 2, the Q/K/V dense projections; LN1 has no dense projection.

Workspaces are caller supplied: three 32-element INT32 row buffers, one 4096-element INT32 feature buffer and one 4096-element INT64 feature buffer. Input, output and workspace buffers must be mutually disjoint. There is no allocation inside the forward functions.

## What the test checks

Each block receives eight actual inputs from the validated Python integer chain. For block 0, these are the benchmark plus seven calibration input encodings; for later blocks, they are the previous Python block's actual INT8 outputs. Five additional inputs exercise zero variance, the INT8 endpoints, alternating extrema and a single feature outlier.

The 13 cases per block give **104 LN1 cases** and **425984 INT8 outputs per path**. The test first calls the real block-specific forward function, then checks the individual stages against Python:

- Feature sums.
- INT64 squared differences and stored INT32 squared differences.
- Variance before and after epsilon, and inverse standard deviation.
- INT64 normalized values and stored INT32 normalized values.
- INT64 affine values and final INT8 output.

It also checks the input remains unchanged and verifies guards around all writable arrays. These are LN1 component tests. The C implementations of Parts 2–7 and the C eight-block chain have not been added yet.

## Step 1: install the additive files

From the project root, inspect the archive contents first. Adjust the download location if your browser saved it elsewhere.

```bash
cd "$HOME/Documents/ACompArc/model/project"
```

```bash
tar -tzf "$HOME/Downloads/tinystories-allblocks-ln1-c.tar.gz"
```

All listed paths start with `experiments/integer_inference_allblocks/`. The initial installation adds the new files listed above.

```bash
tar -xzf "$HOME/Downloads/tinystories-allblocks-ln1-c.tar.gz"
```

Optionally verify the packaged files from the experiment directory:

```bash
cd experiments/integer_inference_allblocks
```

```bash
sha256sum -c c_ln1_package.sha256
```

```bash
cd "$HOME/Documents/ACompArc/model/project"
```

## Step 2: export the LN1 checkpoints

Keep the same virtual environment used for the successful parameter export and regression. This command needs only NumPy and the existing Python package; it does not load FP32 models or LiteRT.

```bash
python experiments/integer_inference_allblocks/python/export_ln1_fixtures.py \
    --params experiments/integer_inference_allblocks/build/parameter_export \
    --out-dir experiments/integer_inference_allblocks/build/fixtures/ln1
```

```bash
printf 'LN1 fixture export exit status: %s\n' "$?"
```

Expected final message:

```text
PASS: 8 blocks, 104 LN1 cases, 425984 expected INT8 outputs; C parameters verified
```

The exporter reads `parameters.json/.npz` and the packaged `tests/data/chain_reference.npz`. It verifies their profile and checksums. Before creating fixtures, it renders the C data in a temporary directory and compares all 17 generated C files with `build/parameter_export/c/`. It leaves your C parameter files untouched.

Outputs are `ln1_golden.h`, `ln1_case_names.c`, eight `blockN_ln1_golden.c` files, and `manifest.json`. They live under the ignored build directory. Full INT64 checkpoints make the generated fixture sources relatively large; they are test data, not runtime model state. Existing fixture outputs require `--overwrite` or a new output directory.

## Step 3: select the external dependencies

The CV-Wally checkout and the custom Spike installation remain outside the project repository.

```bash
export CVW_ROOT="$HOME/Documents/ACompArc/cvw"
```

```bash
export SPIKE_XTILE="$HOME/Documents/ACompArc/spike-xtile-install/bin/spike"
```

```bash
test -f "$CVW_ROOT/examples/C/common/crt.S"
```

```bash
printf 'CV-Wally runtime check exit status: %s\n' "$?"
```

```bash
test -x "$SPIKE_XTILE"
```

```bash
printf 'Custom Spike executable check exit status: %s\n' "$?"
```

Both checks should return 0. `SPIKE_XTILE` must name the installed simulator containing both XTILE and XREQUANT. You can use another verified installation path if needed.

## Step 4: compile the bare-metal LN1 test

The new Makefile uses `rv64imc_zicsr`, ABI `lp64`, code model `medany`, and the existing `crt.S`, `syscalls.c`, and `test.ld`. There is no native-host target: the production requantization source uses RV64 custom instructions.

```bash
make -C experiments/integer_inference_allblocks/tests/ln1 \
    CVW_ROOT="$CVW_ROOT" \
    riscv
```

```bash
printf 'LN1 build exit status: %s\n' "$?"
```

Expected ELF:

```text
experiments/integer_inference_allblocks/build/tests/ln1/test_allblocks_ln1.elf
```

If compilation fails, inspect the error before proceeding. The source files and the generated parameter/fixture headers are tracked by compiler dependency files.

## Step 5: run on the custom Spike

```bash
make -C experiments/integer_inference_allblocks/tests/ln1 \
    CVW_ROOT="$CVW_ROOT" \
    SPIKE="$SPIKE_XTILE" \
    run
```

```bash
printf 'All-block LN1 Spike exit status: %s\n' "$?"
```

Expected output:

```text
Block 0: 13 LN1 cases, all checkpoints and forward PASS
Block 1: 13 LN1 cases, all checkpoints and forward PASS
Block 2: 13 LN1 cases, all checkpoints and forward PASS
Block 3: 13 LN1 cases, all checkpoints and forward PASS
Block 4: 13 LN1 cases, all checkpoints and forward PASS
Block 5: 13 LN1 cases, all checkpoints and forward PASS
Block 6: 13 LN1 cases, all checkpoints and forward PASS
Block 7: 13 LN1 cases, all checkpoints and forward PASS
PASS: 8 blocks, 104 LN1 cases, 425984 outputs per path, 0 mismatches; guards unchanged
```

Require exit status 0. Any mismatch prints the block, case, checkpoint, index, actual value and expected value, then returns failure. Checkpoint failures identify whether to inspect arithmetic, requantization, or block parameter wiring.

## Local verification and remaining work

The uploaded 17 C parameter files matched the previously validated exporter output byte for byte. Local C checks on copies of the uploaded kernels and the new entry points passed all 104 cases and checkpoints, including buffer guards, with `-Wall -Wextra -Werror` and undefined-behavior sanitization.

Those local checks used a separate temporary requantization emulator because the RISC-V toolchain and Spike were unavailable here. The emulator is not included in this package. The production backend remains the uploaded XREQUANT code. The Makefile dependency paths and build commands passed a dry run using placeholder runtime files; an actual RV64 build and custom-instruction execution still require Steps 4–5 on your laptop.

After the Spike LN1 test passes, continue with Part 2: Q/K/V accumulation through XTILE, followed by the existing XREQUANT INT32-to-INT8 conversion for each projection. Keep the existing block-0 experiments as the comparison reference.
