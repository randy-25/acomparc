# Complete C block stack and scalar/custom instruction comparison

All seven component paths have passed on your combined XTILE/XREQUANT Spike for blocks 0–7. This package adds complete block entry points, the actual eight-block C chain, and a scalar backend for comparison using the same current numerical profile.

## Changes to review before extraction

| Path, relative to this experiment | Purpose |
|---|---|
| `src/blocks/block0.c/.h` through `block7.c/.h` | Add INT8-output `blockN_forward()`; existing component and boundary functions remain unchanged |
| `src/block_workspace.h` | Common caller-supplied workspace, with the working block-0 buffer-reuse arrangement |
| `src/model.c/.h` | Registry of the eight complete block functions and the actual block-0-through-7 driver |
| `src/backends/scalar/projection.c`, `requantization.c` | Ordinary integer C alternatives with the same APIs and arithmetic contracts |
| `python/export_full_fixtures.py` | Export the frozen notebook regression vectors as C data |
| `tests/full/test_allblocks.c`, `Makefile` | Correctness and instruction-profile builds, selected by backend and mode |
| `python/analyze_instruction_comparison.py` | Check custom opcodes and compare raw profile logs |
| `tests/full/test_analysis.py` | Test the opcode/log parser using explicitly synthetic records |

The archive contains complete updated versions of the 16 block files. `docs/c_full_changes.patch` shows their additions. It is for review; do not apply it again after extraction.

The verified accelerated files `src/projection.c`, `src/requantization.c`, `src/xtile.h` and `src/xrequant.h` are unchanged. Existing component Makefiles continue to use them. The new full-stack Makefile selects exactly one projection/requantization backend at build time:

| `BACKEND` | Dense accumulation | Requantization |
|---|---|---|
| `custom` | Existing XTILE backend | Existing XREQUANT backend |
| `scalar` | New scalar INT8 dot-product loops | New scalar implementations of the same five conversion modes |

Both builds share all block orchestration, LayerNorm, attention, GELU, residual arithmetic, parameters and fixtures. Both use `-O2`, `rv64imc_zicsr`, `lp64`, `medany`, and `-fno-tree-loop-distribute-patterns`, with no LTO. Only backend source selection and its printed label differ within a build mode.

`c_full_package.sha256` describes this snapshot. Earlier snapshot checksums differ for files subsequently extended. The original scalar and XTILE block-0 experiments remain separate.

## Integration and workspace behavior

Each `blockN_forward(input, output, workspace)` calls the existing seven component functions and the final INT8 boundary conversion. Inputs and outputs each contain 4096 INT8 values at their exported boundary scales.

The workspace reuses:

- The 16384-element MLP INT32 product allocation for LayerNorm's 4096-element feature buffer and projection accumulators in earlier phases.
- The 4096-element INT64 allocation for LayerNorm fine values and GELU branch-sum chunks.
- The first 4096 elements of the attention score allocation for the retained Part-4 residual and then the final INT32 sum, after attention no longer needs those scores.

After a complete block, `workspace->score_workspace[0..4095]` contains its INT32 pre-boundary sum at scale 2^-16. The next block overwrites it. Other workspace values are temporary.

The chain API is:

```c
void tinystories_blocks_forward(
    const int8_t *input,
    int8_t *output,
    int8_t *scratch,
    TransformerBlockWorkspace *workspace
);
```

Input, output and scratch each contain 4096 INT8 values and must be mutually disjoint and separate from the workspace. The driver alternates block outputs between scratch and output; block 7 finishes in output. Every next block receives the previous C block's actual codes directly. The exported boundary scales already agree at each edge. No intermediate floating-point bridge or allocation is used.

This integrates the transformer block stack. Token embeddings, final LayerNorm/language-model head and autoregressive token generation remain subsequent work.

## Correctness and profiling modes

`MODE=test` runs all eight saved cases twice:

1. The actual eight-block driver, checking final INT8 output, final INT32 sum, scratch's block-6 output, input preservation and guards.
2. A chain of complete `blockN_forward()` calls, checking every block's INT8 output and INT32 pre-boundary sum against the frozen notebook vectors.

This gives eight complete-driver runs plus 64 checked block boundaries per backend. All 262144 INT8 and 262144 INT32 reference values must match exactly. The 13-case component fixtures and their edge tests remain available separately.

`MODE=profile` defaults to the benchmark case (`PROFILE_CASES=1`). For each chosen case it measures three separately validated passes:

1. One raw interval around the complete driver.
2. A raw interval around each complete block call.
3. Seven raw stage intervals per block. Stage 7 includes the final INT8 boundary conversion.

Validation and printing occur outside the intervals. The counter probe requires adjacent reads to differ by 1 and a read separated by eight nops to differ by 9. Results remain **raw**; no overhead subtraction is applied. Sums of stage/block intervals are labeled aggregates, and need not equal an independently measured whole-block/chain interval.

Scalar and custom instruction ratios measure retired instructions. A custom RUN retires as one instruction while performing substantial internal work in Spike. These ratios do **not** establish accelerator cycle speedup, CPI, memory latency or CV-Wally RTL performance.

## Step 1: install the integration package

From the project root, inspect the archive; adjust its download location if needed.

```bash
cd "$HOME/Documents/ACompArc/model/project"
```

```bash
tar -tzf "$HOME/Downloads/tinystories-allblocks-full-comparison-c.tar.gz"
```

After reviewing the changes above, extract:

```bash
tar -xzf "$HOME/Downloads/tinystories-allblocks-full-comparison-c.tar.gz"
```

Optional delivery check:

```bash
cd experiments/integer_inference_allblocks
```

```bash
sha256sum -c c_full_package.sha256
```

```bash
cd "$HOME/Documents/ACompArc/model/project"
```

## Step 2: export the integration fixtures

Use the existing virtual environment. The exporter needs NumPy, the earlier Python modules and the already verified parameter archive. It does not execute LiteRT or load the FP32 checkpoint.

```bash
python experiments/integer_inference_allblocks/python/export_full_fixtures.py \
    --params experiments/integer_inference_allblocks/build/parameter_export \
    --out-dir experiments/integer_inference_allblocks/build/fixtures/full
```

```bash
printf 'Full-stack fixture export exit status: %s\n' "$?"
```

Expected:

```text
PASS: 8 input cases, 64 block boundaries, 262144 INT8 and 262144 INT32 reference values; C parameters verified
```

The exporter verifies the frozen-vector/profile hashes and all 17 generated C parameter files. It creates `full_golden.h`, `full_golden.c` and `manifest.json` under the ignored build directory. Existing outputs require `--overwrite` or a new directory.

## Step 3: compile and test the custom integration first

```bash
make -C experiments/integer_inference_allblocks/tests/full \
    BACKEND=custom MODE=test \
    CVW_ROOT="$HOME/Documents/ACompArc/cvw" \
    riscv
```

```bash
printf 'Custom full-stack build exit status: %s\n' "$?"
```

```bash
make -C experiments/integer_inference_allblocks/tests/full \
    BACKEND=custom MODE=test \
    CVW_ROOT="$HOME/Documents/ACompArc/cvw" \
    SPIKE="$HOME/Documents/ACompArc/spike-xtile-install/bin/spike" \
    run
```

```bash
printf 'Custom full-stack Spike exit status: %s\n' "$?"
```

Each case should report that the actual chain and every boundary pass. Expected final line:

```text
PASS: backend=custom; 8 complete C chains, 64 block checks, 262144 INT8 and 262144 INT32 boundary values; 0 mismatches; guards unchanged
```

Require status 0 before proceeding.

## Step 4: compile and test the scalar integration

```bash
make -C experiments/integer_inference_allblocks/tests/full \
    BACKEND=scalar MODE=test \
    CVW_ROOT="$HOME/Documents/ACompArc/cvw" \
    riscv
```

```bash
printf 'Scalar full-stack build exit status: %s\n' "$?"
```

```bash
make -C experiments/integer_inference_allblocks/tests/full \
    BACKEND=scalar MODE=test \
    CVW_ROOT="$HOME/Documents/ACompArc/cvw" \
    SPIKE="$HOME/Documents/ACompArc/spike-xtile-install/bin/spike" \
    run
```

```bash
printf 'Scalar full-stack Spike exit status: %s\n' "$?"
```

Expect the same passing result with `backend=scalar`, and status 0. Using the same installed Spike keeps the simulator environment consistent; the scalar executable contains no custom-instruction backend. Scalar simulation performs the dense loops as ordinary CPU instructions and can take longer.

Test binaries are isolated under `build/tests/full/custom/test/` and `build/tests/full/scalar/test/`.

## Step 5: build the benchmark profiling executables and assembly

Profile the same benchmark case in both binaries. These directories are separate from correctness-test objects and include the profile case count, preventing stale macro settings when changing case coverage.

```bash
make -C experiments/integer_inference_allblocks/tests/full \
    BACKEND=custom MODE=profile PROFILE_CASES=1 \
    CVW_ROOT="$HOME/Documents/ACompArc/cvw" \
    riscv disasm
```

```bash
printf 'Custom profile/disassembly build exit status: %s\n' "$?"
```

```bash
make -C experiments/integer_inference_allblocks/tests/full \
    BACKEND=scalar MODE=profile PROFILE_CASES=1 \
    CVW_ROOT="$HOME/Documents/ACompArc/cvw" \
    riscv disasm
```

```bash
printf 'Scalar profile/disassembly build exit status: %s\n' "$?"
```

Artifacts are:

| Backend | Directory under `build/tests/full/` |
|---|---|
| Custom | `custom/profile1/` |
| Scalar | `scalar/profile1/` |

Each contains `test_allblocks.elf` and `test_allblocks.dis`. The disassembly preserves symbol names, including `projection_accumulate_i8` and the requantization functions, for direct inspection.

## Step 6: collect raw custom counts

The redirection records the full run. Counter output is printed after validated intervals; no printing occurs inside them.

```bash
make -C experiments/integer_inference_allblocks/tests/full \
    BACKEND=custom MODE=profile PROFILE_CASES=1 \
    CVW_ROOT="$HOME/Documents/ACompArc/cvw" \
    SPIKE="$HOME/Documents/ACompArc/spike-xtile-install/bin/spike" \
    run > experiments/integer_inference_allblocks/build/tests/full/custom/profile1/profile.log 2>&1
```

```bash
printf 'Custom profile Spike exit status: %s\n' "$?"
```

```bash
cat experiments/integer_inference_allblocks/build/tests/full/custom/profile1/profile.log
```

Require status 0, the counter sanity result `adjacent=1, eight_nops=9`, profile rows, and:

```text
PASS: backend=custom mode=profile cases=1; all chain, block and stage outputs match
```

## Step 7: collect raw scalar counts

```bash
make -C experiments/integer_inference_allblocks/tests/full \
    BACKEND=scalar MODE=profile PROFILE_CASES=1 \
    CVW_ROOT="$HOME/Documents/ACompArc/cvw" \
    SPIKE="$HOME/Documents/ACompArc/spike-xtile-install/bin/spike" \
    run > experiments/integer_inference_allblocks/build/tests/full/scalar/profile1/profile.log 2>&1
```

```bash
printf 'Scalar profile Spike exit status: %s\n' "$?"
```

```bash
cat experiments/integer_inference_allblocks/build/tests/full/scalar/profile1/profile.log
```

Require the same sanity result, status 0, and:

```text
PASS: backend=scalar mode=profile cases=1; all chain, block and stage outputs match
```

`PROFILE_CASES=8` optionally profiles every saved case; use that value consistently for both backends. Those builds go under `profile8/`. The first comparison uses one matched benchmark case; it does not claim coverage of instruction-count variation across all inputs.

## Step 8: compare assembly and raw instruction counts

```bash
python experiments/integer_inference_allblocks/python/analyze_instruction_comparison.py \
    --scalar-dis experiments/integer_inference_allblocks/build/tests/full/scalar/profile1/test_allblocks.dis \
    --custom-dis experiments/integer_inference_allblocks/build/tests/full/custom/profile1/test_allblocks.dis \
    --scalar-log experiments/integer_inference_allblocks/build/tests/full/scalar/profile1/profile.log \
    --custom-log experiments/integer_inference_allblocks/build/tests/full/custom/profile1/profile.log \
    --out-dir experiments/integer_inference_allblocks/build/instruction_comparison
```

```bash
printf 'Instruction comparison exit status: %s\n' "$?"
```

The analyzer requires a successfully completed profile log from each backend, matching case/region sets and consistent raw aggregate rows. It detects custom instructions using the encoded opcode and funct3 fields, including when objdump prints `.4byte` or `.insn` instead of a custom mnemonic. Scalar assembly must contain neither custom opcode; custom assembly must contain both XTILE RUN and XREQUANT RUN.

Outputs:

| File | Contents |
|---|---|
| `report.md` | Custom opcode sites, whole-chain/stage count comparison, reductions and remaining stage shares |
| `static_functions.csv` | Static disassembly instruction/byte counts by function, with custom-opcode sites |
| `scalar_kernels.dis`, `custom_kernels.dis` | Extracted dense/requantization function assembly for inspection |
| `dynamic_regions.csv` | Raw per-case, per-block and per-stage scalar/custom counts, differences and ratios |
| `comparison.json` | Machine-readable static and dynamic results |

Static instruction sites are distinct from dynamic execution counts. The report uses sums over matched cases; stage totals also aggregate all eight blocks. Reported ratios are scalar/custom **retired-instruction ratios**, not measured hardware speedups. A region's share is relative to the custom stage intervals, excluding separately measured driver overhead.

For assembly-only analysis, omit both profile-log arguments and choose a different output directory. Dynamic analysis will then be marked pending. Existing reports require `--overwrite` or a new directory.

## Step 9: share the actual measurements for review

After the runs pass, package only the two disassemblies, two logs and comparison outputs:

```bash
tar -czf /tmp/tinystories-instruction-comparison.tar.gz \
    experiments/integer_inference_allblocks/build/tests/full/custom/profile1/test_allblocks.dis \
    experiments/integer_inference_allblocks/build/tests/full/custom/profile1/profile.log \
    experiments/integer_inference_allblocks/build/tests/full/scalar/profile1/test_allblocks.dis \
    experiments/integer_inference_allblocks/build/tests/full/scalar/profile1/profile.log \
    experiments/integer_inference_allblocks/build/instruction_comparison
```

Upload `/tmp/tinystories-instruction-comparison.tar.gz`. We can then inspect the actual scalar MAC/requantization loops, the custom setup/RUN sequence, measured reductions, and the remaining bottlenecks. No new instruction-count results are asserted before those laptop measurements exist.

## Local verification status

On local copies, the new scalar backend and the custom path with temporary instruction emulators both passed the eight complete-driver cases and every INT8/INT32 block boundary. The new scalar backend also passed the existing exhaustive 256-code GELU and Part-7 boundary/alias tests. Compilation used warnings as errors and undefined-behavior sanitization.

All existing component function bodies are unchanged. Both backends and both build modes passed Makefile dependency/build-command dry runs using placeholder runtime files. Profiling C logic was syntax checked; real RV64 assembly and counter execution were unavailable locally. Opcode/log parser tests use explicitly synthetic records and are not performance measurements.

The actual RV64 builds, custom-Spike integration tests, disassemblies and profile logs remain laptop checks. The temporary instruction emulators are outside this package. Native scalar tests exercise the delivered scalar backend itself.

Parser checks can be run independently:

```bash
python -m unittest discover \
    -s experiments/integer_inference_allblocks/tests/full \
    -p 'test_*.py'
```
