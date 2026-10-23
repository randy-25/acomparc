# Spike Implementation: XTILE and XREQUANT

This directory packages one modified Spike simulator containing both custom engines:

| Engine | Interface | Role |
|---|---|---|
| XTILE | Five custom-0 instructions, opcode `0x0b` | Centered integer matrix accumulation |
| XREQUANT | Five custom-1 instructions, opcode `0x2b` | Buffer-wide integer requantization in five conversion modes |

Detailed ISA and arithmetic documentation belongs in [XTILE](../xtile/README.md) and [XREQUANT](../xrequant/README.md). This document covers source packaging, rebuilding, and validation.

Both engines are functional implementations in Spike. Spike does not model CV-Wally cycle timing. No RTL implementation is included here.

## Files

| Item | Purpose |
|---|---|
| `base_commit.txt` | Original Spike commit |
| `xtile_xrequant.patch` | Combined changes relative to that base, including all ten new instruction headers |
| `modified/riscv/` | Copies of the five modified shared files and ten new instruction headers |
| `sources.sha256` | SHA-256 hashes of those 15 copied source files |
| `refresh_package.py` | Re-export this package from an external modified Spike checkout without changing that checkout or its Git index |

Base Spike commit:

```text
76ced61f540a55d6d26a23e6dadd02f83ea634ca
```

### Verification status of the supplied package

The combined package was assembled from the uploaded 15-file source snapshot. Its patch was applied to the original five shared files reconstructed by reversing the earlier XTILE patch. Those reconstructed files match the base blob IDs recorded in that earlier patch. Patch application reproduces all 15 uploaded files byte for byte, including all ten instruction headers.

The ten custom instruction encodings were checked against all 1,000 MATCH/MASK pairs found in the supplied `encoding.h`; no overlaps were found involving these custom encodings.

The developer's existing installation passed the engine tests and integrated block-0 tests. A fresh full checkout/build/test from this combined package has not yet been demonstrated. The steps below perform that remaining reproduction check.

## How to follow this tutorial

Use Bash and keep the same terminal open throughout the tutorial. Shell variables and the current directory carry over between commands.

Each code block is a separate step. Run one block, check its result, and then continue. A command split across lines with trailing backslashes is still one command. No test loop automatically compiles or runs later tests.

Stop when a command fails. Do not run a previously built ELF after a failed compilation. The status-check blocks must run immediately after the command they check; another intervening command would replace `$?`. The status check itself prints the previous result and does not automatically stop your shell.

Commands such as `git apply --check`, `git diff --check`, `cmp`, and `test` normally produce no output on success. When needed, inspect their status immediately with `printf 'Exit status: %s\n' "$?"`.

## 1. Requirements and paths

Required tools:

- Git, Make, Python 3, GCC/G++, device-tree-compiler, and Boost system/regex development libraries.
- RISC-V bare-metal GCC and binutils on PATH.
- A separate CV-Wally checkout containing `examples/C/common/crt.S`, `syscalls.c`, `test.ld`, and their included headers.

### 1.1 Enter the project repository

Replace the placeholder with your project repository's absolute path. On the original development machine it is `/home/randy/Documents/ACompArc/model/project`.

Enter that directory:

```bash
cd "/absolute/path/to/project"
```

Set the external CV-Wally path. On the original machine it is `/home/randy/Documents/ACompArc/cvw`:

```bash
export CVW_ROOT="/absolute/path/to/cvw"
```

Record the project root before changing directories:

```bash
spike_project_root="$PWD"

spike_package="$spike_project_root/experiments/spike"

spike_reproduce_root="$spike_project_root/../spike-xtile-xrequant-reproduce"

spike_reproduce_prefix="$HOME/.local/spike-xtile-xrequant-reproduce"

export SPIKE_REPRODUCE="$spike_reproduce_prefix/bin/spike"
```

`$HOME` is the standard shell variable for the current user's home directory. It is not reassigned. The source checkout and installation prefix are outside this project repository.

`SPIKE_XTILE` remains the variable for the existing working simulator. This tutorial uses `SPIKE_REPRODUCE` for the fresh binary.

The reproduction source directory must be new. The installation prefix must be separate from existing simulator installations. If either chosen path is already in use, choose a different reproduction path before proceeding.

Record the CV-Wally commit:

```bash
git -C "$CVW_ROOT" rev-parse HEAD
```

Check the RISC-V compiler:

```bash
riscv64-unknown-elf-gcc --version
```

## 2. Verify the packaged source copies

Enter the package directory:

```bash
cd "$spike_package"

sha256sum -c sources.sha256
```

Check the exit status immediately. Continue only if it is **0**:

```bash
printf 'Source checksum exit status: %s\n' "$?"
```

Expect all 15 entries to report `OK`, with exit status 0.

## 3. Clone the original Spike and apply the combined patch

Clone into the new external source directory:

```bash
git clone https://github.com/riscv-software-src/riscv-isa-sim.git \
    "$spike_reproduce_root"

cd "$spike_reproduce_root"
```

Read the pinned base commit:

```bash
spike_base_commit="$(cat "$spike_package/base_commit.txt")"
```

Display it. Expect `76ced61f540a55d6d26a23e6dadd02f83ea634ca`:

```bash
printf '%s\n' "$spike_base_commit"
```

Check out that exact version:

```bash
git checkout --detach "$spike_base_commit"

git apply --check "$spike_package/xtile_xrequant.patch"
```

Apply the combined patch:

```bash
git apply "$spike_package/xtile_xrequant.patch"
```


Check the tracked changes for whitespace errors:

```bash
git diff --check
```


## 4. Verify every patched source file

Compare each file individually with its packaged copy. Each `cmp` command should produce no output and return status 0. If any comparison reports a difference or missing file, stop before building.

These comparisons include all ten new instruction headers, which ordinary `git diff` can omit while they are untracked.


```bash
cmp "$spike_package/modified/riscv/encoding.h" \
    "$spike_reproduce_root/riscv/encoding.h"

cmp "$spike_package/modified/riscv/processor.cc" \
    "$spike_reproduce_root/riscv/processor.cc"

cmp "$spike_package/modified/riscv/processor.h" \
    "$spike_reproduce_root/riscv/processor.h"

cmp "$spike_package/modified/riscv/riscv.mk.in" \
    "$spike_reproduce_root/riscv/riscv.mk.in"

cmp "$spike_package/modified/riscv/trap.h" \
    "$spike_reproduce_root/riscv/trap.h"

cmp "$spike_package/modified/riscv/insns/xtile_cfg.h" \
    "$spike_reproduce_root/riscv/insns/xtile_cfg.h"

cmp "$spike_package/modified/riscv/insns/xtile_set_ab.h" \
    "$spike_reproduce_root/riscv/insns/xtile_set_ab.h"

cmp "$spike_package/modified/riscv/insns/xtile_set_c.h" \
    "$spike_reproduce_root/riscv/insns/xtile_set_c.h"

cmp "$spike_package/modified/riscv/insns/xtile_set_zp.h" \
    "$spike_reproduce_root/riscv/insns/xtile_set_zp.h"

cmp "$spike_package/modified/riscv/insns/xtile_run.h" \
    "$spike_reproduce_root/riscv/insns/xtile_run.h"

cmp "$spike_package/modified/riscv/insns/xrequant_cfg.h" \
    "$spike_reproduce_root/riscv/insns/xrequant_cfg.h"

cmp "$spike_package/modified/riscv/insns/xrequant_set_io.h" \
    "$spike_reproduce_root/riscv/insns/xrequant_set_io.h"

cmp "$spike_package/modified/riscv/insns/xrequant_set_mul.h" \
    "$spike_reproduce_root/riscv/insns/xrequant_set_mul.h"

cmp "$spike_package/modified/riscv/insns/xrequant_set_zp.h" \
    "$spike_reproduce_root/riscv/insns/xrequant_set_zp.h"

cmp "$spike_package/modified/riscv/insns/xrequant_run.h" \
    "$spike_reproduce_root/riscv/insns/xrequant_run.h"
```

## 5. Configure, build, and install Spike

Return to the reproduced source root:

```bash
cd "$spike_reproduce_root"

mkdir build

cd build
```

Configure the separate installation prefix:

```bash
../configure --prefix="$spike_reproduce_prefix"
```

Compile with two parallel jobs:

```bash
make -j2
```

Install the compiled simulator:

```bash
make install
```

Check that the expected executable exists and is executable:

```bash
test -x "$SPIKE_REPRODUCE"
```

Display the executable path to use for all verification below:

```bash
printf '%s\n' "$SPIKE_REPRODUCE"
```

## 6. Prepare XREQUANT verification

All test sources must already be present in this project. The tests use the CV-Wally bare-metal runtime. Differential tests link the unchanged scalar requantization implementation as their reference.

CFG validation is compiled and run first. Each subsequent test has its own compilation and execution commands. A successful compilation is required before running that test.

Warnings from the shared runtime about unused parameters/functions can occur; determine success from the compiler's exit status and inspect any actual errors.

Return to the project root:

```bash
cd "$spike_project_root"
```

Record the runtime directory:

```bash
xrequant_common="$CVW_ROOT/examples/C/common"
```

Record the unchanged scalar reference directory:

```bash
xrequant_reference="experiments/integer_inference/src"
```

Create the output directory:

```bash
mkdir -p experiments/xrequant/build/i32_to_i8

mkdir -p experiments/xrequant/build/i64_modes

mkdir -p experiments/xrequant/build/i8_to_i32

mkdir -p experiments/xrequant/build/two_round
```

### 6.1 CFG validation

Compile:

```bash
riscv64-unknown-elf-gcc \
    -std=gnu11 -O2 -Wall -Wextra \
    -march=rv64imc_zicsr -mabi=lp64 -mcmodel=medany \
    -fno-tree-loop-distribute-patterns \
    -I "$xrequant_common" -I "$xrequant_reference" \
    experiments/xrequant/tests/i32_to_i8/test_xrequant_cfg_validation.c \
    experiments/xrequant/tests/i32_to_i8/xrequant_cfg_test.S \
    "$xrequant_common/crt.S" "$xrequant_common/syscalls.c" \
    -nostdlib -static -T "$xrequant_common/test.ld" -lgcc \
    -o experiments/xrequant/build/i32_to_i8/test_xrequant_cfg_validation.elf
```

Check the exit status immediately. Continue only if it is **0**:

```bash
printf 'Compile exit status: %s\n' "$?"
```

Run only after compilation returns status 0:

```bash
"$SPIKE_REPRODUCE" \
    --isa=rv64imc_zicsr \
    experiments/xrequant/build/i32_to_i8/test_xrequant_cfg_validation.elf
```

Check the exit status immediately. Continue only if it is **0**:

```bash
printf 'Test exit status: %s\n' "$?"
```

Expected: 20 CFG cases, 9 expected traps, 11 RUN checks passed.

### 6.2 INT32 to INT8 smoke test

Compile:

```bash
riscv64-unknown-elf-gcc \
    -std=gnu11 -O2 -Wall -Wextra \
    -march=rv64imc_zicsr -mabi=lp64 -mcmodel=medany \
    -fno-tree-loop-distribute-patterns \
    -I "$xrequant_common" -I "$xrequant_reference" \
    experiments/xrequant/tests/i32_to_i8/test_xrequant_i32_to_i8.c \
    "$xrequant_common/crt.S" "$xrequant_common/syscalls.c" \
    -nostdlib -static -T "$xrequant_common/test.ld" -lgcc \
    -o experiments/xrequant/build/i32_to_i8/test_xrequant_i32_to_i8.elf
```

Check the exit status immediately. Continue only if it is **0**:

```bash
printf 'Compile exit status: %s\n' "$?"
```

Run only after compilation returns status 0:

```bash
"$SPIKE_REPRODUCE" \
    --isa=rv64imc_zicsr \
    experiments/xrequant/build/i32_to_i8/test_xrequant_i32_to_i8.elf
```

Check the exit status immediately. Continue only if it is **0**:

```bash
printf 'Test exit status: %s\n' "$?"
```

Expected: 2 runs, 24 INT8 outputs, zero mismatches; output guards unchanged.

### 6.3 INT32 to INT8 differential test

Compile:

```bash
riscv64-unknown-elf-gcc \
    -std=gnu11 -O2 -Wall -Wextra \
    -march=rv64imc_zicsr -mabi=lp64 -mcmodel=medany \
    -fno-tree-loop-distribute-patterns \
    -I "$xrequant_common" -I "$xrequant_reference" \
    experiments/xrequant/tests/i32_to_i8/test_xrequant_i32_to_i8_differential.c \
    "$xrequant_reference/requantization.c" \
    "$xrequant_common/crt.S" "$xrequant_common/syscalls.c" \
    -nostdlib -static -T "$xrequant_common/test.ld" -lgcc \
    -o experiments/xrequant/build/i32_to_i8/test_xrequant_i32_to_i8_differential.elf
```

Check the exit status immediately. Continue only if it is **0**:

```bash
printf 'Compile exit status: %s\n' "$?"
```

Run only after compilation returns status 0:

```bash
"$SPIKE_REPRODUCE" \
    --isa=rv64imc_zicsr \
    experiments/xrequant/build/i32_to_i8/test_xrequant_i32_to_i8_differential.elf
```

Check the exit status immediately. Continue only if it is **0**:

```bash
printf 'Test exit status: %s\n' "$?"
```

Expected: 220 cases, 16,115 outputs match the scalar reference; guards and unused slots unchanged.

### 6.4 INT64 conversion modes

Compile:

```bash
riscv64-unknown-elf-gcc \
    -std=gnu11 -O2 -Wall -Wextra \
    -march=rv64imc_zicsr -mabi=lp64 -mcmodel=medany \
    -fno-tree-loop-distribute-patterns \
    -I "$xrequant_common" -I "$xrequant_reference" \
    experiments/xrequant/tests/i64_modes/test_xrequant_i64_modes.c \
    "$xrequant_reference/requantization.c" \
    "$xrequant_common/crt.S" "$xrequant_common/syscalls.c" \
    -nostdlib -static -T "$xrequant_common/test.ld" -lgcc \
    -o experiments/xrequant/build/i64_modes/test_xrequant_i64_modes.elf
```

Check the exit status immediately. Continue only if it is **0**:

```bash
printf 'Compile exit status: %s\n' "$?"
```

Run only after compilation returns status 0:

```bash
"$SPIKE_REPRODUCE" \
    --isa=rv64imc_zicsr \
    experiments/xrequant/build/i64_modes/test_xrequant_i64_modes.elf
```

Check the exit status immediately. Continue only if it is **0**:

```bash
printf 'Test exit status: %s\n' "$?"
```

Expected: 56 cases, 4,102 outputs match scalar references; guards, unused slots, and inactive buffers unchanged.

### 6.5 INT8 to INT32 conversion

Compile:

```bash
riscv64-unknown-elf-gcc \
    -std=gnu11 -O2 -Wall -Wextra \
    -march=rv64imc_zicsr -mabi=lp64 -mcmodel=medany \
    -fno-tree-loop-distribute-patterns \
    -I "$xrequant_common" -I "$xrequant_reference" \
    experiments/xrequant/tests/i8_to_i32/test_xrequant_i8_to_i32.c \
    "$xrequant_reference/requantization.c" \
    "$xrequant_common/crt.S" "$xrequant_common/syscalls.c" \
    -nostdlib -static -T "$xrequant_common/test.ld" -lgcc \
    -o experiments/xrequant/build/i8_to_i32/test_xrequant_i8_to_i32.elf
```

Check the exit status immediately. Continue only if it is **0**:

```bash
printf 'Compile exit status: %s\n' "$?"
```

Run only after compilation returns status 0:

```bash
"$SPIKE_REPRODUCE" \
    --isa=rv64imc_zicsr \
    experiments/xrequant/build/i8_to_i32/test_xrequant_i8_to_i32.elf
```

Check the exit status immediately. Continue only if it is **0**:

```bash
printf 'Test exit status: %s\n' "$?"
```

Expected: 48 cases, 3,504 outputs match the scalar reference; inputs, guards, and unused slots unchanged.

### 6.6 Two-stage rounding conversion

Compile:

```bash
riscv64-unknown-elf-gcc \
    -std=gnu11 -O2 -Wall -Wextra \
    -march=rv64imc_zicsr -mabi=lp64 -mcmodel=medany \
    -fno-tree-loop-distribute-patterns \
    -I "$xrequant_common" -I "$xrequant_reference" \
    experiments/xrequant/tests/two_round/test_xrequant_two_round.c \
    "$xrequant_reference/requantization.c" \
    "$xrequant_common/crt.S" "$xrequant_common/syscalls.c" \
    -nostdlib -static -T "$xrequant_common/test.ld" -lgcc \
    -o experiments/xrequant/build/two_round/test_xrequant_two_round.elf
```

Check the exit status immediately. Continue only if it is **0**:

```bash
printf 'Compile exit status: %s\n' "$?"
```

Run only after compilation returns status 0:

```bash
"$SPIKE_REPRODUCE" \
    --isa=rv64imc_zicsr \
    experiments/xrequant/build/two_round/test_xrequant_two_round.elf
```

Check the exit status immediately. Continue only if it is **0**:

```bash
printf 'Test exit status: %s\n' "$?"
```

Expected: 515 cases, 37,517 outputs; directed rounding and scalar comparisons pass.

### 6.7 RUN exceptions

Compile:

```bash
riscv64-unknown-elf-gcc \
    -std=gnu11 -O2 -Wall -Wextra \
    -march=rv64imc_zicsr -mabi=lp64 -mcmodel=medany \
    -fno-tree-loop-distribute-patterns \
    -I "$xrequant_common" -I "$xrequant_reference" \
    experiments/xrequant/tests/i32_to_i8/test_xrequant_i32_to_i8_exceptions.c \
    experiments/xrequant/tests/i32_to_i8/xrequant_run_test.S \
    "$xrequant_common/crt.S" "$xrequant_common/syscalls.c" \
    -nostdlib -static -T "$xrequant_common/test.ld" -lgcc \
    -o experiments/xrequant/build/i32_to_i8/test_xrequant_i32_to_i8_exceptions.elf
```

Check the exit status immediately. Continue only if it is **0**:

```bash
printf 'Compile exit status: %s\n' "$?"
```

Run only after compilation returns status 0:

```bash
"$SPIKE_REPRODUCE" \
    --isa=rv64imc_zicsr \
    experiments/xrequant/build/i32_to_i8/test_xrequant_i32_to_i8_exceptions.elf
```

Check the exit status immediately. Continue only if it is **0**:

```bash
printf 'Test exit status: %s\n' "$?"
```

Expected: 13 cases, 9 expected traps; output and guard checks pass.

### 6.8 Memory-access faults

This test requires **`-m1`**, because its fault addresses assume exactly 1 MiB of Spike RAM.

Compile:

```bash
riscv64-unknown-elf-gcc \
    -std=gnu11 -O2 -Wall -Wextra \
    -march=rv64imc_zicsr -mabi=lp64 -mcmodel=medany \
    -fno-tree-loop-distribute-patterns \
    -I "$xrequant_common" -I "$xrequant_reference" \
    experiments/xrequant/tests/i32_to_i8/test_xrequant_i32_to_i8_memory_faults.c \
    experiments/xrequant/tests/i32_to_i8/xrequant_run_test.S \
    "$xrequant_common/crt.S" "$xrequant_common/syscalls.c" \
    -nostdlib -static -T "$xrequant_common/test.ld" -lgcc \
    -o experiments/xrequant/build/i32_to_i8/test_xrequant_i32_to_i8_memory_faults.elf
```

Check the exit status immediately. Continue only if it is **0**:

```bash
printf 'Compile exit status: %s\n' "$?"
```

Run only after compilation returns status 0:

```bash
"$SPIKE_REPRODUCE" -m1 \
    --isa=rv64imc_zicsr \
    experiments/xrequant/build/i32_to_i8/test_xrequant_i32_to_i8_memory_faults.elf
```

Check the exit status immediately. Continue only if it is **0**:

```bash
printf 'Test exit status: %s\n' "$?"
```

Expected: 6 cases, 4 expected memory-access traps; output checks pass.

Expected traps are handled and checked by the relevant tests. The complete test must still exit with status 0. The listed exception and memory-fault suites primarily exercise mode 2; they are not exhaustive fault coverage of every conversion mode.

## 7. Prepare XTILE RUN verification

The four suites below verify that XTILE still works in the freshly reproduced combined simulator. They cover numerical results, boundary tiles, exceptions, and memory faults. They are the existing RUN regression suites, not every XTILE test in the repository.

Return to the project root:

```bash
cd "$spike_project_root"
```

Record the runtime directory:

```bash
xtile_common="$CVW_ROOT/examples/C/common"
```

Create the output directory:

```bash
mkdir -p experiments/xtile/build/run
```

### 7.1 INT8 matrix multiplication

Compile:

```bash
riscv64-unknown-elf-gcc \
    -std=gnu11 -O2 -Wall -Wextra \
    -march=rv64imc_zicsr -mabi=lp64 -mcmodel=medany \
    -fno-tree-loop-distribute-patterns \
    -I "$xtile_common" \
    experiments/xtile/tests/run/test_xtile_run_i8.c \
    "$xtile_common/crt.S" "$xtile_common/syscalls.c" \
    -nostdlib -static -T "$xtile_common/test.ld" -lgcc \
    -o experiments/xtile/build/run/test_xtile_run_i8.elf
```

Check the exit status immediately. Continue only if it is **0**:

```bash
printf 'Compile exit status: %s\n' "$?"
```

Run only after compilation returns status 0:

```bash
"$SPIKE_REPRODUCE" \
    --isa=rv64imc_zicsr \
    experiments/xtile/build/run/test_xtile_run_i8.elf
```

Check the exit status immediately. Continue only if it is **0**:

```bash
printf 'Test exit status: %s\n' "$?"
```

Expected: 9 matching INT32 outputs; output guards unchanged.

### 7.2 Boundary and arithmetic cases

Compile:

```bash
riscv64-unknown-elf-gcc \
    -std=gnu11 -O2 -Wall -Wextra \
    -march=rv64imc_zicsr -mabi=lp64 -mcmodel=medany \
    -fno-tree-loop-distribute-patterns \
    -I "$xtile_common" \
    experiments/xtile/tests/run/test_xtile_run_edges.c \
    "$xtile_common/crt.S" "$xtile_common/syscalls.c" \
    -nostdlib -static -T "$xtile_common/test.ld" -lgcc \
    -o experiments/xtile/build/run/test_xtile_run_edges.elf
```

Check the exit status immediately. Continue only if it is **0**:

```bash
printf 'Compile exit status: %s\n' "$?"
```

Run only after compilation returns status 0:

```bash
"$SPIKE_REPRODUCE" \
    --isa=rv64imc_zicsr \
    experiments/xtile/build/run/test_xtile_run_edges.elf
```

Check the exit status immediately. Continue only if it is **0**:

```bash
printf 'Test exit status: %s\n' "$?"
```

Expected: 17 cases, 557 outputs, all checks pass.

### 7.3 RUN exceptions

Compile:

```bash
riscv64-unknown-elf-gcc \
    -std=gnu11 -O2 -Wall -Wextra \
    -march=rv64imc_zicsr -mabi=lp64 -mcmodel=medany \
    -fno-tree-loop-distribute-patterns \
    -I "$xtile_common" \
    experiments/xtile/tests/run/test_xtile_run_exceptions.c \
    experiments/xtile/tests/run/xtile_run_test.S \
    "$xtile_common/crt.S" "$xtile_common/syscalls.c" \
    -nostdlib -static -T "$xtile_common/test.ld" -lgcc \
    -o experiments/xtile/build/run/test_xtile_run_exceptions.elf
```

Check the exit status immediately. Continue only if it is **0**:

```bash
printf 'Compile exit status: %s\n' "$?"
```

Run only after compilation returns status 0:

```bash
"$SPIKE_REPRODUCE" \
    --isa=rv64imc_zicsr \
    experiments/xtile/build/run/test_xtile_run_exceptions.elf
```

Check the exit status immediately. Continue only if it is **0**:

```bash
printf 'Test exit status: %s\n' "$?"
```

Expected: 16 cases, 13 expected traps, all checks pass.

### 7.4 Memory-access faults

Use **`-m1`** for this memory-fault suite.

Compile:

```bash
riscv64-unknown-elf-gcc \
    -std=gnu11 -O2 -Wall -Wextra \
    -march=rv64imc_zicsr -mabi=lp64 -mcmodel=medany \
    -fno-tree-loop-distribute-patterns \
    -I "$xtile_common" \
    experiments/xtile/tests/run/test_xtile_run_memory_faults.c \
    experiments/xtile/tests/run/xtile_run_test.S \
    "$xtile_common/crt.S" "$xtile_common/syscalls.c" \
    -nostdlib -static -T "$xtile_common/test.ld" -lgcc \
    -o experiments/xtile/build/run/test_xtile_run_memory_faults.elf
```

Check the exit status immediately. Continue only if it is **0**:

```bash
printf 'Compile exit status: %s\n' "$?"
```

Run only after compilation returns status 0:

```bash
"$SPIKE_REPRODUCE" -m1 \
    --isa=rv64imc_zicsr \
    experiments/xtile/build/run/test_xtile_run_memory_faults.elf
```

Check the exit status immediately. Continue only if it is **0**:

```bash
printf 'Test exit status: %s\n' "$?"
```

Expected: 12 cases, 10 expected memory-access traps, all checks pass.

## 8. Compile and verify integrated block 0

The separate accelerated workload must already contain both engine integrations. The Makefile's `riscv` target compiles the RISC-V workload without executing it. Running the ELF directly afterwards keeps compilation and execution separate.

Return to the project root:

```bash
cd "$spike_project_root"
```

Compile the full-block RISC-V test as needed:

```bash
make -C experiments/integer_inference_xtile/tests/block0/full \
    CVW_ROOT="$CVW_ROOT" \
    ISA=rv64imc_zicsr \
    riscv
```

Check the exit status immediately. Continue only if it is **0**:

```bash
printf 'Full-block compile exit status: %s\n' "$?"
```

Run the compiled ELF with the fresh simulator:

```bash
"$SPIKE_REPRODUCE" \
    --isa=rv64imc_zicsr \
    experiments/integer_inference_xtile/build/block0/full/test_block0.elf
```

Check the exit status immediately. Continue only if it is **0**:

```bash
printf 'Full-block test exit status: %s\n' "$?"
```

Expected: eight cases, 32,768 full-block T107 outputs, zero mismatches, exit status 0.

## 9. Optional profiling verification

This section separates profile compilation, execution, status checking, and log display. A unique log preserves previous measurements.

Return to the project root:

```bash
cd "$spike_project_root"
```

Compile the profiling ELF as needed:

```bash
make -C experiments/integer_inference_xtile/tests/block0/profile \
    CVW_ROOT="$CVW_ROOT" \
    ISA=rv64imc_zicsr \
    riscv
```

Check the exit status immediately. Continue only if it is **0**:

```bash
printf 'Profile compile exit status: %s\n' "$?"
```

Allocate a new log filename:

```bash
spike_profile_log="$(mktemp /tmp/block0-spike-reproduce-profile.XXXXXX.log)"
```

Display the log path:

```bash
printf '%s\n' "$spike_profile_log"
```

Run profiling and capture both normal output and errors in that log. Output appears when you display the log in the following step:

```bash
"$SPIKE_REPRODUCE" \
    --isa=rv64imc_zicsr \
    experiments/integer_inference_xtile/build/block0/profile/test_block0_profile.elf \
    > "$spike_profile_log" 2>&1
```

Check the exit status immediately. Continue only if it is **0**:

```bash
printf 'Profile test exit status: %s\n' "$?"
```

Display the saved output, including any error if the previous command failed:

```bash
cat "$spike_profile_log"
```

Expected: eight cases, 32,768 outputs checked per path, with zero mismatches for both full and staged execution.

The existing working installation reported an average raw full-block count of 4,538,779.5 retired instructions across eight cases after both integrations. Exact counts depend on workload source and compiler build. Report these as ISA-level instruction counts, not CV-Wally cycle speedups.

Reproduction is established only after the fresh simulator build succeeds and the applicable verification tests pass. Record the observed results before updating the package's verification status.

## 10. Refresh this package after later Spike edits

Use the external source checkout as the development tree. From this project repository's root:

```bash
python3 experiments/spike/refresh_package.py \
    "$HOME/Documents/ACompArc/riscv-isa-sim-xtile"
```

Replace that source path on another machine. The script reads the five shared files from the pinned base commit and the 15 current source files from the external checkout. A temporary Git repository generates a combined patch, including untracked instruction headers. It checks patch application and exact source equality before updating the copied sources, patch, and checksum manifest.

It changes only this source package, not the external checkout's files, Git index, or installed simulator. It exports only the explicit 15-file manifest; if a future engine change requires additional files, extend that manifest before exporting. A source refresh does not establish clean-build success: repeat the fresh-build and regression checks with new reproduction paths.

## 11. Migrating from the earlier XTILE-only package

The canonical package location is now `experiments/spike/`. The old location was `experiments/xtile/spike/`.

Place the combined package at the new location, verify it, and update references to the old package paths in documentation and scripts. Replace references to the earlier `xtile.patch` with the shared `xtile_xrequant.patch` where they describe the current combined simulator. The old package can be removed after the replacement is in place and reproduction has passed.

This reorganizes in-project submission files. It does not move the external Spike checkout or change the current simulator installation. XTILE and XREQUANT keep their own tests and architecture READMEs.
