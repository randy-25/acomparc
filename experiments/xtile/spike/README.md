# XTILE Spike Implementation

This directory contains the source modifications for five custom instructions: `xtile.cfg`, `xtile.set_ab`, `xtile.set_c`, `xtile.set_zp`, and `xtile.run`.

The simulator implements functional matrix multiplication. It does not model CV-Wally cycle timing.

## Files

| File/directory | Purpose |
|---|---|
| `base_commit.txt` | Original Spike commit |
| `xtile.patch` | All XTILE changes relative to that commit, including new instruction files |
| `modified/` | Copies of the modified and added source files |

Base Spike commit:

```text
76ced61f540a55d6d26a23e6dadd02f83ea634ca
```

## Requirements

- Git, Make, GCC/G++, device-tree-compiler, and Boost system/regex development libraries.
- RISC-V bare-metal tools on PATH: `riscv64-unknown-elf-gcc` and `riscv64-unknown-elf-objdump`.
- A separate [CV-Wally checkout](https://github.com/openhwgroup/cvw), providing `examples/C/common/crt.S`, `syscalls.c`, `test.ld`, and their included headers. CV-Wally is not included in this repository.

Use Bash for the commands below. Start from **this Git repository's root** and set your paths:

```bash
export CVW_ROOT="/absolute/path/to/cvw"

xtile_project_root="$PWD"
xtile_patch="$xtile_project_root/experiments/xtile/spike/xtile.patch"
xtile_install_prefix="$HOME/.local/spike-xtile"
export SPIKE_XTILE="$xtile_install_prefix/bin/spike"
```

| Variable | Meaning |
|---|---|
| `CVW_ROOT` | Your external CV-Wally checkout |
| `xtile_project_root` | This repository's root, captured before changing directories |
| `xtile_install_prefix` | Installation directory for the modified Spike |
| `SPIKE_XTILE` | Full path to the modified Spike executable |

`$HOME` is the current user's home directory. The installation prefix can be changed to another writable location. If using an existing modified Spike installation, set `SPIKE_XTILE` to its executable and skip the build section.

Record the CV-Wally version used with your results:

```bash
git -C "$CVW_ROOT" rev-parse HEAD
```

## Build the modified Spike

Create a clean Spike checkout beside this repository. Proceed to each step only after the preceding command succeeds.

```bash
cd "$xtile_project_root/.."
git clone https://github.com/riscv-software-src/riscv-isa-sim.git spike-xtile-reproduce
cd spike-xtile-reproduce
git checkout 76ced61f540a55d6d26a23e6dadd02f83ea634ca

git apply --check "$xtile_patch"
git apply "$xtile_patch"

mkdir -p build
cd build
../configure --prefix="$xtile_install_prefix"
make -j2
make install
```

This separate installation preserves the original simulator. `xtile.patch` must contain the newly added instruction files as well as modifications to existing files.

## Compile a test

Return to the repository root and select a test name from the table below, without `.c`:

```bash
cd "$xtile_project_root"
mkdir -p experiments/xtile/build/run

xtile_common="$CVW_ROOT/examples/C/common"
xtile_test=test_xtile_run_i8
xtile_extra_args=()

case "$xtile_test" in
    test_xtile_run_exceptions|test_xtile_run_memory_faults)
        xtile_extra_args+=(experiments/xtile/tests/run/xtile_run_test.S)
        ;;
    test_projection_i8)
        xtile_extra_args+=(
            -I experiments/integer_inference/src
            experiments/integer_inference/src/projection.c
        )
        ;;
esac

riscv64-unknown-elf-gcc \
    -std=gnu11 -O2 -Wall -Wextra \
    -march=rv64imc_zicsr -mabi=lp64 -mcmodel=medany \
    -fno-tree-loop-distribute-patterns \
    -I "$xtile_common" \
    "experiments/xtile/tests/run/${xtile_test}.c" \
    "${xtile_extra_args[@]}" \
    "$xtile_common/crt.S" "$xtile_common/syscalls.c" \
    -nostdlib -static -T "$xtile_common/test.ld" -lgcc \
    -o "experiments/xtile/build/run/${xtile_test}.elf"

printf 'Compile exit status: %s\n' "$?"
```

## Run the compiled test

Only run after compilation returns status 0. The memory-fault suite requires **1 MiB RAM**; the command selects `-m1` for that test automatically.

```bash
xtile_spike_options=(--isa=rv64imc_zicsr)

if [ "$xtile_test" = test_xtile_run_memory_faults ]; then
    xtile_spike_options+=(-m1)
fi

"$SPIKE_XTILE" "${xtile_spike_options[@]}" \
    "experiments/xtile/build/run/${xtile_test}.elf"

printf 'Spike exit status: %s\n' "$?"
```

| Test name | Expected result |
|---|---|
| `test_xtile_run_i8` | 9 matching INT32 outputs; output sentinels unchanged |
| `test_xtile_run_edges` | 17 cases, 557 outputs, all checks pass |
| `test_xtile_run_exceptions` | 16 cases, 13 expected traps, all checks pass |
| `test_xtile_run_memory_faults` | 12 cases, 10 expected memory-access traps, all checks pass |
| `test_projection_i8` | Scalar counterpart: 9 matching INT32 outputs; output sentinels unchanged |

Successful tests return exit status 0. To run another test, change `xtile_test` in the compilation block and repeat both compilation and execution. The exception helper and original scalar kernel are selected automatically as needed.

Generate a separate disassembly for the selected test:

```bash
riscv64-unknown-elf-objdump -d \
    "experiments/xtile/build/run/${xtile_test}.elf" \
    > "experiments/xtile/build/run/${xtile_test}.dis"
```

## Instruction-count result

For the M=3, N=3, K=4 INT8 test, the supplied disassemblies give:

| Implementation | Executed CPU instructions |
|---|---:|
| Scalar projection, including argument preparation and function call/body | 438 |
| XTILE, including configuration, address/zero-point preparation, and RUN | 12 |

This is a **97.26% instruction reduction** for that computation region, excluding test initialization, printing, and verification. The counts are derived from disassembly and loop iterations; they are not a cycle-speedup measurement. Full-block XTILE performance has not yet been measured.
