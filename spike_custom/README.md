# Custom Spike: XTILE and XREQUANT

Builds a Spike simulator supporting both custom engines.

## Files

| Path | Purpose |
|---|---|
| `base_commit.txt` | Exact upstream Spike revision |
| `xtile_xrequant.patch` | Combined XTILE/XREQUANT changes |
| `modified/riscv/` | Copies of the modified source files |

The patch is the build input. The modified source copies are provided
for inspection; applying the patch already installs those changes.

## Prerequisites

On Ubuntu:

```bash
sudo apt-get update
sudo apt-get install
    git build-essential device-tree-compiler \
    libboost-regex-dev libboost-system-dev
```

## Build

Start from the model project's root directory. Keep the same terminal open throughout these steps and stop if any command fails.

Record the package location and choose a new external checkout:

```bash
spike_package="$PWD/spike_custom"
spike_source="$HOME/riscv-isa-sim-custom"
```

The chosen checkout directory must not already exist.

Clone upstream Spike and select the recorded revision:

```bash
git clone https://github.com/riscv-software-src/riscv-isa-sim.git "$spike_source"

cd "$spike_source"
git checkout --detach "$(cat "$spike_package/base_commit.txt")"
```

Check and apply the combined patch for the modification of Spike:

```bash
git apply --check "$spike_package/xtile_xrequant.patch"
git apply "$spike_package/xtile_xrequant.patch"
```

Configure and compile:

```bash
mkdir build
cd build
../configure --prefix="$HOME/.local/spike-custom"
make -j2
```

Use the executable directly from the build directory:

```bash
export SPIKE_CUSTOM="$spike_source/build/spike"
test -x "$SPIKE_CUSTOM"
```

Alternatively, install it into the configured prefix:

```bash
make install
export SPIKE_CUSTOM="$HOME/.local/spike-custom/bin/spike"
```

## Run

Return to the model project root. Follow the root README to build the
custom-backend test, then execute:

```bash
"$SPIKE_CUSTOM" --isa=rv64imc_zicsr build/tests/full/custom/test/test_allblocks.elf
```

The simulator must support both XTILE and XREQUANT.
Spike provides functional execution and instruction counts; it does
not model accelerator hardware cycle latency.
