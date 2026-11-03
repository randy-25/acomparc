# EE66455 - Advanced Computer Architecture Project

### Name: Randy Revaldo Pratama / 阮金鴻 <br> Student ID: 115165425

Presentation Slides = [Click here to go to the folder](PPTSlides/)

## Description
This project implements integer inference for TinyStories-3M’s eight transformer blocks on bare-metal RV64IMC. Two custom instructions,
XTILE for matrix accumulation and XREQUANT for fixed-point requantization, accelerate the computation. A modified Spike simulator verifies correctness and compares retired-instruction counts against the scalar C implementation.

## Model and execution scope

| Property | Configuration |
|---|---|
| Model | roneneldan/TinyStories-3M, GPT-Neo |
| Transformer blocks | 8 |
| Hidden features | 128 |
| Attention heads | 16 |
| Features per head | 8 |
| MLP hidden features | 512 |
| Batch / sequence length | 1 / 32 |
| ISA / ABI | rv64imc_zicsr / lp64 |

The C implementation executes blocks 0–7 with integer arithmetic and INT8 block boundaries. Internal calculations use INT32 and INT64.

Embedding, final model LayerNorm, tokenization, and the vocabulary head remain in the Python host workflow.

The W8A8 TFLite exports contain mixed INT8/FP32 operations. The bare-metal implementation replaces the required operations with the project's integer numerical policy.

## Retired-Instruction Profile Summary

- [Measurements and stage comparison](results/instruction_comparison_benchmark/report.md)
- [Detailed MAC, assembly, and bottleneck analysis](docs/instruction_analysis.md)

<i>The MAC instruction estimates and bottleneck analysis were calculated with assistance from ChatGPT using the measured Spike instruction counts and compiled disassembly.</i>

## Customized Instruction Cycle Comparison
- [Xtile Instruction Description](docs/xtile.md)
- [Xrequant Instruction Description](docs/xrequant.md)
- [Original Scalar Cycle Analysis](docs/scalar_cycle_analysis.md)
- [Xtile Based Instruction Rough Analysis](docs/systolic_array_estimation_cycle.md)
- [Rough Cycle Comparison](docs/comparison_scalar_custom.md)

## Directory layout

```text
project/
├── src/                              # Reusable implementation
│   ├── sw/                           # Bare-metal RV64IMC C code
│   │   ├── blocks/                   # Block 0–7 orchestration
│   │   ├── backends/scalar/           # Scalar projection and requantization
│   │   └── *.c, *.h                  # Shared kernels and custom-instruction wrappers
│   └── python/
│       └── tinystories_integer/      # Integer reference and parameter-export library
│
├── scripts/                          # Commands for export, validation, and analysis
│
├── models/
│   ├── tinystories_3m_fp32/           # FP32 checkpoint and tokenizer
│   ├── tinystories_3m_blocks_w8a8/    # Timestamped eight-block TFLite exports
│   ├── profiles/                     # Frozen integer policy and model hashes
│   └── artifact_manifest.json        # Artifact identities and checksums
│
├── tests/
│   ├── sw/
│   │   ├── full/                     # Eight-block C correctness and profiling
│   │   └── data/                     # Frozen regression vectors
│   └── python/                       # Archive and profiling-parser unit tests
│
├── experiments/
│   ├── component_tests/              # Individual-stage tests across all eight blocks
│   ├── block0/
│   │   ├── qproj/                    # Historical Q-projection experiment and results
│   │   └── notebooks/                # Original block-0 analysis
│   └── ...                           # Other experiments and Spike modification work
│
├── notebooks/                        # Current eight-block numerical analysis
│
├── docs/
│   ├── numerical_spec.md             # Equations, formats, and requantization rules
│   └── history/inference/            # Historical implementation guides and manifests
│
├── results/
│   ├── instruction_comparison_benchmark/  # Scalar/custom measurements and assembly
│   └── validation/                   # Retained numerical validation reports
│
├── requirements.txt                     # Core and host Python dependencies
├── build/                            # Generated outputs; ignored by Git
│   ├── parameter_export/             # Parameter archives and generated C files
│   ├── fixtures/                     # Generated C test checkpoints
│   └── tests/                        # Compiled binaries and profiling outputs
│
└── README.md                         # Project overview and execution instructions
```

`src/` holds the implementation, `scripts/` provides executable workflows,
and `tests/` verifies the main implementation. Smaller experiments remain
under `experiments/`. Generated files go into `build/`, while retained
measurement evidence belongs in `results/`.

## Prerequisites

### Python Dependency List

Use the project's existing virtual environment for the verified workflow.

Core parameter and integer-reference tools require NumPy. Reading TFLite
models also requires the supported generated schema available in the
project's LiteRT environment. Full host diagnostics require PyTorch,
Transformers, and LiteRT.

Dependency lists are provided in:

- `requirements.txt`

Install project requirements:

```bash
python -m pip install -r requirements.txt
```

TFLite quantization additionally requires `torchao` and `litert-torch`, using versions compatible with the original export environment. The saved model export report records the package versions.

### Baremetal Requirements

#### RISC-V GNU toolchain

Install the compiler and Toolchain: 

```
https://github.com/riscv-collab/riscv-gnu-toolchain
```

Select the **bare-metal Newlib toolchain**, which provides `riscv64-unknown-elf-gcc`. Ensure the toolchain supports this project's `rv64imc_zicsr` architecture and `lp64` ABI, including a compatible `libgcc`.

Verify:

```bash
riscv64-unknown-elf-gcc --version
riscv64-unknown-elf-objdump --version
```

#### CV-Wally runtime

Clone the [CV-Wally repository](https://github.com/openhwfoundation/cvw)
to obtain the bare-metal runtime files:

```bash
git clone https://github.com/openhwfoundation/cvw.git /path/to/cvw
export CVW_ROOT="/path/to/cvw"
```

Replace `/path/to/cvw` with your chosen location.

The Makefiles use the runtime under `$CVW_ROOT/examples/C/common/`, including `crt.S`, `syscalls.c`, `test.ld`, and supporting headers. Keep this directory intact.

Verify:

```bash
ls "$CVW_ROOT/examples/C/common/"{crt.S,syscalls.c,test.ld}
```

The software tests use these runtime files directly. Running the Wally RTL simulator is not required for Spike execution.

#### Spike simulators

Two simulator builds are used:

- **Standard Spike** for the original scalar backend.
- **Modified Spike with both XTILE and XREQUANT** for the custom backend.

Follow the **[Spike setup guide](spike_custom/README.md)** for standard installation,
the custom build's base revision and patches, and verification commands.

After installation, configure the custom executable:

```bash
export SPIKE_CUSTOM="/path/to/custom/spike"
```

`SPIKE_CUSTOM` must point to the executable, not its source directory. For example:
```bash
export SPIKE_CUSTOM="$HOME/Documents/ACompArc/riscv-isa-sim-xtile/build/spike"
```

## Existing model inputs

The default verified model directory is:

```text
models/tinystories_3m_blocks_w8a8/20261011T134954617091Z/
```

It contains blocks 0–7. Host validation also uses the saved validation
inputs and the FP32 model/tokenizer under:

```text
models/tinystories_3m_fp32/
```

The numerical profile is:

```text
models/profiles/int8_boundaries_v1.json
```

## Export integer parameters

```bash
python scripts/export_c_parameters.py
```

Default outputs:

```text
build/parameter_export/parameters.json
build/parameter_export/parameters.npz
build/parameter_export/c/integer_parameter_types.h
build/parameter_export/c/block0_params.h
build/parameter_export/c/block0_params.c
...
build/parameter_export/c/block7_params.h
build/parameter_export/c/block7_params.c
```

The export includes weights, biases where present, zero points, requantization multipliers and shifts, LayerNorm constants, softmax/GELU lookup tables, residual conversions, and block-boundary conversions.

Existing outputs require `--overwrite` or another `--out-dir`.

## Generate full-chain C fixtures

The fixtures here provides the input reference and output golden reference for the testing. 

```bash
python scripts/export_full_fixtures.py
```

Default fixture directory:

```text
build/fixtures/full/
```

## Build and run the scalar C chain

Compile:

```bash
make -C tests/sw/full \
    BACKEND=scalar MODE=test \
    CVW_ROOT="$CVW_ROOT" riscv
```

Run on spike:

```bash
make -C tests/sw/full \
    BACKEND=scalar MODE=test \
    CVW_ROOT="$CVW_ROOT" SPIKE=spike run
```

You can use `"$SPIKE_CUSTOM"` for this scalar run too. 

## Build and run the custom C chain

Compile:
```bash
make -C tests/sw/full \
    BACKEND=custom MODE=test \
    CVW_ROOT="$CVW_ROOT" riscv
```

Run on spike:

```bash
make -C tests/sw/full \
    BACKEND=custom MODE=test \
    CVW_ROOT="$CVW_ROOT" SPIKE="$SPIKE_CUSTOM" run
```

Both backends use the same block parameters and numerical policy. The custom backend uses XTILE for dense projection accumulation and XREQUANT for supported conversions. Shared software performs the other calculations. Computation and requantization remain separate calls.

Expected for each backend:

- Eight complete C chains
- 64 block checks
- 262144 INT8 boundary values
- 262144 INT32 boundary values
- Zero mismatches and unchanged guards

Build outputs are separated by backend and mode under `build/tests/full/`.

## Retired-instruction profiling

Run profiling for input case 0 across all eight blocks:

```bash
# Scalar backend
make -C tests/sw/full \
    BACKEND=scalar MODE=profile PROFILE_CASES=1 \
    CVW_ROOT="$CVW_ROOT" SPIKE=spike run

# Custom backend
make -C tests/sw/full \
    BACKEND=custom MODE=profile PROFILE_CASES=1 \
    CVW_ROOT="$CVW_ROOT" SPIKE="$SPIKE_CUSTOM" run
```

`SPIKE_CUSTOM` must point to the modified Spike executable.

Output format:

```text
PROFILE,backend,case,block,region,raw_minstret
```

Counts measure retired instructions, not clock cycles. `chain` measures
the complete eight-block execution; `forward` measures one block.
Stage measurements are reported separately. Block `-1` identifies
chain-level records.

For stage comparisons, sum each stage across blocks 0–7. Use the
directly measured `chain` count for the whole-chain comparison.

## Export new W8A8 TFLite blocks

```bash
python scripts/export_tflite.py
```

Loads the local FP32 checkpoint, calibrates each block using captured FP32 block inputs, and exports into a new timestamped model directory.

Outputs include eight TFLite blocks, `validation_inputs.npz`, and `export_report.json`. 


## Host text generation

```bash
python scripts/generate_text.py \
    --params build/parameter_export \
    --prompt 'Once upon a time, a small fox' \
    --max-new-tokens 20 \
    --context-mode prefix \
    --out build/fox_generation.json
```

This is a Python host workflow using FP32 embedding and final-head computations around the selected transformer-block path.

Use `--help` on each script for its available options.

## AI Usage

ChatGPT was used to assist with code generation and retired-instruction count analysis.



