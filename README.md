# TinyStories integer inference on RV64IMC


## Environment

Host environment: Ubuntu 24.04, Python 3, and the existing virtual environment.
The current virtual environment is at `../.venv` relative to the project root.
Use its matching Jupyter kernel for the notebooks.

### Python packages

The complete pinned package list is in [requirements.txt](requirements.txt).
The main packages are:

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.13.0 | PyTorch model execution |
| `transformers` | 5.17.0 | TinyStories / GPT-Neo model and tokenizer |
| `torchao` | 0.18.0 | Quantization tooling |
| `litert-torch` | 0.9.4 | PyTorch-to-LiteRT conversion |
| `ai-edge-litert` | 2.2.0 | TFLite model interpretation |
| `numpy` | 2.5.3 | Tensor data and NPZ fixtures |
| `ipykernel` | 7.3.0 | Jupyter kernel for the virtual environment |

### C, RISC-V, and RTL tools

| Tool | Purpose |
|---|---|
| GCC and Make | Host correctness tests and builds |
| `riscv64-unknown-elf-gcc` and matching binutils | Bare-metal compilation, linking, and ELF inspection |
| Spike | RISC-V execution and instruction traces |
| CV-Wally | Bare-metal runtime files and processor RTL |
| Verilator and CV-Wally `wsim` | RTL simulation |

CV-Wally defaults to `$HOME/Documents/ACompArc/cvw`. The runtime files are
`crt.S`, `syscalls.c`, and `test.ld` under `examples/C/common/`.

For a different CV-Wally location, pass it to the run recorder:

```bash
python3 scripts/run_experiment.py block0_qproj \
  --make-var CVW_ROOT=/path/to/cvw
```

### Current baseline build settings

| Setting | Value |
|---|---|
| ISA | `rv64imc_zicsr_zifencei` |
| ABI | `lp64` |
| Code model | `medany` |
| Optimization | `-O2` |
| Linking | Static bare-metal; `-nostdlib -static`, CV-Wally `test.ld`, and `-lgcc` |

The existing Makefile's RTL target defaults to `rv64gc` as a smoke-test
configuration. Select the intended RV64IMC RTL configuration for hardware measurements.

## Experiments

See [EXPERIMENTS.md](EXPERIMENTS.md) for the experiment register and the complete
steps to create, configure, record, verify, and report an experiment.
See [results/README.md](results/README.md) for the results index.

## Notebook workflow

| Notebook | Purpose |
| --- | --- |
| `01_inspect_model.ipynb` | Architecture, parameters, tied weights |
| `02_export_block0.ipynb` | Original multi-prompt PT2E/LiteRT conversion and comparisons |
| `03_inspect_block0_graph.ipynb` | Operators, shapes, types, quantization metadata |
| `04_export_qproj_to_c.ipynb` | Synthetic case extraction and an accumulator-only C test |
