# Environment

`../requirements.txt` is preserved exactly from the supplied zip. It records a pinned
Python package environment, including LiteRT Torch 0.9.4, Torch 2.13.0, TorchAO 0.18.0,
Transformers 5.17.0, and ai-edge-litert 2.2.0. Its packages were not installed or upgraded
during cleanup. Use the user's existing `.venv` and matching Jupyter kernel.

The C/Spike experiment does not need ML packages. It needs GCC, Make,
`riscv64-unknown-elf-` GCC/binutils, Spike, and CV-Wally `examples/C/common`.
The Makefile uses RV64IMC plus Zicsr/Zifencei, lp64, medany, static bare-metal linking.
Future recorded runs capture resolved flags, compiler/Spike output, CV-Wally Git state,
runtime source/header bytes, Python version, and package list.

The imported baseline did not record original compiler/Spike versions, external runtime
source versions, or execution timestamps. Those fields remain unknown. The local cleanup
host's GCC check is recorded as a separate validation, not as the original environment.

Model downloads require network access or a populated cache for the pinned revision.
The wrapper preserves the user's original `layer_past` and tuple-return APIs. A successful
fresh installation from the pinned list has not been demonstrated in this environment.
