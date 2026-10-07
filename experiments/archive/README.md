# Source map


| Original notebook | Active workflow | Historical source |
| --- | --- | --- |
| `inspect_model.ipynb` | `notebooks/01_inspect_model.ipynb` | `experiments/archive/inspect_model.ipynb` |
| `testoneTransformer.ipynb` | Optional preliminary FP32 conversion in notebook 02 | `experiments/archive/testoneTransformer.ipynb` |
| `OneTransformer(1).ipynb` | Multi-prompt export in 02, graph analysis in 03, optional generation experiment | `experiments/archive/OneTransformer.ipynb` |
| `model_to_c.ipynb` | `notebooks/04_export_qproj_to_c.ipynb` | `experiments/archive/model_to_c.ipynb` |



Archived notebooks retain exploratory source and historical paths. They are not the supported execution sequence. Active notebooks clear saved outputs and use relative paths, independent kernels, and unique run records. The multi-prompt quantization path
and original calibration prompt strings are retained.

The measured C kernel, reference headers, and profiler are preserved in `experiments/block0_qproj/`. They are distinct from notebook 04's generated accumulator test. The original user-supplied Git history is retained in the output zip.
