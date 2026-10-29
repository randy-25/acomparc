# TinyStories blocks 0–7: reusable integer reference and parameter export

This package extracts the final calculations from `notebooks/analyze_integer_inference.ipynb`. It keeps the working block-0 experiments intact. All files belong under `experiments/integer_inference_allblocks/` in the project repository.

The Python reference executes all seven parts of every block using integer activation arithmetic. The exporter reads the eight inspected W8A8 TFLite models and produces Python parameter archives and C parameter data. C kernels and C orchestration for all eight blocks are the next implementation step; these scripts do not implement that port or invoke Spike.

## Fixed numerical policy

Every block has INT8 input and output with zero point 0. Adjacent blocks share the same boundary scale:

| Edge | Scale |
|---|---|
| Block 0 input | 2^-8 |
| Block 0 output / block 1 input | 2^-7 |
| Outputs of blocks 1–6 / inputs of blocks 2–7 | 2^-8 |
| Block 7 output | 2^-7 |

The seven parts are LN1; Q/K/V projections; causal attention; output projection plus residual; LN2; MLP expansion plus GELU; MLP output plus residual and final INT8 conversion. The equations, intermediate widths, multipliers and conversion points are in [docs/numerical_spec.md](docs/numerical_spec.md).

Dense layers use INT8 weights and activations with INT32 accumulators. Residuals inside a block use INT32 storage at scale 2^-16. Wider intermediate products use checked INT64 arithmetic. Multipliers are permitted to exceed positive INT32 where the validated INT64 product bounds allow it. Nothing recalibrates scales during export or inference.

Initial embedding encoding uses nearest-even rounding. Ordinary runtime requantization uses `(a*M + 2^(r-1)) >> r`, including its negative-value tie rule. The final GELU multiply retains the existing two-round Q31 rule. Attention remains the existing software calculation. These rules preserve the arithmetic used by XTILE and the five existing XREQUANT modes; the future C port can retain their existing execution roles.

The profile checks model SHA-256 values and the 74-operation graph layout. It applies to this model family and these eight model files. A changed model or numerical policy needs fresh validation and a new profile. Run Python normally, without `-O`, so the arithmetic and parameter assertions remain active.

## Files and outputs

| Command | Purpose | Output |
|---|---|---|
| `python/export_parameters.py` | Read model constants and compute fixed integer parameters | `parameters.json`, `parameters.npz`, and `c/` |
| `python/run_integer.py` | Run actual INT8 codes through eight blocks | NPZ with all block outputs and INT32 pre-boundary values; JSON diagnostics |
| `python/validate_integer.py --mode regression` | Compare with saved notebook extraction vectors | `regression.json`; nonzero exit on mismatch |
| `python/validate_integer.py --mode full` | Reconstruct the four host diagnostic paths | `validation.json`, `chain_fixtures.npz` |
| `python/generate_text.py` | Greedy text comparison for four paths | JSON with tokens, text, logit margins and clipping counts |

`parameters.npz` stores integer arrays, deduplicating identical arrays such as the shared exponential LUT. JSON carries scalar coefficients, scales, tensor IDs, shapes, profile and checksums. Loading uses `allow_pickle=False`. No notebook execution is required.

The `c/` directory contains `integer_parameter_types.h` and `block0_params.h/.c` through `block7_params.h/.c`. Every block has 15 arrays, two LayerNorm objects and 99 numeric macros: the original 94 plus five input/output-boundary macros. These are integer C constants with INT64 multiplier declarations where needed. The shared type header defines `LayerNormI8Params`; the eventual C kernels must use this definition without declaring it a second time. These generated headers are not a drop-in modification of the old block-0 source tree.

## Step 1: enter the project and existing environment

Extract or copy the package so this README is at `experiments/integer_inference_allblocks/README.md`. It adds no changes to `experiments/integer_inference/` or `experiments/integer_inference_xtile/`.

```bash
cd "$HOME/Documents/ACompArc/model/project"
```

```bash
source .venv/bin/activate
```

Use the environment in which the notebook passed. NumPy is sufficient for integer execution and regression. Export additionally needs the generated schema from `ai-edge-litert` (or TensorFlow). Full validation and text generation additionally need PyTorch, Transformers, LiteRT, and the existing local FP32 model/tokenizer directory. They load local files only.

For a separate environment, `requirements-core.txt` and `requirements-host.txt` list the dependencies. They are not a version lock; retain your working notebook environment for reproduction. The original TFLite export report records NumPy 2.5.3, ai-edge-litert 2.2.0, PyTorch 2.13.0 and Transformers 5.17.0.

## Step 2: confirm the model files

```bash
ls models/tinystories_3m_blocks_w8a8/20261011T134954617091Z/block*_w8a8.tflite
```

Expect files for blocks 0 through 7. For full host validation, also retain `validation_inputs.npz` in that directory and the model/tokenizer at `models/tinystories_3m_fp32/`.

## Step 3: export all eight blocks

This creates a new staging output. It does not replace the working block-0 parameters.

```bash
python experiments/integer_inference_allblocks/python/export_parameters.py \
    --out-dir experiments/integer_inference_allblocks/build/parameter_export
```

```bash
printf 'Export exit status: %s\n' "$?"
```

Expect a final `PASS: exported eight blocks ...` and status 0. Errors stop the command. Inspect an error before proceeding. Existing output files require an explicit `--overwrite` or a different output directory.

Defaults locate the model directory and frozen profile relative to this package, rather than to your shell's working directory. `--models-dir` and `--profile` can supply explicit paths.

## Step 4: check extraction regression first

This needs only NumPy and the exported archive. It compares 64 block calls against independently evaluated original notebook definitions, covering 262144 INT8 outputs and 262144 INT32 pre-boundary outputs. The vectors also anchor block 0 to the existing C fixture's 32768 INT32 outputs.

```bash
python experiments/integer_inference_allblocks/python/validate_integer.py \
    --mode regression \
    --params experiments/integer_inference_allblocks/build/parameter_export \
    --out-dir experiments/integer_inference_allblocks/build/regression
```

```bash
printf 'Regression exit status: %s\n' "$?"
```

Expect zero mismatches and status 0. This checks preservation of the validated calculations; it does not measure quality on unseen text.

## Step 5: rerun the numerical and language-model diagnostics

The full check uses benchmark plus seven calibration sequences. It reconstructs their FP32 block boundaries, checks the saved benchmark boundaries and reconstructed final logits, then evaluates original FP32, direct mixed TFLite, TFLite with INT8 boundaries, and integer blocks with INT8 boundaries. The common FP32 final LayerNorm and language-model head turn each final hidden state into logits.

```bash
python experiments/integer_inference_allblocks/python/validate_integer.py \
    --mode full \
    --params experiments/integer_inference_allblocks/build/parameter_export \
    --out-dir experiments/integer_inference_allblocks/build/full_validation
```

```bash
printf 'Full validation exit status: %s\n' "$?"
```

`validation.json` records per-case/per-block error, saturation counts, predicted IDs, and next-token loss. Its loss summary weights all 248 targets equally: 31 from the benchmark and 217 from calibration. `chain_fixtures.npz` saves inputs, actual integer block outputs, INT32 pre-boundary values and the mixed-path INT8 outputs for later C fixture generation.

The saved notebook's combined cross-entropy/perplexity results are:

| Path | Cross-entropy, nats/token | Perplexity |
|---|---:|---:|
| Original FP32 | 3.879092 | 48.380271 |
| Direct TFLite | 3.914339 | 50.115930 |
| TFLite with INT8 boundaries | 3.897120 | 49.260385 |
| Integer blocks with INT8 boundaries | 3.911756 | 49.986634 |

The integer path's final benchmark hidden-state relative L2 error was 15.6708%, RMSE 0.0112653; all block-boundary clipping counts were zero. Calibration sequences were used during model calibration/policy analysis and are not a held-out quality evaluation.

## Step 6: generate text with a short prefix

Each path uses its own generated prefix. This command uses greedy argmax, at most 20 new tokens, and the notebook's EOS handling. Short prefixes are padded to the fixed graph size, with the prediction read at the last real input position. Causal attention excludes future padding from that prediction.

```bash
python experiments/integer_inference_allblocks/python/generate_text.py \
    --params experiments/integer_inference_allblocks/build/parameter_export \
    --prompt 'Once upon a time, a small fox' \
    --max-new-tokens 20 \
    --context-mode prefix \
    --out experiments/integer_inference_allblocks/build/fox_generation.json
```

```bash
printf 'Generation exit status: %s\n' "$?"
```

Prefix mode accepts 1–31 prompt tokens and stops when the prompt plus generated tokens reaches 32, or on EOS. `--paths integer` selects just the integer-block path, while retaining FP32 embedding and final-head computations.

## Step 7: repeat the full 32-token-window test

This matches the notebook's final hedgehog test by explicitly keeping the first 32 tokenizer tokens of the supplied prompt. The script prints the effective prompt and token count before running.

```bash
python experiments/integer_inference_allblocks/python/generate_text.py \
    --params experiments/integer_inference_allblocks/build/parameter_export \
    --prompt 'On a rainy afternoon, a little hedgehog found a red umbrella beside the garden gate. He wanted to return it to its owner, so he asked a friendly duck to help him search.' \
    --context-mode rolling32 \
    --truncate-prompt-to32 \
    --max-new-tokens 20 \
    --out experiments/integer_inference_allblocks/build/window32_generation.json
```

```bash
printf '32-token generation exit status: %s\n' "$?"
```

Every step recomputes the last 32 tokens with position indices 0–31; there is no KV cache. This is a rolling 32-token context, not a growing 52-token attention context. With no truncation option, rolling mode requires exactly 32 prompt tokens. The saved notebook reported zero initial-input and boundary clips throughout this test for both INT8-boundary paths.

## Optional: execute already encoded inputs

Inputs must have dtype INT8 and shape `[1,32,128]` or `[cases,1,32,128]`, using block 0's scale 2^-8 and zero point 0. The packaged regression file provides eight such inputs.

```bash
python experiments/integer_inference_allblocks/python/run_integer.py \
    --params experiments/integer_inference_allblocks/build/parameter_export \
    --input-codes experiments/integer_inference_allblocks/tests/data/chain_reference.npz \
    --input-key inputs \
    --out experiments/integer_inference_allblocks/build/integer_outputs.npz
```

`block_outputs` has shape `[cases,8,1,32,128]` and dtype INT8. `pre_boundary_outputs` has the same shape with INT32 values at scale 2^-16. `output` selects the final block. The adjacent JSON records stage saturations and softmax row-sum errors. This command uses neither LiteRT inference nor floating-point activation calculations.

## Verification and provenance

Local checks on the uploaded copies passed: all eight parameter bundles equal the original notebook calculations; all 64 block/case forwards match; all legacy block-0 final INT32 fixtures match; block-0 C export matches 94 numeric macros and all 15 arrays; all eight generated C data files compile with `gcc -std=c11 -Wall -Wextra -Werror`; parameter archive round-trip and corruption checks pass. The reproduced benchmark error matches the saved notebook.

PyTorch, Transformers and LiteRT were unavailable in the extraction environment. The constant-data checks used a test-only FlatBuffer reader; the production exporter uses the official generated TFLite schema. Therefore the production export dependency path, full host validation and generation commands still need the laptop run described above. No claim is made that those commands have run in your repository.

`docs/extraction_provenance.json` records source cells and hashes. `docs/notebook_validation_outputs.json` preserves the relevant saved notebook outputs. `tests/data/chain_reference.json` records vector provenance and checksums. The original notebook remains the numerical-development record; the Python scripts are the reusable execution/export path.

Archive checks can also be run independently:

```bash
python -m unittest discover \
    -s experiments/integer_inference_allblocks/tests \
    -p 'test_*.py'
```

Build outputs are ignored locally. Commit the scripts, profile, documentation, tests and notebook after reviewing them. Do not move the implementation outside `experiments/` until the C implementation and its comparison against these Python results pass.
