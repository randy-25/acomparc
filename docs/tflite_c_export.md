# TinyStories model and C parameter export

Run commands from the project root using the project's Python environment.

## Existing verified models to C parameters

```bash
python scripts/export_c_parameters.py
```

Default inputs:

- Models: `models/tinystories_3m_blocks_w8a8/20261011T134954617091Z/`
- Numerical profile: `models/profiles/int8_boundaries_v1.json`

Default outputs under `build/parameter_export/`:

- `parameters.json`
- `parameters.npz`
- `c/integer_parameter_types.h`
- `c/block0_params.h` and `.c` through `c/block7_params.h` and `.c`

The exporter reads TFLite constants and quantization metadata, then
prepares the integer parameters required by `src/sw/`.

| Parameter group | Exported contents |
|---|---|
| Projections | INT8 weights, biases where present, zero points, multipliers, shifts |
| LayerNorm | Encoded gamma/beta, integer epsilon, reciprocal-square-root and conversion constants |
| Attention | Score/context conversion constants and exponential lookup table |
| GELU | Intermediate conversion constants, tanh lookup table, final rounding parameters |
| Residuals and boundaries | Alignment factors and conversions into the selected integer formats |

The fixed-point choices come from the project's numerical policy;
they are not all specified by the TFLite model itself.

For another output directory:

```bash
python scripts/export_c_parameters.py --out-dir build/new_parameter_export
```

Existing outputs require `--overwrite` or a new output directory.

## FP32 checkpoint to new W8A8 TFLite blocks

```bash
python scripts/export_tflite.py
```

Loads the local checkpoint and tokenizer from
`models/tinystories_3m_fp32/`.

Exports eight blocks into a new timestamped directory under
`models/tinystories_3m_blocks_w8a8/`, together with:

- `validation_inputs.npz`
- `export_report.json`

The workflow retains the notebook's calibration corpus and checks
isolated block execution, saved-model reload consistency, finite outputs,
and six INT8 projections per block.

These TFLite graphs contain mixed INT8/FP32 operations.
They are not the bare-metal integer implementation.

The extracted script passed its command-line help check.
A complete quantization rerun after extraction has not yet been verified.

## Model compatibility

The C exporter checks TFLite hashes against the frozen profile.
A new export must satisfy the supported graph layout and integer policy
and undergo numerical validation before replacing the verified models.

Changing recorded hashes alone does not establish compatibility.

## C test fixtures

```bash
python scripts/export_full_fixtures.py
```

Writes test inputs and golden INT8/INT32 block-boundary outputs as C
arrays under `build/fixtures/full/`.

Fixtures are test data, separate from model parameters.

## Implementation and arithmetic

The scripts use the shared implementation under
`src/python/tinystories_integer/`.

See [the numerical specification](numerical_spec.md) for equations,
formats, rounding rules, and requantization points.

Use `--help` on either exporter to inspect its available options.
