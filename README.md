# TinyStories integer inference on RV64IMC

--

## Starting a new experiment

```bash
python3 scripts/new_experiment.py block0_kproj --purpose "Port and verify block-0 K projection"
```

Fill in the generated README/configuration and add its source and input data before running it. New experiment creation does not copy q-projection dimensions, constants, results, or a success status. For software/ISA variants, choose separate IDs such as `block0_qproj_b1` or `block0_qproj_x1`; keep the B0 run as a comparison reference. See `docs/experiment_workflow.md` and `EXPERIMENTS.md`.

## Notebook workflow

| Notebook | Purpose |
| --- | --- |
| `01_inspect_model.ipynb` | Architecture, parameters, tied weights |
| `02_export_block0.ipynb` | Original multi-prompt PT2E/LiteRT conversion and comparisons |
| `03_inspect_block0_graph.ipynb` | Operators, shapes, types, quantization metadata |
| `04_export_qproj_to_c.ipynb` | Synthetic case extraction and an accumulator-only C test |