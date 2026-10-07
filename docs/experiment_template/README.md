# {{ID}}

Purpose: {{PURPOSE}}

Status: planned. No successful run has been recorded.

## Fill in before running

- Question/hypothesis and one primary change from the comparison baseline.
- Model/export identity and input paths with SHA256 values.
- Exact tensor shapes, dtypes, zero points, bias handling, and known MAC/output counts.
- Source files, fixed-point semantics, toolchain flags, and correctness reference.
- Commands and build output names in `experiment.json`.
- Validation checks and comparison run IDs.

Use `../block0_qproj/experiment.json` as a concrete example. Its tensor IDs, MAC counts,
requantization constants, and scalar-loop detection are specific to q-projection; replace
them for a different kernel. List every source/helper used so the run snapshots capture it.
Enable `capture_make_settings` and `capture_cvw_runtime` for the CV-Wally Makefile workflow;
that Makefile must expose `print-config`. `{build_dir}`, `{run_dir}`, `{python}`, and
`{riscv_prefix}` are expanded in command arguments. Commands run in this experiment folder.

## Run

From the project root, once the configuration is complete:

```bash
python3 scripts/run_experiment.py {{ID}} --dry-run
python3 scripts/run_experiment.py {{ID}}
```

Review the new saved `run.json`, logs, and correctness/profiling checks. Compare with a
named baseline run using the same input, workload, scope, and compiler configuration.
