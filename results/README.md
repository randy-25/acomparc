# Experiment Results

This folder contains the results of each experiment. Open an experiment's results README for its measurements, tables, analysis, and supporting evidence.

## Experiment index

| Experiment | Description | Results |
|---|---|---|
| `block0_qproj` | Block-0 q-projection: scalar RV64IMC baseline (B0) | [Results and analysis](block0_qproj/Result.md) |

Add a row to this table when another experiment has results to report.

## Saved runs

Each experiment keeps individual run records in:

`<experiment>/runs/<run-id>/`

The experiment's results README identifies the runs used in its tables and comparisons. Each saved run includes metadata, source/input snapshots, logs, and available profiling evidence.

To list saved command-line and imported runs, execute from the project root:

```bash
python3 scripts/list_runs.py