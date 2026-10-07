# Experiments and results

This file is the main guide for creating experiments, recording runs, and reporting
results. Run the commands below from the project root with the existing virtual
environment active.

## Experiment register

| ID | Question or scope | Status | Evidence |
|---|---|---|---|
| `block0_qproj` | Scalar INT8 Q projection with Q39 requantization (B0) | Verified case-0 baseline; refresh run not yet executed | [Results](results/block0_qproj/README.md), [imported run](results/block0_qproj/runs/20261007-imported-b0/run.json) |
| `generation_kv_cache` | FP32 full-model generation/cache sanity check | Notebook available; not rerun during cleanup | [Notebook](experiments/generation_kv_cache/01_generate.ipynb) |

Original exploratory notebooks are kept in `experiments/archive/` as historical
source. Add new experiments to the register when their folders are created. Mark
new entries as planned until there is execution evidence.

## Experiments, runs, and reports

An **experiment** defines a question, implementation, workload, and correctness
reference. A **run** is one execution of that experiment with a unique ID.

| Location | Purpose |
|---|---|
| `experiments/<experiment>/README.md` | Purpose, implementation, workload, assumptions, and execution instructions |
| `experiments/<experiment>/experiment.json` | Inputs, hashes, source dependencies, commands, and validation checks |
| `experiments/<experiment>/build/runs/<run-id>/` | Build output and the large local trace for that execution |
| `results/<experiment>/runs/<run-id>/` | Saved metadata, snapshots, logs, profile, and configured compact build outputs |
| `results/<experiment>/runs/<run-id>/README.md` | Manually written report for that completed run, if retained |
| `results/<experiment>/README.md` | Latest complete results report, including tables, analysis, and comparisons |
| `results/README.md` | Index linking to each experiment's results README |

The recorder creates the run evidence. It does **not** automatically write or update
these Markdown results reports. Update the parent results README only after reviewing
the completed run, and identify the exact run used in every reported table.

## Starting a new experiment: complete workflow

The K-projection commands below illustrate the workflow. Creating its folder does
not implement or validate a K-projection kernel.

### 1. Choose an ID and comparison reference

Choose a descriptive experiment ID and write down its question and scope.

- `block0_kproj`: a different projection kernel.
- `block0_qproj_b1`: a software optimization of the Q-projection baseline.
- `block0_qproj_x1`: a Q-projection variant using a custom instruction.

Keep the B0 experiment and its saved runs as comparison references. Use a separate
experiment ID for an implementation or ISA variant that should be compared with B0.
Use a new run ID when repeating the same experiment.

### 2. Create the experiment folder

```bash
python3 scripts/new_experiment.py block0_kproj \
  --purpose "Port and verify block-0 K projection"
```

This creates `experiments/block0_kproj/README.md` and `experiment.json`, using
`docs/experiment_template/`. It refuses an existing experiment folder.

Creation does not copy q-projection dimensions, constants, inputs, results, or a
success status. The new configuration has empty inputs, workload, sources, and
commands; fill these in before running.

### 3. Define and save the workload and reference data

Inspect the selected model/export and identify the actual kernel's operation and
tensors. Record the model identity, tensor IDs, shapes, dtypes, scales, zero points,
bias handling, accumulator width, and output conversion rule.

Save the input, weights, and expected outputs under distinct paths in
`artifacts/exports/`. Keep the selected model under `artifacts/models/`. Record
SHA256 values for all configured input files. For example, after replacing these
example filenames with the real paths:

```bash
sha256sum artifacts/models/your_model.tflite artifacts/exports/your_case.npz
```

For a dense projection with batch B, token count T, input features K, and output
features N, the expected workload is B × T × N × K MACs and B × T × N output elements.
Check the actual tensor layout before using that formula.

If calibration or quantization changes, save a distinct artifact and identify it
in the experiment. Do not overwrite the frozen B0 model, data, or reference headers.

### 4. Add the implementation, build rules, and correctness checks

Add the kernel source, generated data/reference headers, Makefile, and profiling
helper under `experiments/block0_kproj/`.

The test must compare the implementation with the chosen reference and return a
nonzero exit code on a mismatch. Define what is checked: accumulator values, final
quantized outputs, or both. Host and Spike tests should use the same fixed case.

For the existing Makefile workflow, support an externally supplied `BUILD_DIR`,
plus the configured test/disassembly/trace targets. This lets the recorder build
into a fresh run directory.

The q-projection profiler detects a particular scalar loop. Adapt or replace it
when function names, loop structure, instruction semantics, or kernel type change.
Do not assume it correctly profiles every new experiment.

### 5. Complete the README and experiment configuration

Fill the experiment README with its question, workload, arithmetic, reference,
limitations, and named comparison run. Use `experiments/block0_qproj/experiment.json`
as an example of configuration structure, then supply the new experiment's values.

| Configuration field | What to supply |
|---|---|
| `id`, `purpose`, `variant`, `status` | Experiment identity and current state; use `planned` before validation |
| `workload` | Actual dimensions, MAC/output counts, and measured function |
| `inputs` | Project-root-relative input paths and their SHA256 values |
| `source_files` | Project-root-relative paths for every source, header, build rule, and helper used |
| `commands` | Argument arrays for tests, disassembly/trace generation, and profiling |
| `profile_filename` | Profile JSON filename relative to the saved run directory, or `null` when unused |
| `build_outputs` | Named output files relative to the run's build directory; include the role `trace` for the large trace |
| `required_profile_checks` | Boolean profile fields that must be `true` for a successful run |
| `capture_make_settings` | Enable when the Makefile exposes a working `print-config` target |
| `capture_cvw_runtime` | Enable for the CV-Wally runtime workflow with resolved `CVW_ROOT` and `CVW_COMMON` settings |

Commands execute inside `experiments/<experiment>/`. The recorder expands these
literal placeholders in command arguments:

- `{python}`: the Python executable running the recorder.
- `{build_dir}`: the unique absolute build directory.
- `{run_dir}`: the unique absolute saved-results directory.
- `{riscv_prefix}`: the configured RISC-V tool prefix.

Use `{build_dir}` for the Makefile's `BUILD_DIR` and `{run_dir}` for saved profile
output. Include expected workload checks supported by the new profiler. The B0
checks `scalar_macs_match_expected` and `requantizations_match_expected_outputs`
apply only when those fields and meanings are supported.

### 6. Check the planned commands

Once the source, data, build rules, and configuration are ready:

```bash
python3 scripts/run_experiment.py block0_kproj \
  --run-id kproj-b0-01 \
  --dry-run
```

Check the experiment ID, Python/toolchain, build paths, ELF/trace paths, expected
counts, and saved profile path. A dry run prints command expansion; it does not
execute tests, verify input hashes, or establish correctness.

### 7. Execute a fresh recorded run

Remove `--dry-run` while keeping the same unused run ID:

```bash
python3 scripts/run_experiment.py block0_kproj \
  --run-id kproj-b0-01
```

For another CV-Wally location, add:

```bash
--make-var CVW_ROOT=/path/to/cvw
```

The existing Makefile defaults to `$HOME/Documents/ACompArc/cvw`. `BUILD_DIR` is
reserved by the recorder and cannot be overridden through `--make-var`.

The recorder saves configured source/input snapshots, command logs, environment
information, and output hashes. It checks configured input hashes, changes during
execution, required profile checks, and the profiled ELF hash when supplied.

The large trace remains in the unique build directory and its hash is recorded.
Add `--keep-trace` to copy the raw trace into the saved run; raw logs remain ignored
by Git.

The recorder refuses an existing run ID. For another execution, choose a new ID
such as `kproj-b0-02`.

### 8. Verify the saved evidence

```bash
python3 scripts/list_runs.py
python3 -m json.tool results/block0_kproj/runs/kproj-b0-01/run.json
```

Inspect the command logs and profile named in the configuration. Confirm:

- The recorded status is `passed`, with successful command return codes.
- Correctness checks passed against the stated reference.
- Observed MAC/output or other workload checks match the expected case.
- The profile identifies the intended function and the correct ELF.
- Input/source hashes, flags, measurement scope, and comparison reference are recorded.

Failed or interrupted runs keep their logs and status. Diagnose the failure and use
a new run ID for the next attempt. Do not describe a partial or failed run as a
successful result. A `passed` status means the configured commands/checks passed;
its strength depends on the correctness checks implemented for that experiment.

Counter output such as `mcycle` and `minstret` is retained in command logs when the
test prints it. Read counters from this same run; do not substitute older values.

### 9. Write the completed results report

After all intended checks and measurements are complete, write the full report at:

`results/block0_kproj/README.md`

Include the following in order:

1. Experiment scope, variant, selected run ID, and execution date from `run.json`.
2. Links to that run's metadata, logs, profile, ELF/disassembly when available, and source snapshot.
3. Workload, quantization/arithmetic assumptions, and correctness results.
4. Measurement tables: instruction breakdown, instruction mix, counters where available, and sizes.
5. Derived metrics with their formulas and measurement scope.
6. Analysis, named baseline comparison, and limitations.
7. Reproduction commands for the same configuration using a fresh run ID.

This parent README contains the latest **complete report**, not just a link to a
profile. Keep the old run directories. Retain a run-specific README alongside a
completed run's evidence when preserving its written analysis for later comparison.

Links in the parent report use paths such as `runs/kproj-b0-01/profile.json`. Links
in a run-specific README use `profile.json`. Adjust relative links for each location.

Do not invent instruction subregion counts or isolated operation cycles that were
not measured. Derive a finer breakdown only from that run's disassembly/trace and
state how it was calculated.

### 10. Update the indexes

Add or update the experiment's row in this file and in `results/README.md`. Point
the results index to `results/<experiment>/README.md` and describe its actual status.
A new experiment's success does not replace or invalidate the recorded B0 reference.

## Refreshing an existing experiment

For `block0_qproj`, the implementation and configuration already exist. Skip new
experiment creation and start with a fresh run ID:

```bash
python3 scripts/run_experiment.py block0_qproj \
  --run-id b0-refresh-01 --dry-run

python3 scripts/run_experiment.py block0_qproj \
  --run-id b0-refresh-01

python3 scripts/list_runs.py
```

The dry run for `b0-refresh-01` has been reviewed. Actual execution and result
verification are still pending. After it completes, follow steps 8–10 and update
`results/block0_qproj/README.md` with that run's complete results. Keep
`20261007-imported-b0` as the earlier reference.

`make run`, `make dry-run`, and `make runs` are project-root shortcuts. For a specific
experiment and run ID, use `make run EXP=block0_qproj RUN_ID=b0-refresh-01`.
Running `make test` inside an experiment produces local build output without a saved
run record.

## What a saved run contains

`results/<experiment>/runs/<run-id>/run.json` records origin/status, UTC times, Git
commit and dirty state, configuration, input/source hashes, commands and return codes,
tool metadata, captured external runtime information, output hashes, and key metrics.
`snapshot/` contains the listed source/input bytes and captured external runtime files.
For profiling experiments, the configured profile describes the selected measurement
scope. Configured ELF/disassembly outputs are saved as compact evidence.

The runner does not modify kernel source, commit, or push. Tests must return nonzero
on failures; recording a run does not create a reference implementation for a new kernel.

## Notebook runs

Each active notebook snapshots its file/shared helpers and creates a unique record in
`results/notebook_runs/`. A final cell marks it completed and records output hashes.
Completion does not mean quantization errors meet an accuracy target. Interrupted
notebooks remain `started`, because a live Jupyter kernel cannot reliably catch every
UI interruption. Restart the kernel for another independent execution.

Notebook 02 writes reexports with unique paths. Select a reexport explicitly in
`src/python/workflow_config.py` for later analysis and retain its manifest; keep the
frozen baseline intact. Notebook 04 creates an accumulator-only test with its own
output path, while the measured requantization kernel remains in its experiment folder.

## Comparison requirements

Compare matching input data, MAC/output counts, measurement scope, flags, correctness
rules, and counter sources. State deliberately changed conditions. Spike instruction
counts do not establish CV-Wally cycle speedup. Keep B0, B1, X1, and X2 separate and
document dependencies between variants.

## What to keep in Git

Keep experiment descriptions/configuration, source, small frozen inputs/weights,
artifact identities, compact run snapshots/reports, and environment records. Ignore
build folders, long traces, waveforms, virtual environments, and caches. Raw traces
can be archived separately with their recorded hashes.

Review changes before committing. Git history and source snapshots identify a run
even when uncommitted edits were intentionally tested. Publishing to GitHub is a
separate step; the workflow does not push automatically.
