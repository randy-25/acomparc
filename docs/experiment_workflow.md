# Working with experiments and runs

1. Choose an experiment ID and state its question in the experiment README.
2. Identify model/data by path and SHA256; use a new artifact identity if calibration changes.
3. Record workload, source files, compiler/runtime settings, and expected correctness checks.
4. Review command expansion with `--dry-run`, then execute a fresh recorded run.
5. Inspect logs and validation results, including failures. Commit the small evidence.
6. Name the baseline run when analyzing another variant and record what changed.

## A run record

`results/<experiment>/runs/<run-id>/run.json` contains origin/status, UTC times, Git
commit and dirty state, configuration, input/source hashes, commands and return codes,
tool metadata, external CV-Wally runtime snapshots, output hashes, and key metrics.
`snapshot/` holds the exact listed source and input bytes. `profile.json` describes
the selected instruction scope. The ELF and disassembly are retained as compact
evidence; the large trace is hashed and retained in the unique local build directory.

The recorder refuses an existing run ID. It marks failed/interrupted runs and preserves
their logs instead of labeling partial execution successful. Changed configured input
hashes or changed files during execution invalidate the run. List every source dependency
in the configuration. Tests must return nonzero on failures; run recording does not
create a correctness oracle for a new kernel.

The runner does not mutate kernel source, commit, or push. The config commands run in the
experiment directory. Future kernels can use different command arrays and build output
names; replace the q-projection profiler when its scalar-loop assumptions no longer apply.

## Notebook runs

Each active notebook snapshots its file/shared helpers and creates a unique record in
`results/notebook_runs/`. A final cell marks it completed and records output hashes.
Completion does not mean its quantization errors meet an accuracy target. Interrupted
notebooks remain `started`, because a live Jupyter kernel cannot reliably catch every
UI interruption. Restart the kernel for another independent execution.

Notebook 02 writes reexports with unique paths. Select a reexport explicitly in
`workflow_config.py` for later analysis and retain its manifest; never overwrite the
frozen baseline. Notebook 04 creates an accumulator-only test with its own output path,
while the measured requantization kernel remains in its experiment directory.

## What to commit

Commit descriptions, configurations, source, small frozen inputs/weights, artifact
identities, compact run snapshots/reports, and environment records. Ignore build
folders, long traces, waveforms, virtual environments, and caches. Avoid committing
machine credentials or local `.env` files. Raw traces can be archived separately with
their recorded hash if needed for review. Git history plus source snapshots identifies
a run even when uncommitted edits were intentionally tested.

## Comparison requirements

Compare matching input data, MAC/output counts, scope, flags, correctness rules, and
counter sources. Spike instruction counts cannot establish CV-Wally cycle speedup.
Keep B0, B1, X1, and X2 separate and document dependencies between variants.
