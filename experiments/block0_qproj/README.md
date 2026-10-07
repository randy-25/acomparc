# Block-0 q-projection baseline (B0)

Question: how many scalar RV64IMC instructions does the verified INT8 projection need?

The fixed workload is 32 tokens × 128 outputs × 128 reduction elements = 524288 MACs. Inputs and weights are signed INT8, input zero point -5, weight zero point 0. The INT32 sum is requantized using Q39 multiplier 1205592262, shift 39, output zero point -17, then saturated to INT8. There is no bias. The C kernel and both headers are preserved byte-for-byte from the supplied project and match its Git HEAD source.

## Frozen files

- `qproj_acc_test.c`: measured kernel, requantization, INT8 correctness check, counters.
- `qproj_data.h`: input, weights, accumulator reference.
- `qproj_y_ref.h`: 4096 expected INT8 outputs.
- `qproj_profile.py`: ELF-based trace analysis; source-specific scalar-loop detector.
- `experiment.json`: workload, input hashes, source list, commands, output names.
- Shared data: `../../artifacts/exports/q_proj_case0.npz`.
- Matching model: `../../artifacts/models/block0_w8a8_baseline.tflite`.

The C test uses the headers, not the NPZ or TFLite file at runtime. The model/NPZ record
their offline reference origin. The fixed-point rounding rule matches this golden
case; general equivalence to LiteRT for every accumulator has not been established.

## Record the next baseline run

Use the installed toolchain and your existing virtual environment. Run the following
commands from the project root. If you are currently in `experiments/block0_qproj/`,
first run `cd ../..` to return to the project root.

```bash
python3 scripts/run_experiment.py block0_qproj --dry-run
python3 scripts/run_experiment.py block0_qproj
python3 scripts/list_runs.py
```

The dry run prints the planned commands without executing them. The recorder then runs host and Spike tests, creates fresh disassembly and trace, and profiles that exact ELF. Each recorded run saves its source/input snapshot, command logs, profile, and run metadata under `results/block0_qproj/runs/<run-id>/`.

Expected host and Spike output: 4096 INT8 outputs, zero mismatches, PASS. Fresh profiling must also match 524288 scalar MACs and 4096 requantization multiplies.

The large trace stays in the run's build directory, while its hash is saved with the result. Add `--keep-trace` to archive the raw trace inside the saved run; raw logs remain ignored by Git.

The Makefile defaults CV-Wally to `$HOME/Documents/ACompArc/cvw`. For another location:

```bash
python3 scripts/run_experiment.py block0_qproj --make-var CVW_ROOT=/path/to/cvw
```

From the project root, use `make run`, `make dry-run`, and `make runs` as shortcuts. Running `make test` inside `experiments/block0_qproj/` still works, but creates disposable build output without a saved run record.

## Existing build and results

`build/qproj/` preserves the supplied current scratch build and trace. `build/legacy/` preserves earlier output. New recorded runs use `build/runs/<run-id>/` so those files stay intact.

The Makefile still supports `test`, `trace`, `disasm`, `inspect`, and `rtl`. Its default `rv64gc` RTL configuration is an existing smoke-test configuration, not the final RV64IMC hardware baseline. Configure and evaluate the intended RTL separately.