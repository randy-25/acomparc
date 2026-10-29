# Block-0 Q-projection baseline

Historical scalar RV64IMC experiment: 524288 MACs and 4096 INT8 outputs.

## Files

- `qproj_acc_test.c`: kernel, requantization, checks, and counters.
- `qproj_data.h`: frozen inputs, weights, and accumulator reference.
- `qproj_y_ref.h`: expected INT8 outputs.
- `qproj_profile.py`: ELF and Spike trace profiler.
- `experiment.json`: retained experiment metadata.
- [Historical measurements and analysis](results/Result.md).

The C test consumes the frozen headers.
Shared fixture: `fixtures/q_proj_case0.npz`.
Historical reference model: `../../../models/block0_w8a8_baseline.tflite`.

Requantization uses multiplier 1205592262 and right shift 39.
Input, weight, and output zero points are -5, 0, and -17.
The rounding rule is verified for the frozen case.

## Run

From the project root, execute:
`make -C experiments/block0/qproj test`

Build outputs are stored under `build/`.
Historical evidence is preserved under `results/`.

The Makefile retains its historical ISA default.
Spike instruction counts are separate from hardware cycle measurements.
