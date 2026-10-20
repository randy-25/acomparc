# Reproducing the scoped replay

The scripts are an analysis harness for the exact supplied ELF, not changes to the inference project. Use a separate temporary working directory.

1. Create `analysis_work/` in that directory and copy `prepare_replay.py` and `replay.cpp` into it.
2. Extract the original `latest.zip` so the supplied experiment tree is at `analysis_input/latest/`. For example, that directory must contain `build/block0/profile/test_block0_profile.elf` and `.dis`, plus `src/`, `params/`, and `fixtures/`.
3. From the temporary working directory, run:

```bash
python3 analysis_work/prepare_replay.py
g++ -std=c++17 -O3 -Wall -Wextra analysis_work/replay.cpp -o analysis_work/replay
./analysis_work/replay
```

The preparation step checks that each decoded inference instruction matches the ELF bytes. The replay writes `analysis_work/case0.json` through `case7.json` and prints the full-body instruction counts and final-output mismatch counts. All eight mismatch counts must be zero. The expected full-body counts are in `counts.json`; add six for the supplied full raw `minstret` measurement. Per-stage bodies are also recorded there.

The replay starts directly at `block0_forward()` and checks the supplied ELF-embedded goldens. It relies on fixed fixture, workspace-table, and function addresses specific to the SHA-256 identified binary. Do not apply those addresses or expected counts to a newly compiled ELF. It models only the instruction subset used by this inference region and does not model cycles, startup, CSRs, traps, devices, or Spike itself.

`helper_counts.csv` lists wrapper-exclusive and helper-inclusive counts in call order. `instruction_frequencies.csv` lists complete function-body execution counts per instruction site, and `stage_instruction_mix.csv` splits opcode counts by stage. The 48 instructions of `block0_forward()` itself are present in the first CSV and excluded from the per-stage mixes.
