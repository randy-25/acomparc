# Profiling parser fix and measured findings

The comparison script failed because it skipped the whole-chain profile record. Both uploaded logs contain the CSV header joined directly to that record:

```text
PROFILE_HEADER,backend,case,block,region,raw_minstretPROFILE,scalar,0,-1,chain,461087565
PROFILE_HEADER,backend,case,block,region,raw_minstretPROFILE,custom,0,-1,chain,36302539
```

The compiler converted the literal header `printf` into a call to `puts`. In the uploaded scalar disassembly, the call is at `0x8000538a`, and `puts` starts at `0x80005012`. This bare-metal implementation prints the string without appending a newline. The original parser accepted only lines starting with `PROFILE,`, so it missed the joined chain row. The stage and block measurements remain present.

## Changes and application

The updated Python parser removes only the exact known header prefix before parsing a record. It retains checks for matching backends, positive counts, duplicate rows, complete regions and correct stage/block sums. It also checks the declared completed case count and reports missing/unexpected region names. Report percentages use four decimal places so the QKV reduction remains visibly below 100%.

The C benchmark and measured binaries are unchanged. No Spike rebuild or new profiling run is needed for these logs.

From the project root, extract the supplied archive (adjust its download location):

```bash
cd "$HOME/Documents/ACompArc/model/project"
```

```bash
tar -xzf "$HOME/Downloads/tinystories-profile-parser-fix.tar.gz"
```

Run the Python regression tests:

```bash
python -m unittest discover \
    -s experiments/integer_inference_allblocks/tests/full \
    -p test_analysis.py
```

Rerun the comparison with the existing files:

```bash
python experiments/integer_inference_allblocks/python/analyze_instruction_comparison.py \
    --scalar-dis experiments/integer_inference_allblocks/build/tests/full/scalar/profile1/test_allblocks.dis \
    --custom-dis experiments/integer_inference_allblocks/build/tests/full/custom/profile1/test_allblocks.dis \
    --scalar-log experiments/integer_inference_allblocks/build/tests/full/scalar/profile1/profile.log \
    --custom-log experiments/integer_inference_allblocks/build/tests/full/custom/profile1/profile.log \
    --out-dir experiments/integer_inference_allblocks/build/instruction_comparison
```

If that output already exists from a successful analysis, append `--overwrite`. A continuation backslash must be the last character on its line, without a trailing space.

## Main result

For the single benchmark case across blocks 0–7, the whole-chain raw count falls from **461,087,565** to **36,302,539** retired instructions: **92.1268% fewer**, or a **12.701× scalar/custom instruction ratio**. Both uploaded profile logs report successful chain, block and stage output checks. These files establish one measured benchmark case; they do not establish an eight-case profiling run.

No counter-overhead correction is applied. Adjacent reads report 1 instruction and the eight-NOP sanity check reports 9 in each log. Chain, full-block and individual-stage intervals are separate measurements, so their sums need not be identical.

The full stage table and exact per-block values are in `instruction_comparison_benchmark/report.md` and `dynamic_regions.csv`.

## What the assembly shows

The scalar disassembly contains no XTILE/XREQUANT opcodes. The custom disassembly contains five XTILE instruction sites (one for each operation) and 25 XREQUANT sites (five operations across five buffer wrappers). These are static assembly sites, not dynamic execution counts.

The scalar dense accumulation inner loop is:

```asm
80000a86: lb    t3,0(a0)
80000a8a: lb    t1,0(t4)
80000a8e: addi  a0,a0,1
80000a90: subw  t3,t3,a7
80000a94: mulw  t1,t1,t3
80000a98: addi  t4,t4,1
80000a9a: addw  t5,t1,t5
80000a9e: bne   t6,a0,80000a86
```

This compiled loop retires eight instructions per multiply-accumulate, including loads, input-zero-point subtraction, pointer advances and the loop branch. From the fixed matrix shapes, six dense projections perform 6,291,456 MACs per block, or 50,331,648 across eight blocks. The loop therefore accounts for 402,653,184 instructions, approximately 87.33% of the scalar whole-chain count. This is a calculation from the observed loop and dimensions, not an instruction-category trace.

The custom dense path instead contains these encoded instructions (names annotated from opcode/funct3):

```asm
80000a78: 0007800b  xtile_cfg
80000a7c: 00b5100b  xtile_set_ab
80000a80: 0006a00b  xtile_set_c
80000a84: 01c8b00b  xtile_set_zp
80000a88: 0000400b  xtile_run
```

Configuration preparation and software bias handling still surround these sites. One RUN performs the configured matrix operation inside Spike; internal tile work is not retired as separate scalar instructions. Similarly, XREQUANT replaces per-element multiplication, rounding, shifting, zero-point addition, clamping and stores with buffer operations. The static wrappers are shown in the supplied `scalar_kernels.dis` and `custom_kernels.dis`.

This explains why QKV falls from 102,736,208 to 1,636 retired instructions. It does not imply that the hardware completes those matrix operations in 1,636 cycles.

## Remaining work indicated by the profile

Attention contributes **63.3868%** of custom stage intervals; MLP expansion plus GELU contributes **26.7336%**. Together they account for **90.1204%**. Dense projections and buffer requantization have shifted the software bottleneck toward those stages.

The MLP/GELU stage still includes both expansion and activation processing. A substage measurement would be required to quantify GELU alone. The modest attention reduction is consistent with buffer requantization at its output while attention computation remains scalar.

The current result measures the combined XTILE/XREQUANT backend. It does not separately measure each extension's contribution, physical accelerator latency, memory-system costs or RTL speedup.

## Validation of this fix

Six local analyzer tests passed, covering encoded custom operations, compressed instructions, normal/joined headers, completeness and aggregate checks, missing/extra/duplicate records, declared case counts and CLI output generation. Test fixtures are explicitly synthetic and are not used for the reported performance numbers.

The corrected analyzer was also run against the actual uploaded logs/disassemblies. Their SHA-256 digests and the supplied source-file digests are recorded in `instruction_comparison_benchmark/input_provenance.json`.
