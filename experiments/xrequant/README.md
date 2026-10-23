# XREQUANT: Buffer-Wide Integer Requantization Engine

## Overview and implementation status

XREQUANT converts a contiguous buffer of signed integer values using retained scaling parameters. Software supplies the conversion mode, element count, shift, source and destination addresses, multiplier, and zero points. A blocking RUN instruction loads, converts, and stores the configured elements.

The five-instruction interface and all five conversion modes are implemented in the external modified Spike checkout. Directed, differential, configuration, exception, and memory-fault tests have passed as described below. All five existing requantization functions in the separate `experiments/integer_inference_xtile/` implementation use XREQUANT. The original scalar implementation in `experiments/integer_inference/` remains the software reference.

Spike implements functional behavior, not CV-Wally cycle timing. CV-Wally RTL has not been implemented. A combined XTILE + XREQUANT source package has been prepared from the uploaded source snapshot at `experiments/spike/`. Verifying a fresh reproduction build remains pending.

## Architecture

| Component | Responsibility |
|---|---|
| CPU instruction interface | Decode CFG, SET_IO, SET_MUL, SET_ZP, and RUN |
| Retained setup registers | Hold configuration, addresses, multiplier, zero points, and four setup-valid flags |
| Controller | Traverse elements in increasing index order and coordinate completion or faults |
| Address generator | Derive contiguous source and destination addresses from the element index and mode |
| Input formatter | Load signed INT8, INT32, or INT64 and promote to the required arithmetic width |
| Centering logic | Subtract the input zero point for mode 3 |
| Scaling datapath | Perform the mode-dependent multiplication, rounding, and arithmetic right shifts |
| Output formatter | Check INT32 results or add an output zero point and saturate to INT8 |
| Memory and exception interface | Perform target-memory accesses and report faults while preserving earlier stores |

“Buffer-wide” means one RUN processes the complete configured buffer. It does not require all elements to execute in parallel or to fit simultaneously in an internal buffer. The eventual RTL may reuse an arithmetic unit over multiple cycles; parallelism, buffering, and memory integration remain hardware design choices.

XTILE and XREQUANT have separate retained state. XTILE produces raw matrix accumulators; XREQUANT performs subsequent conversions. Bias handling and block orchestration remain separate.

## Buffer and storage contract

Source and destination are contiguous arrays. Parameters are uniform across one RUN; there is no per-element or per-channel parameter array, explicit stride, or matrix-shape field.

| Mode | Source type | Destination type | Source bytes/element | Destination bytes/element |
|---:|---|---|---:|---:|
| 0 | INT64 | INT32 | 8 | 4 |
| 1 | INT64 | INT8 | 8 | 1 |
| 2 | INT32 | INT8 | 4 | 1 |
| 3 | INT8 | INT32 | 1 | 4 |
| 4 | INT32 | INT8 | 4 | 1 |

For element `i`:

```text
source_address      = source_base      + i * source_bytes
destination_address = destination_base + i * destination_bytes
```

A contiguous matrix can be flattened into one buffer when every element uses the same parameters. For example, a `[32,128]` accumulator matrix is processed with COUNT=4096. Mode 2 reads 16,384 bytes and writes 4,096 bytes.

Software must supply accessible, sufficiently large, non-overlapping buffers. The engine does not detect allocation boundaries or overlapping arrays. In-place conversion is outside the supported software contract. Normal Spike MMU rules apply to accesses, including alignment and access permissions.

## Instruction interface

| Instruction | CPU-register inputs | Behavior |
|---|---|---|
| `xrequant.cfg rs1` | Packed 64-bit configuration | Validate and retain MODE, SHIFT, and COUNT |
| `xrequant.set_io rs1, rs2` | Source and destination base addresses | Retain both full RV64 addresses |
| `xrequant.set_mul rs1` | Signed INT64 multiplier | Retain the multiplier |
| `xrequant.set_zp rs1, rs2` | Input and output zero points | Retain signed INT32 zero points |
| `xrequant.run` | None | Convert and store the complete configured buffer |

The setter/configuration instructions retain register values; they do not access the buffers. SET_ZP interprets the low 32 bits of each source register as signed INT32 and ignores the upper 32 bits.

The diagnostic instruction names describe the interface. Programs currently emit GNU `.insn` encodings; assembler mnemonic support has not been added.

### Retained state

State belongs to each Spike hart and is separate from CPU registers x0–x31. Reset clears the retained fields and marks CFG, IO, MUL, and ZP invalid.

RUN requires all four setup categories, including a supplied multiplier for mode 0 even though that mode does not use multiplication. Explicit zero points must be supplied even when both are zero.

Setters may execute before CFG. Updating one category does not invalidate the others. An invalid CFG leaves the previous configuration intact. Configuration and setup remain retained after successful execution or a fault; RUN does not consume them. Software must ensure the retained values are consistent with the selected mode.

## Configuration register

| Bits | Field | Width | Meaning |
|---|---|---:|---|
| `[63:41]` | RESERVED | 23 | Must be zero |
| `[40:9]` | COUNT | 32 | Number of elements, stored directly |
| `[8:3]` | SHIFT | 6 | Mode-dependent right shift |
| `[2:0]` | MODE | 3 | Conversion mode |

```c
uint64_t cfg =
    (uint64_t)mode |
    ((uint64_t)shift << 3) |
    ((uint64_t)count << 9);
```

Validate arguments before packing; overflowing fields can otherwise alter neighboring fields.

| Constraint | Accepted values |
|---|---|
| MODE | 0–4; modes 5–7 are rejected |
| COUNT | 1–4,294,967,295 |
| SHIFT, modes 0–3 | 1–63 |
| SHIFT, mode 4 | 0–31; this is the second right shift |
| RESERVED | All zero |

The maximum encoded count is an encoding limit, not a promise that the corresponding buffers fit target memory. A zero-length software helper returns before issuing the engine instructions; CFG itself rejects COUNT=0.

## Instruction encoding

All five instructions use a 32-bit R-style encoding in custom-1 opcode space.

| Field | Bits | Value or role |
|---|---|---|
| funct7 | `[31:25]` | Zero |
| rs2 | `[24:20]` | Second source register where used; otherwise x0 |
| rs1 | `[19:15]` | First source register where used; otherwise x0 |
| funct3 | `[14:12]` | Selects the operation |
| rd | `[11:7]` | x0; no CPU-register result |
| opcode | `[6:0]` | `0101011`, hexadecimal `0x2b` |

| Instruction | funct3 | MATCH | MASK |
|---|---:|---|---|
| `xrequant.cfg` | 0 | `0x0000002b` | `0xfff07fff` |
| `xrequant.set_io` | 1 | `0x0000102b` | `0xfe007fff` |
| `xrequant.set_mul` | 2 | `0x0000202b` | `0xfff07fff` |
| `xrequant.set_zp` | 3 | `0x0000302b` | `0xfe007fff` |
| `xrequant.run` | 4 | `0x0000402b` | `0xffffffff` |

Recognition uses `(instruction_word & MASK) == MATCH`. RUN has a fixed complete encoding, `0x0000402b`.

```asm
.insn r 0x2b, 0, 0, x0, rs1, x0    # CFG
.insn r 0x2b, 1, 0, x0, rs1, rs2   # SET_IO
.insn r 0x2b, 2, 0, x0, rs1, x0    # SET_MUL
.insn r 0x2b, 3, 0, x0, rs1, rs2   # SET_ZP
.insn r 0x2b, 4, 0, x0, x0, x0     # RUN
```

Here `rs1` and `rs2` are placeholders for selected CPU register names. The prototype requires RV64 and is recognized by the modified Spike without a separate XREQUANT ISA-string enable flag. Tests use `--isa=rv64imc_zicsr`. XTILE uses custom-0 (`0x0b`), allowing both interfaces in the same simulator.

## Arithmetic semantics

All input loads are signed. Arithmetic uses checked signed INT64 intermediates and arithmetic signed right shifts. Products and rounding additions must fit INT64 before shifting, even when the final shifted result would be small.

Define:

```text
round_shift(v, s) = (v + 2^(s-1)) >> s
clamp8(v)        = min(127, max(-128, v))
```

For modes 0–3, this rounds to nearest with exact ties toward positive infinity: `3/2` becomes 2 and `-3/2` becomes -1. It is not ties-away-from-zero rounding.

### Modes 0–3

Let `x` be the signed input, `m` the multiplier, `s` the configured shift, and `zin`/`zout` the zero points.

| Mode | Operation | Zero-point requirements |
|---:|---|---|
| 0 | `y = round_shift(x, s)`; require INT32 range | `zin=0`, `zout=0`; multiplier ignored |
| 1 | `y = clamp8(round_shift(x*m, s) + zout)` | `zin=0` |
| 2 | `y = clamp8(round_shift(x*m, s) + zout)` | `zin=0` |
| 3 | `y = round_shift((x-zin)*m, s)`; require INT32 range | `zout=0` |

Modes 1 and 2 differ in input element width. Modes 0 and 3 perform no INT32 saturation: an out-of-range converted value raises an overflow exception.

Modes 1–3 accept a signed INT64 multiplier. Mode 3 promotes the INT8 input before subtracting the full signed INT32 input zero point. Modes 1, 2, and 4 accept a full signed INT32 output zero point, added after rounding and before INT8 saturation.

For mode 2 with multiplier 3, shift 3, and output zero point -5:

```text
x =  20: product =  60; rounded = ( 60 + 4) >> 3 =  8; y =   3
x = -20: product = -60; rounded = (-60 + 4) >> 3 = -7; y = -12
```

### Mode 4: two-stage INT32-to-INT8 conversion

Mode 4 preserves the existing two-stage rounding behavior used at the GELU output boundary. It cannot generally be replaced with one multiply and a single combined shift.

Requirements:

- Input zero point is zero.
- Multiplier is positive and no greater than INT32_MAX.
- Configured SHIFT is the second right shift, in 0–31.

For each element:

```c
/* Signed INT64 intermediates; input is signed INT32. */
product = input * multiplier;
high = (product + (INT64_C(1) << 30)) >> 31;

mask = (INT64_C(1) << second_shift) - 1;
remainder = high & mask;
threshold = (mask >> 1) + (high < 0);
rounded = (high >> second_shift) + (remainder > threshold);

output = clamp8(rounded + output_zero_point);
```

Stage 1 uses nearest rounding with ties toward positive infinity. Stage 2 uses nearest rounding with ties away from zero. A second shift of zero leaves `high` unchanged. The existing implementation relies on two's-complement signed values and arithmetic right shift on the GCC targets.

## Execution and fault handling

RUN first validates required setup and mode-specific parameters, then processes elements in increasing index order. Each element is loaded, converted, and stored before processing the next element.

| Condition | Exception cause | `mepc` | `mtval` |
|---|---:|---|---|
| Invalid CFG | 2: illegal instruction | CFG instruction address | CFG instruction bits |
| Missing setup or invalid mode-specific RUN parameters | 2: illegal instruction | RUN instruction address | `0x0000402b` |
| Checked arithmetic overflow or INT32 output-range failure | 25 (`0x19`): XREQUANT overflow | RUN instruction address | `0x0000402b` |
| Tested unmapped source read | 5: load access fault | RUN instruction address | Faulting memory address |
| Tested unmapped destination write | 7: store access fault | RUN instruction address | Faulting memory address |

Other memory exceptions follow normal MMU behavior. Arithmetic checks cover multiplication where applicable, rounding-offset addition, output-zero-point addition where applicable, and INT32 output range where required. INT8 saturation is normal conversion behavior, not an overflow exception.

Earlier successful stores remain visible if a later element faults. There is no rollback, retained progress index, or automatic retry. Treat the complete output as incomplete after a fault. After correcting the cause, rerun from the beginning using valid, non-overlapping buffers.

XREQUANT overflow cause 25 is distinct from XTILE overflow cause 24. A successful RUN retires as one CPU instruction while performing multiple simulated memory accesses. Future hardware can require many cycles and memory transactions.

## Software integration

The accelerated implementation preserves the five existing function interfaces:

| Existing function | Mode |
|---|---:|
| `requantize_shift_i64_to_i32` | 0 |
| `requantize_i64_to_i8` | 1 |
| `requantize_i32_to_i8` | 2 |
| `requantize_i8_to_i32` | 3 |
| `requantize_two_round_i32_to_i8` | 4 |

In `experiments/integer_inference_xtile/src/`, `xrequant.h` supplies instruction wrappers and `requantization.c` configures and runs the engine. The RUN wrapper has a compiler `memory` clobber so accesses are ordered with the surrounding C code as required by the current single-threaded bare-metal use.

The existing conversion boundaries, scale constants, bias handling, and block orchestration are preserved. Attention matrix computation remains scalar; its existing requantization calls now use XREQUANT. GELU arithmetic remains on the CPU, with its requantization calls accelerated. A complete GELU lookup-table replacement is not implemented.

## Verification results

These results were reported from the working modified Spike installation. The current source snapshot was also inspected and packaged. A fresh build from the combined exported package has not yet been demonstrated. Memory-fault and RUN-exception suites below primarily exercise mode 2 and do not establish exhaustive fault coverage for every width and mode.

| Test path below `tests/` | Reported result |
|---|---|
| `i32_to_i8/test_xrequant_cfg_validation.c` | 20 CFG cases, 9 expected traps, 11 RUN checks |
| `i32_to_i8/test_xrequant_i32_to_i8.c` | 2 runs, 24 INT8 outputs, zero mismatches; guards unchanged |
| `i32_to_i8/test_xrequant_i32_to_i8_differential.c` | 220 cases, 16,115 outputs match scalar reference |
| `i64_modes/test_xrequant_i64_modes.c` | 56 cases, 4,102 outputs match scalar references |
| `i8_to_i32/test_xrequant_i8_to_i32.c` | 48 cases, 3,504 outputs match scalar reference |
| `two_round/test_xrequant_two_round.c` | 515 cases, 37,517 outputs; directed rounding and scalar comparisons pass |
| `i32_to_i8/test_xrequant_i32_to_i8_exceptions.c` | 13 cases, 9 expected traps; output and guard checks pass |
| `i32_to_i8/test_xrequant_i32_to_i8_memory_faults.c` | 6 cases, 4 expected memory-access traps; output checks pass |

Configuration validation covers all five supported modes, shift boundaries, rejected modes 5–7, zero count, reserved bits, and retained configuration after rejection. Arithmetic tests include negative values, saturation, wide products, rounding boundaries, guard checks, and retained setup reuse. Exception tests cover missing setup, invalid zero point, checked overflow, partial output, and recovery.

### Rerun existing test binaries

Run from the project repository root in Bash. Set `SPIKE_XTILE` to the full path of the installed simulator containing both engines. The existing variable name is retained for compatibility even though the executable now supports XREQUANT as well.

The commands assume the corresponding ELF files have already been compiled. CFG validation runs first. Only the memory-fault suite uses `-m1` because its addresses assume a 1 MiB Spike RAM configuration.

```bash
(
    set -e
    : "${SPIKE_XTILE:?Set SPIKE_XTILE to the modified Spike executable}"

    for xrequant_test in \
        i32_to_i8/test_xrequant_cfg_validation \
        i32_to_i8/test_xrequant_i32_to_i8 \
        i32_to_i8/test_xrequant_i32_to_i8_differential \
        i64_modes/test_xrequant_i64_modes \
        i8_to_i32/test_xrequant_i8_to_i32 \
        two_round/test_xrequant_two_round \
        i32_to_i8/test_xrequant_i32_to_i8_exceptions \
        i32_to_i8/test_xrequant_i32_to_i8_memory_faults
    do
        xrequant_options=(--isa=rv64imc_zicsr)
        if [ "$xrequant_test" = \
             i32_to_i8/test_xrequant_i32_to_i8_memory_faults ]; then
            xrequant_options+=(-m1)
        fi

        "$SPIKE_XTILE" "${xrequant_options[@]}" \
            "experiments/xrequant/build/${xrequant_test}.elf"
        printf '\nPASS: %s completed.\n' "$xrequant_test"
    done
    printf 'PASS: XREQUANT regression completed.\n'
)
```

## Block-0 instruction-count results

The integrated full-block test passed eight cases with 32,768 final T107 outputs and zero mismatches. The profiling test also passed eight cases with 32,768 outputs checked per path for both complete and separately staged execution.

Average raw `minstret` deltas across eight cases:

| Implementation | Average full-block retired instructions |
|---|---:|
| Earlier recorded scalar baseline | 57,181,472.5 |
| XTILE only | 6,627,485.5 |
| XTILE + XREQUANT | 4,538,779.5 |

Adding XREQUANT saves 2,088,706 retired instructions per block on average: a 31.52% reduction relative to XTILE alone. Combined reduction relative to the earlier scalar baseline is 92.06%.

Case 0 stage comparison:

| Stage | XTILE only | XTILE + XREQUANT | Reduction |
|---|---:|---:|---:|
| LN1 | 256,445 | 154,079 | 39.92% |
| QKV | 135,341 | 202 | 99.85% |
| Attention | 2,922,571 | 2,877,525 | 1.54% |
| Output projection + residual | 155,941 | 74,043 | 52.52% |
| LN2 | 256,384 | 154,018 | 39.93% |
| MLP expansion + GELU | 2,753,098 | 1,213,106 | 55.94% |
| MLP output + residual | 147,750 | 65,850 | 55.43% |

The MLP expansion + GELU stage saves 1,539,992 instructions in case 0. Attention's reduction comes from its requantization calls rather than an XTILE replacement of attention matrix computation.

After both engines, attention accounts for 63.40% and MLP expansion + GELU for 26.73% of case 0's full-block raw instruction count. These identify the next software optimization targets.

Counts are reported without measurement-overhead correction. Counter sanity was `adjacent=1`, `eight_nops=9`; the raw full-block count exceeded the sum of raw stage deltas by nine in every reported case. This difference is not a measurement of all profiling overhead.

These figures measure retired CPU instructions. They do not count the engine's internal memory transactions as separate CPU load/store instructions, and they do not establish CV-Wally cycle speedup, bandwidth requirements, or hardware cost.

## Shared Spike source packaging

The canonical location for the combined simulator package is `experiments/spike/`, alongside `experiments/xtile/` and `experiments/xrequant/`.

| Shared-package item | Purpose |
|---|---|
| `../spike/README.md` | Combined source-export and clean-rebuild instructions |
| `../spike/base_commit.txt` | Original Spike base commit |
| `../spike/xtile_xrequant.patch` | One patch containing both engines and every new instruction header |
| `../spike/modified/riscv/` | Copies of all modified and added Spike sources for both engines |

Base Spike commit:

```text
76ced61f540a55d6d26a23e6dadd02f83ea634ca
```

The original external Spike working checkout remains the development source. Moving the in-project packaging folder does not move that external checkout or change the installed executable.

The supplied combined patch includes all 15 source files, including all ten new instruction headers. Its application reproduces the uploaded source snapshot byte for byte against the original file subset reconstructed from the earlier XTILE patch. A fresh full checkout/build/test remains pending. Use the shared package's `refresh_package.py` after later source changes; renaming the earlier XTILE-only patch would not add XREQUANT, and ordinary `git diff` would omit untracked headers.
