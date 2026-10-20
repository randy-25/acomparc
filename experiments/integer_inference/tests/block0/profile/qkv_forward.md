# QKV Forward: Disassembly Analysis

## 1. Function Overview

`block0_qkv_forward()` computes the Q, K, and V projections for block 0.

```c
void block0_qkv_forward(
    const int8_t *input,
    int8_t *q_output,
    int8_t *k_output,
    int8_t *v_output,
    int32_t *accumulator
);
```

The input is an INT8 tensor containing 32 tokens with 128 features per token. Each projection produces an INT8 tensor of the same shape.

The execution order is:

1. Q accumulation into the shared INT32 accumulator.
2. Q requantization into the Q output.
3. K accumulation into the shared INT32 accumulator.
4. K requantization into the K output.
5. V accumulation into the shared INT32 accumulator.
6. V requantization into the V output.

Accumulation and requantization are implemented by separate shared functions:

- `projection_accumulate_i8()`
- `requantize_i32_to_i8()`

The accumulator storage is reused because each projection is requantized before the next accumulation starts.

## 2. Wrapper: `block0_qkv_forward()`

### 2.1 Function Entry and Q Accumulation Preparation

#### Assembly

```asm
0000000080000c86 <block0_qkv_forward>:
    80000c86: 7179        addi  sp,sp,-48
    80000c88: 08000813    li    a6,128
    80000c8c: f022        sd    s0,32(sp)
    80000c8e: e84a        sd    s2,16(sp)
    80000c90: e44e        sd    s3,8(sp)
    80000c92: e052        sd    s4,0(sp)
    80000c94: 843a        mv    s0,a4
    80000c96: 87c2        mv    a5,a6
    80000c98: 8a2e        mv    s4,a1
    80000c9a: 89b2        mv    s3,a2
    80000c9c: 8936        mv    s2,a3
    80000c9e: 58bd        li    a7,-17
    80000ca0: 86ba        mv    a3,a4
    80000ca2: 4601        li    a2,0
    80000ca4: 02000713    li    a4,32
    80000ca8: 00032597    auipc a1,0x32
    80000cac: 6f058593    addi  a1,a1,1776 # 80033398 <block0_q_weights>
    80000cb0: f406        sd    ra,40(sp)
    80000cb2: ec26        sd    s1,24(sp)
    80000cb4: 84aa        mv    s1,a0
    80000cb6: e27ff0ef    jal   80000adc <projection_accumulate_i8>
```

#### Incoming Arguments

At entry to `block0_qkv_forward()`, the argument registers contain:

| Register | Argument |
|---|---|
| `a0` | Input activation pointer |
| `a1` | Q output pointer |
| `a2` | K output pointer |
| `a3` | V output pointer |
| `a4` | Shared INT32 accumulator pointer |

These registers are subsequently rearranged to prepare the arguments for `projection_accumulate_i8()`.

#### Stack Allocation and Register Preservation

```asm
addi sp,sp,-48
```

This reserves 48 bytes on the stack.

Six registers are saved using `sd`, which stores eight bytes per instruction:

| Saved register | Stack location |
|---|---|
| `s4` | `0(sp)` |
| `s3` | `8(sp)` |
| `s2` | `16(sp)` |
| `s1` | `24(sp)` |
| `s0` | `32(sp)` |
| `ra` | `40(sp)` |

The required storage is:

$$6 \times 8 = 48\text{ bytes}$$

The caller's original `s0–s4` values must be preserved because this function reuses those registers. The original return address in `ra` must also be preserved because subsequent function calls overwrite it.

These stack slots hold saved register values. They do not hold the activation or projection tensors.

#### Retaining Workspace Pointers

The following instructions retain the five pointers across helper calls:

```asm
mv s0,a4
mv s4,a1
mv s3,a2
mv s2,a3
mv s1,a0
```

| Register | Pointer retained |
|---|---|
| `s0` | Shared INT32 accumulator |
| `s1` | Input activation |
| `s2` | V output |
| `s3` | K output |
| `s4` | Q output |

Argument registers may be overwritten by the called functions. The saved registers allow the wrapper to retain these pointers throughout QKV execution. Each `mv` copies a pointer value.

#### Preparing Q Accumulation Arguments

Immediately before the `jal`, the registers contain:

| Register | Projection argument | Value |
|---|---|---|
| `a0` | Input pointer | Input activation |
| `a1` | Weight pointer | `block0_q_weights` |
| `a2` | Bias pointer | `NULL` |
| `a3` | Accumulator pointer | Shared INT32 scratch |
| `a4` | Token count | 32 |
| `a5` | Input-feature count | 128 |
| `a6` | Output-feature count | 128 |
| `a7` | Input zero point | −17 |

The dimensions and zero point are prepared by:

```asm
li a6,128
mv a5,a6
li a7,-17
mv a3,a4
li a2,0
li a4,32
```

The accumulator pointer is copied from `a4` into `a3` before `a4` is overwritten with the token count.

The Q weight address is constructed by:

```asm
auipc a1,0x32
addi  a1,a1,1776
```

Together, these produce address `0x80033398`, corresponding to `block0_q_weights`. They construct the pointer; they do not load weight elements.

The input pointer already occupies `a0`, so no additional instruction is needed to prepare that argument.

#### Calling the Projection Kernel

```asm
80000cb6: jal 80000adc <projection_accumulate_i8>
```

This transfers execution to the projection kernel and places the return address, `0x80000cba`, in `ra`.

The kernel computes:

$$
A[t,o] =
\sum_{i=0}^{127}
\left(X[t,i]-Z_X\right)W_Q[o,i],
\qquad Z_X=-17.
$$

There is no bias contribution because the bias pointer is `NULL`.

The result is stored in the shared INT32 accumulator. Q requantization will subsequently convert this result into the INT8 Q output.

#### Instruction Accounting

This section contains no loops or conditional branches. Each listed instruction executes once per invocation of `block0_qkv_forward()`.

| Purpose | Instructions | Executed count |
|---|---|---:|
| Reserve stack storage | `addi sp,sp,-48` | 1 |
| Save registers | Six `sd` instructions | 6 |
| Copy pointers and arguments | Seven `mv` instructions | 7 |
| Load constant arguments | Four `li` instructions | 4 |
| Construct Q weight address | `auipc` + `addi` | 2 |
| Call Q accumulation | `jal` | 1 |
| **Total** | | **21** |

There are **20 preparation instructions plus one call instruction**.

These instructions belong to the QKV wrapper. Instructions executed inside `projection_accumulate_i8()` are counted separately.

### 2.2 Q Requantization Preparation

After Q accumulation finishes, execution returns to `0x80000cba`.

The shared accumulator now contains 4,096 INT32 values. These are converted into INT8 values and stored in the Q output buffer.

#### Assembly

```asm
    80000cba: 4c59b6b7    lui  a3,0x4c59b
    80000cbe: 85d2        mv   a1,s4
    80000cc0: 8522        mv   a0,s0
    80000cc2: a3568693    addi a3,a3,-1483
    80000cc6: 57d1        li   a5,-12
    80000cc8: 02700713    li   a4,39
    80000ccc: 6605        lui  a2,0x1
    80000cce: d39ff0ef    jal  80000a06 <requantize_i32_to_i8>
```

#### Corresponding C

```c
requantize_i32_to_i8(
    accumulator, q_output, elements,
    BLOCK0_Q_MULTIPLIER,
    BLOCK0_QKV_REQUANT_SHIFT,
    BLOCK0_Q_OUTPUT_ZERO_POINT
);
```

#### Argument Preparation

| Register | Argument | Value |
|---|---|---|
| `a0` | INT32 source pointer | Accumulator pointer retained in `s0` |
| `a1` | INT8 destination pointer | Q output pointer retained in `s4` |
| `a2` | Element count | 4,096 |
| `a3` | Integer multiplier | `0x4c59aa35` |
| `a4` | Right shift | 39 |
| `a5` | Output zero point | −12 |

The multiplier is constructed using two instructions:

```asm
lui  a3,0x4c59b       # 0x4c59b000
addi a3,a3,-1483      # 0x4c59aa35
```

The element count is loaded by:

```asm
lui a2,0x1           # 0x1000 = 4096
```

The pointer copies do not move tensor data. They place the existing buffer addresses into argument registers.

#### Control Flow and Instruction Count

There are seven argument-preparation instructions and one call instruction:

| Component | Executed instructions |
|---|---:|
| Argument preparation | 7 |
| Call to requantization | 1 |
| **Wrapper subtotal** | **8** |

Instructions executed inside `requantize_i32_to_i8()` are counted separately.

The call sets the return address to `0x80000cd2`. After requantization returns:

- The INT8 Q output is complete.
- The accumulator storage can be reused for K accumulation.
- Execution continues with K accumulation preparation.

### 2.3 K Accumulation and Requantization Preparation

After Q requantization returns, execution resumes at `0x80000cd2`.

The Q output is complete, so the shared INT32 accumulator can now be reused for K.

#### K Accumulation Assembly

```asm
    80000cd2: 08000813    li    a6,128
    80000cd6: 87c2        mv    a5,a6
    80000cd8: 86a2        mv    a3,s0
    80000cda: 8526        mv    a0,s1
    80000cdc: 58bd        li    a7,-17
    80000cde: 02000713    li    a4,32
    80000ce2: 4601        li    a2,0
    80000ce4: 0002e597    auipc a1,0x2e
    80000ce8: 6b458593    addi  a1,a1,1716 # 8002f398 <block0_k_weights>
    80000cec: df1ff0ef    jal   80000adc <projection_accumulate_i8>
```

#### Argument Mapping

| Register | Argument | Value |
|---|---|---|
| `a0` | Input pointer | Activation pointer retained in `s1` |
| `a1` | Weight pointer | `block0_k_weights` |
| `a2` | Bias pointer | `NULL` |
| `a3` | Accumulator pointer | Shared scratch pointer retained in `s0` |
| `a4` | Token count | 32 |
| `a5` | Input-feature count | 128 |
| `a6` | Output-feature count | 128 |
| `a7` | Input zero point | −17 |

K uses the same accumulation kernel and dimensions as Q, with a different weight pointer.

The call returns to `0x80000cf0`.

#### K Requantization Assembly

```asm
    80000cf0: 68b6f6b7    lui  a3,0x68b6f
    80000cf4: 85ce        mv   a1,s3
    80000cf6: 8522        mv   a0,s0
    80000cf8: 06b1        addi a3,a3,12
    80000cfa: 5789        li   a5,-30
    80000cfc: 02700713    li   a4,39
    80000d00: 6605        lui  a2,0x1
    80000d02: d05ff0ef    jal  80000a06 <requantize_i32_to_i8>
```

#### Argument Mapping

| Register | Argument | Value |
|---|---|---|
| `a0` | INT32 source pointer | Shared accumulator |
| `a1` | INT8 destination pointer | K output pointer retained in `s3` |
| `a2` | Element count | 4,096 |
| `a3` | Multiplier | `0x68b6f00c` |
| `a4` | Right shift | 39 |
| `a5` | Output zero point | −30 |

The call uses the shared requantization kernel described in Section 4.

After it returns to `0x80000d06`, the INT8 K output is complete.

#### Wrapper Instruction Counts

| Component | Executed instructions |
|---|---:|
| K accumulation preparation | 9 |
| K accumulation call | 1 |
| K requantization preparation | 7 |
| K requantization call | 1 |
| **Subtotal** | **18** |

These counts exclude instructions executed inside the two kernels.

### 2.4 V Accumulation, Register Restoration, and Tail Call

#### V Accumulation Assembly

```asm
    80000d06: 08000813    li    a6,128
    80000d0a: 8526        mv    a0,s1
    80000d0c: 86a2        mv    a3,s0
    80000d0e: 87c2        mv    a5,a6
    80000d10: 02000713    li    a4,32
    80000d14: 4601        li    a2,0
    80000d16: 0002a597    auipc a1,0x2a
    80000d1a: 68258593    addi  a1,a1,1666 # 8002b398 <block0_v_weights>
    80000d1e: 58bd        li    a7,-17
    80000d20: dbdff0ef    jal   80000adc <projection_accumulate_i8>
```

V accumulation uses the same input, dimensions, accumulator, and input zero point as Q and K. Its weight pointer is `block0_v_weights`.

The nine preparation instructions and one call contribute ten wrapper instructions.

After accumulation finishes, execution returns to `0x80000d24`.

#### V Requantization Preparation and Wrapper Exit

```asm
    80000d24: 8522        mv   a0,s0
    80000d26: 7402        ld   s0,32(sp)
    80000d28: 70a2        ld   ra,40(sp)
    80000d2a: 64e2        ld   s1,24(sp)
    80000d2c: 69a2        ld   s3,8(sp)
    80000d2e: 6a02        ld   s4,0(sp)
    80000d30: 85ca        mv   a1,s2
    80000d32: 6942        ld   s2,16(sp)
    80000d34: 651846b7    lui  a3,0x65184
    80000d38: ca568693    addi a3,a3,-859
    80000d3c: 4791        li   a5,4
    80000d3e: 02700713    li   a4,39
    80000d42: 6605        lui  a2,0x1
    80000d44: 6145        addi sp,sp,48
    80000d46: b1c1        j    80000a06 <requantize_i32_to_i8>
```

#### Argument Mapping

| Register | Argument | Value |
|---|---|---|
| `a0` | INT32 source pointer | Shared accumulator |
| `a1` | INT8 destination pointer | V output |
| `a2` | Element count | 4,096 |
| `a3` | Multiplier | `0x65183ca5` |
| `a4` | Right shift | 39 |
| `a5` | Output zero point | 4 |

#### Preserve Arguments Before Restoring Registers

The accumulator pointer is copied before restoring the caller's original `s0`:

```asm
mv a0,s0
ld s0,32(sp)
```

Likewise, the V output pointer is copied before restoring the caller's original `s2`:

```asm
mv a1,s2
ld s2,16(sp)
```

The remaining loads restore `s1`, `s3`, `s4`, and the original return address in `ra`.

```asm
addi sp,sp,48
```

releases the QKV wrapper's stack frame.

#### Tail Call

The final transfer uses `j`, rather than `jal`:

```asm
j 80000a06 <requantize_i32_to_i8>
```

V requantization is the wrapper's final operation. The compiler therefore restores the caller's state before jumping directly into the requantization kernel.

Unlike `jal`, this jump does not overwrite `ra`. When requantization executes `ret`, it returns directly to the caller of `block0_qkv_forward()`.

There is no separate QKV wrapper return after V requantization.

#### Final-Section Instruction Count

| Component | Executed instructions |
|---|---:|
| V requantization argument preparation | 7 |
| Restore six saved registers | 6 |
| Release stack frame | 1 |
| Tail jump | 1 |
| **Subtotal** | **15** |

These instructions are interleaved in the assembly.

### 2.5 Complete Wrapper Instruction Count

All instructions executed directly by the wrapper are counted below.

| Wrapper section | Executed instructions |
|---|---:|
| Function entry and Q accumulation preparation/call | 21 |
| Q requantization preparation/call | 8 |
| K accumulation preparation/call | 10 |
| K requantization preparation/call | 8 |
| V accumulation preparation/call | 10 |
| V requantization preparation, restoration, and tail jump | 15 |
| **Total** | **72** |

The 72 instructions include calls and the final tail jump. They exclude all instructions executed inside the three accumulation invocations and three requantization invocations.

The full QKV count is therefore:

$$
I_{\text{QKV}}
=
72
+
I_{\text{Q accumulation}}
+
I_{\text{K accumulation}}
+
I_{\text{V accumulation}}
+
I_{\text{Q requantization}}
+
I_{\text{K requantization}}
+
I_{\text{V requantization}}.
$$

The shared accumulation counts are identical for Q, K, and V because their dimensions, bias setting, and execution paths match.

The requantization counts can differ because clipping is data-dependent.


## 3. Shared Kernel: `projection_accumulate_i8()`

### 3.1 Arguments and Mathematical Operation

The kernel receives its eight arguments in `a0–a7`:

| Register on entry | Argument | Value for QKV |
|---|---|---|
| `a0` | Input pointer | Input activation |
| `a1` | Weight pointer | Q, K, or V weights |
| `a2` | Bias pointer | `NULL` |
| `a3` | Output pointer | Shared INT32 accumulator |
| `a4` | Token count | 32 |
| `a5` | Input-feature count | 128 |
| `a6` | Output-feature count | 128 |
| `a7` | Input zero point | −17 |

For each token and output feature, it computes:

$$
A[t,o]
=
\sum_{k=0}^{127}
\left(X[t,k]-Z_X\right)W[o,k].
$$

For QKV, there is no bias contribution because `bias == NULL`.

The C implementation has three nested loops:

```c
for (size_t t = 0; t < token_count; ++t) {
    for (size_t o = 0; o < output_features; ++o) {
        for (size_t k = 0; k < input_features; ++k) {
            // Center input, multiply, and accumulate.
        }

        // Optional bias addition and output storage.
    }
}
```

The assembly implements these loops primarily through pointer updates.

### 3.2 Function Entry and Initial Setup

#### Assembly

```asm
0000000080000adc <projection_accumulate_i8>:
    80000adc: c345        beqz a4,80000b7c
    80000ade: 7179        addi sp,sp,-48
    80000ae0: f422        sd   s0,40(sp)
    80000ae2: f026        sd   s1,32(sp)
    80000ae4: 8442        mv   s0,a6
    80000ae6: 84ba        mv   s1,a4
    80000ae8: 08080063    beqz a6,80000b68
    80000aec: 872a        mv   a4,a0
    80000aee: 00281513    slli a0,a6,0x2
    80000af2: ec4a        sd   s2,24(sp)
    80000af4: e84e        sd   s3,16(sp)
    80000af6: e452        sd   s4,8(sp)
    80000af8: 8936        mv   s2,a3
    80000afa: 00f70f33    add  t5,a4,a5
    80000afe: 00a606b3    add  a3,a2,a0
    80000b02: 4981        li   s3,0
    80000b04: 4a01        li   s4,0
```

#### Dimension Checks

```asm
beqz a4,80000b7c
beqz a6,80000b68
```

These handle zero token count and zero output-feature count.

For QKV, the counts are 32 and 128, so neither branch is taken.

#### Stack Storage

```asm
addi sp,sp,-48
```

The kernel reserves its own 48-byte stack frame.

It saves five registers:

| Register | Stack location |
|---|---|
| `s4` | `8(sp)` |
| `s3` | `16(sp)` |
| `s2` | `24(sp)` |
| `s1` | `32(sp)` |
| `s0` | `40(sp)` |

Five saved 64-bit registers require 40 bytes. The remaining eight bytes provide padding to maintain 16-byte stack alignment.

This kernel does not save `ra` because it does not call another function.

#### Dimensions and Pointer State

| Instruction | Meaning |
|---|---|
| `mv s0,a6` | Preserve output-feature count |
| `mv s1,a4` | Preserve token count |
| `mv a4,a0` | Initialize current input-row pointer |
| `slli a0,a6,0x2` | Calculate `output_features × 4` bytes |
| `mv s2,a3` | Preserve output base pointer |
| `add t5,a4,a5` | Calculate current input-row end pointer |
| `add a3,a2,a0` | Establish output-loop endpoint using bias-base address |
| `li s3,0` | Initialize output-element offset |
| `li s4,0` | Initialize token index `t` |

The compiler uses the advancing bias address as an output-loop counter. Even when the bias pointer is null, this address arithmetic does not itself read memory.

For the nonzero QKV dimensions, all 17 setup instructions execute once per kernel invocation.

### 3.3 Per-Token Setup

#### Corresponding C

```c
const int8_t *input_row = input + t * input_features;
```

The assembly maintains the input-row pointer incrementally. It also prepares the output-row pointer and resets the weight-row offset.

#### Assembly

```asm
    80000b06: 00299293    slli t0,s3,0x2
    80000b0a: 92ca        add  t0,t0,s2
    80000b0c: 8fb2        mv   t6,a2
    80000b0e: 4381        li   t2,0
```

| Register | Meaning |
|---|---|
| `s3` | Output-element offset: `t × output_features` |
| `t0` | Pointer to the current INT32 output element |
| `t6` | Bias pointer/counter, reset for each token |
| `t2` | Weight-row byte offset, initially zero |

The shift by two converts an INT32 element offset into a byte offset:

$$
\text{byte offset}=4\times\text{element offset}.
$$

These four instructions execute once per token.

### 3.4 Per-Output-Feature Setup

#### Corresponding C

```c
const int8_t *weight_row = weights + o * input_features;
int32_t accumulator = 0;
```

#### Assembly

```asm
    80000b10: c7a5        beqz a5,80000b78
    80000b12: 00758e33    add  t3,a1,t2
    80000b16: 853a        mv   a0,a4
    80000b18: 4e81        li   t4,0
```

| Instruction | Meaning |
|---|---|
| `beqz a5,...` | Handle zero input-feature count |
| `add t3,a1,t2` | Calculate the current weight-row pointer |
| `mv a0,a4` | Restart input reading at the current token's first feature |
| `li t4,0` | Initialize the accumulator to zero |

For QKV, `input_features == 128`, so the branch is not taken.

These four instructions execute once per output feature for each token.

### 3.5 Inner Multiplication and Accumulation Loop

#### Corresponding C

```c
for (size_t k = 0; k < input_features; ++k) {
    const int32_t centered_input =
        (int32_t)input_row[k] - input_zero_point;

    const int32_t weight = (int32_t)weight_row[k];

    accumulator += centered_input * weight;
}
```

#### Assembly

```asm
    80000b1a: 00050303    lb   t1,0(a0)
    80000b1e: 000e0803    lb   a6,0(t3)
    80000b22: 0505        addi a0,a0,1
    80000b24: 4113033b    subw t1,t1,a7
    80000b28: 0268083b    mulw a6,a6,t1
    80000b2c: 0e05        addi t3,t3,1
    80000b2e: 01d80ebb    addw t4,a6,t4
    80000b32: ffe514e3    bne  a0,t5,80000b1a
```

#### Instruction Mapping

| Address | Instruction | Purpose |
|---|---|---|
| `80000b1a` | `lb t1,0(a0)` | Load and sign-extend the INT8 input |
| `80000b1e` | `lb a6,0(t3)` | Load and sign-extend the INT8 weight |
| `80000b22` | `addi a0,a0,1` | Advance the input pointer |
| `80000b24` | `subw t1,t1,a7` | Subtract the input zero point |
| `80000b28` | `mulw a6,a6,t1` | Multiply weight by centered input |
| `80000b2c` | `addi t3,t3,1` | Advance the weight pointer |
| `80000b2e` | `addw t4,a6,t4` | Add the product to the accumulator |
| `80000b32` | `bne a0,t5,...` | Repeat until the input-row end |

The loop does not retain an explicit `k` index. Pointer advancement and the end-pointer comparison implement the loop condition.

#### Instructions per Inner Iteration

| Purpose | Executed instructions |
|---|---:|
| Input and weight loads | 2 |
| Input zero-point subtraction | 1 |
| Multiplication | 1 |
| Accumulation addition | 1 |
| Pointer updates | 2 |
| Loop branch | 1 |
| **Total per iteration** | **8** |

Each iteration performs one mathematical MAC, plus input centering and supporting instructions.

The branch executes on every iteration, including the final iteration when it is not taken.

### 3.6 Optional Bias and Output Storage

#### Corresponding C

```c
if (bias != NULL) {
    accumulator += bias[o];
}

output[t * output_features + o] = accumulator;
```

#### Assembly

```asm
    80000b36: c609        beqz a2,80000b40
    80000b38: 000fa503    lw   a0,0(t6)
    80000b3c: 01d50ebb    addw t4,a0,t4
    80000b40: 01d2a023    sw   t4,0(t0)
```

For QKV, the bias pointer is null:

- The `beqz` executes and is taken.
- The bias `lw` and `addw` do not execute.
- The `sw` stores the INT32 accumulator.

The bias check and output store therefore contribute two executed instructions per output element.

### 3.7 Output-Feature and Token Loop Updates

#### Advance to the Next Output Feature

```asm
    80000b44: 0f91        addi t6,t6,4
    80000b46: 0291        addi t0,t0,4
    80000b48: 93be        add  t2,t2,a5
    80000b4a: fcdf93e3    bne  t6,a3,80000b10
```

| Instruction | Meaning |
|---|---|
| `addi t6,t6,4` | Advance the bias pointer/counter |
| `addi t0,t0,4` | Advance the INT32 output pointer |
| `add t2,t2,a5` | Advance the weight-row offset by `input_features` bytes |
| `bne t6,a3,...` | Continue the output-feature loop |

No bias memory is read when the bias pointer is null. Its advancing address is still used to control the loop.

#### Advance to the Next Token

```asm
    80000b4e: 0a05        addi s4,s4,1
    80000b50: 99a2        add  s3,s3,s0
    80000b52: 973e        add  a4,a4,a5
    80000b54: 9f3e        add  t5,t5,a5
    80000b56: fb4498e3    bne  s1,s4,80000b06
```

| Instruction | Meaning |
|---|---|
| `addi s4,s4,1` | Increment token index |
| `add s3,s3,s0` | Advance output-element offset by `output_features` |
| `add a4,a4,a5` | Advance input-row pointer by `input_features` |
| `add t5,t5,a5` | Advance input-row end pointer |
| `bne s1,s4,...` | Continue the token loop |

The pointer updates replace repeated calculations of expressions such as `input + t * input_features`.

### 3.8 Function Exit

#### Assembly

```asm
    80000b5a: 6962        ld   s2,24(sp)
    80000b5c: 69c2        ld   s3,16(sp)
    80000b5e: 6a22        ld   s4,8(sp)
    80000b60: 7422        ld   s0,40(sp)
    80000b62: 7482        ld   s1,32(sp)
    80000b64: 6145        addi sp,sp,48
    80000b66: 8082        ret
```

These instructions:

1. Restore the caller's five saved registers.
2. Release the kernel's 48-byte stack frame.
3. Return to the caller using `ra`.

The normal QKV exit path executes seven instructions.

### 3.9 Loop Boundaries

| C loop | Assembly range | Repetitions for one QKV projection |
|---|---|---|
| Token loop, `t` | `80000b06–80000b56` | 32 tokens |
| Output-feature loop, `o` | `80000b10–80000b4a` | 128 outputs per token |
| Input-feature loop, `k` | `80000b1a–80000b32` | 128 inputs per output |

The ranges are nested. Their instruction counts must not be added as independent blocks without accounting for that nesting.

Zero-dimension alternative paths are outside the normal execution path analyzed here.


## 4. Shared Kernel: `requantize_i32_to_i8()`

### 4.1 Arguments and Mathematical Operation

| Register on entry | Argument |
|---|---|
| `a0` | INT32 input pointer |
| `a1` | INT8 output pointer |
| `a2` | Element count |
| `a3` | INT64 multiplier |
| `a4` | Right shift |
| `a5` | Output zero point |

For each input accumulator $A_i$, the function computes:

$$
R = 2^{s-1},
$$

$$
C_i = \left((A_iM + R)\gg s\right)+Z_Y,
$$

$$
Y_i = \operatorname{clip}_{[-128,127]}(C_i).
$$

Here, $M$ is the integer multiplier, $s$ is the right shift, and $Z_Y$ is the output zero point.

For Q:

- Element count: 4,096.
- Multiplier: `0x4c59aa35`.
- Right shift: 39.
- Rounding offset: $2^{38}$.
- Output zero point: −12.

### 4.2 Assembly

```asm
0000000080000a06 <requantize_i32_to_i8>:
    80000a06: c231        beqz  a2,80000a4a
    80000a08: 060a        slli  a2,a2,0x2
    80000a0a: fff7081b    addiw a6,a4,-1
    80000a0e: 4885        li    a7,1
    80000a10: 010898b3    sll   a7,a7,a6
    80000a14: 962a        add   a2,a2,a0
    80000a16: 07f00e13    li    t3,127
    80000a1a: f8000313    li    t1,-128
    80000a1e: 00052803    lw    a6,0(a0)
    80000a22: 0511        addi  a0,a0,4
    80000a24: 02d80833    mul   a6,a6,a3
    80000a28: 9846        add   a6,a6,a7
    80000a2a: 40e85833    sra   a6,a6,a4
    80000a2e: 983e        add   a6,a6,a5
    80000a30: 010e5463    bge   t3,a6,80000a38
    80000a34: 07f00813    li    a6,127
    80000a38: 00685463    bge   a6,t1,80000a40
    80000a3c: f8000813    li    a6,-128
    80000a40: 01058023    sb    a6,0(a1)
    80000a44: 0585        addi  a1,a1,1
    80000a46: fca61ce3    bne   a2,a0,80000a1e
    80000a4a: 8082        ret
```

### 4.3 Initial Setup

The instructions at `0x80000a06–0x80000a1a` prepare the conversion loop.

| Instructions | Purpose |
|---|---|
| `beqz a2,...` | Return immediately if the element count is zero |
| `slli a2,a2,0x2` | Convert the INT32 element count into a byte count |
| `addiw a6,a4,-1` | Calculate `shift - 1` |
| `li a7,1` | Initialize the rounding-offset calculation |
| `sll a7,a7,a6` | Calculate `1 << (shift - 1)` |
| `add a2,a2,a0` | Calculate the input end pointer |
| `li t3,127` | Prepare the upper clamp bound |
| `li t1,-128` | Prepare the lower clamp bound |

For a nonzero element count, all eight setup instructions execute once.

The function does not allocate stack storage or call another function.

### 4.4 Input Load and Multiplication

#### Corresponding C

```c
const int64_t product =
    (int64_t)input[i] * multiplier;
```

#### Assembly

```asm
80000a1e: lw   a6,0(a0)
80000a22: addi a0,a0,4
80000a24: mul  a6,a6,a3
```

`lw` loads an INT32 value and sign-extends it to the 64-bit register width.

The input pointer advances by four bytes because each source element is INT32.

The multiplication uses `mul`, producing the 64-bit result required here. This differs from the `mulw` instruction used in projection accumulation.

### 4.5 Rounding, Right Shift, and Zero-Point Addition

#### Corresponding C

```c
int64_t converted =
    (product + rounding_offset) >> shift;

converted += (int64_t)output_zero_point;
```

#### Assembly

```asm
80000a28: add a6,a6,a7
80000a2a: sra a6,a6,a4
80000a2e: add a6,a6,a5
```

These instructions:

1. Add the rounding offset.
2. Apply an arithmetic right shift.
3. Add the output zero point.

The arithmetic shift preserves the sign when shifting negative values.

### 4.6 INT8 Clamping

#### Corresponding C

```c
if (converted < -128) {
    converted = -128;
} else if (converted > 127) {
    converted = 127;
}
```

#### Assembly

```asm
80000a30: bge t3,a6,80000a38
80000a34: li  a6,127

80000a38: bge a6,t1,80000a40
80000a3c: li  a6,-128
```

The compiler checks the upper bound first while preserving the result of the C implementation.

| Value before clamping | Executed clamp comparisons | Additional assignment |
|---|---:|---|
| Below −128 | 2 | `li a6,-128` |
| Within −128 to 127 | 2 | None |
| Above 127 | 2 | `li a6,127` |

Both comparisons execute for every element. A clipped element executes one additional instruction.

This is a data-dependent contribution to the instruction count.

### 4.7 Output Store, Loop Control, and Return

#### Corresponding C

```c
output[i] = (int8_t)converted;
```

#### Assembly

```asm
80000a40: sb   a6,0(a1)
80000a44: addi a1,a1,1
80000a46: bne  a2,a0,80000a1e
80000a4a: ret
```

`sb` stores one INT8 result, and the output pointer advances by one byte.

The loop compares the advancing input pointer against the input end pointer. It does not retain an explicit `i` index.

After all elements are processed, `ret` returns to the caller.

### 4.8 Executed Instruction Counts

#### Per Element

| Purpose | Instructions |
|---|---:|
| Load INT32 input | 1 |
| Multiply | 1 |
| Add rounding offset and output zero point | 2 |
| Arithmetic right shift | 1 |
| Clamp comparisons | 2 |
| Store INT8 output | 1 |
| Input/output pointer updates | 2 |
| Loop branch | 1 |
| **Base count per element** | **11** |
| Additional assignment for a clipped element | **1** |

#### Complete Function

For a nonzero element count $N$, let $C$ be the number of elements requiring clipping:

$$
I_{\text{requantization}} = 8 + 11N + C + 1.
$$

The terms represent setup, loop execution, clipping assignments, and return.

For one QKV projection, $N=4096$:

$$
I_{\text{requantization}} = 45{,}065+C. 
$$

This count includes the kernel's `ret`. It excludes argument preparation and the call instruction in the QKV wrapper.

Q, K, and V use this same compiled kernel. Their clipping counts may differ because their accumulator values and quantization parameters differ.


## 5. Complete QKV Instruction Accounting

### 5.1 Accumulation Counts

Each Q, K, and V accumulation uses:

- Token count, T = 32.
- Output features, O = 128.
- Input features, K = 128.
- No bias.

The three loop levels are counted separately. Instructions belonging
to a nested loop are excluded from its enclosing loop's additional count.

#### Instructions at Each Level

| Level | Instructions per execution | Executions per projection | Total |
|---|---:|---:|---:|
| Inner MAC loop | 8 | 32 × 128 × 128 | 4,194,304 |
| Additional output-feature work | 10 | 32 × 128 | 40,960 |
| Additional token work | 9 | 32 | 288 |
| Function entry | 17 | 1 | 17 |
| Function exit | 7 | 1 | 7 |
| **Total per projection** | | | **4,235,576** |

The eight inner-loop instructions consist of:

| Purpose | Instructions per MAC |
|---|---:|
| Input and weight loads | 2 |
| Input zero-point subtraction | 1 |
| Multiplication | 1 |
| Accumulation addition | 1 |
| Pointer updates | 2 |
| Loop branch | 1 |
| **Total** | **8** |

The ten additional instructions per output consist of:

- Four instructions for the input-feature check, pointer preparation,
  and accumulator initialization.
- One bias check and one output store.
- Four instructions for pointer/offset updates and output-loop control.

The bias load and bias addition are skipped because the bias pointer
is null.

The nine additional instructions per token consist of:

- Four instructions for per-token setup.
- Five instructions for token-state updates and loop control.

#### Count Formula

For the nonzero dimensions and null-bias path analyzed here:

$$
I_{\text{accumulation}}
=
17 + T\left[9 + O(10 + 8K)\right] + 7.
$$

Substituting the QKV dimensions:

$$
I_{\text{accumulation}}
=
24 + 32\left[9 + 128(10 + 8 \times 128)\right]
=
4{,}235{,}576.
$$

Q, K, and V execute the same accumulation path:

| Projection | Mathematical MACs | Accumulation-kernel instructions |
|---|---:|---:|
| Q | 524,288 | 4,235,576 |
| K | 524,288 | 4,235,576 |
| V | 524,288 | 4,235,576 |
| **Total** | **1,572,864** | **12,706,728** |

Each mathematical MAC corresponds to one `mulw` and one accumulation
`addw`. Therefore, the three accumulation calls execute **1,572,864
multiplications** and **1,572,864 accumulation additions**.

Input centering, pointer arithmetic, and other instructions are
counted separately from these MAC operations.

Loop branches count on every iteration, including their final
not-taken execution. Kernel totals include their return instructions
but exclude the calls already counted in the QKV wrapper.

### 5.2 Requantization Counts

Each projection produces 32 × 128 = 4,096 INT32 accumulator values.
Requantization independently converts each value into INT8 by:

1. Multiplying by the integer scaling multiplier.
2. Adding the rounding offset.
3. Applying an arithmetic right shift.
4. Adding the output zero point.
5. Clamping to the INT8 range and storing the result.

There is one loop over the output elements.

#### Instructions per Element

| Purpose | Executed instructions |
|---|---:|
| Load INT32 input | 1 |
| Multiply by scaling multiplier | 1 |
| Add rounding offset | 1 |
| Arithmetic right shift | 1 |
| Add output zero point | 1 |
| Clamp comparisons | 2 |
| Store INT8 output | 1 |
| Input/output pointer updates | 2 |
| Loop branch | 1 |
| **Base total per element** | **11** |
| Additional assignment when clipping is required | **1** |

Both clamp comparisons execute for every element. A value outside
the INT8 range executes one additional assignment to either −128 or 127.

#### Count per Projection

For N elements and C clipping assignments:

$$
I_{\text{requantization}}
=
8 + 11N + C + 1.
$$

The eight setup instructions execute once before the loop. The final
return instruction also executes once.

For N = 4,096:

$$
I_{\text{requantization}}
=
8 + 11 \times 4096 + C + 1
=
45{,}065 + C.
$$

#### Combined Q, K, and V Count

| Projection | Elements | Requantization-kernel instructions |
|---|---:|---:|
| Q | 4,096 | 45,065 + C_Q |
| K | 4,096 | 45,065 + C_K |
| V | 4,096 | 45,065 + C_V |
| **Total** | **12,288** | **135,195 + C_Q + C_K + C_V** |

Here, C_Q, C_K, and C_V count elements requiring clipping in each
projection.

Across the three conversion loops, the fixed arithmetic counts are:

| Operation | Executed instructions |
|---|---:|
| Scaling multiplication (`mul`) | 12,288 |
| Rounding-offset addition (`add`) | 12,288 |
| Output-zero-point addition (`add`) | 12,288 |
| Arithmetic right shift (`sra`) | 12,288 |

These kernel totals include setup and return instructions. Argument
preparation, calls, and the V tail jump are counted separately in
the QKV wrapper.

### 5.3 Instruction Breakdown

Instructions are classified by their purpose. Multiplication used inside a MAC is separated from requantization multiplication. Likewise, accumulation additions are separated from pointer updates.

#### Accumulation Inner Loop

```asm
lb    t1,0(a0)       # Input load
lb    a6,0(t3)       # Weight load
addi  a0,a0,1       # Input-pointer update
subw  t1,t1,a7      # Input zero-point subtraction
mulw  a6,a6,t1      # MAC multiplication
addi  t3,t3,1       # Weight-pointer update
addw  t4,a6,t4      # MAC accumulation
bne   a0,t5,...     # Loop control
```

Across Q, K, and V, the inner loop executes:

$$
3 \times 32 \times 128 \times 128
=
1{,}572{,}864\text{ iterations}.
$$

| Purpose | Instructions per iteration | Total across QKV |
|---|---:|---:|
| MAC multiplication (`mulw`) | 1 | 1,572,864 |
| MAC accumulation (`addw`) | 1 | 1,572,864 |
| Input centering (`subw`) | 1 | 1,572,864 |
| Input and weight loads (`lb`) | 2 | 3,145,728 |
| Pointer updates (`addi`) | 2 | 3,145,728 |
| Loop control (`bne`) | 1 | 1,572,864 |
| **Total** | **8** | **12,582,912** |

One mathematical MAC therefore corresponds to eight executed
instructions in this scalar loop. Two perform multiplication and accumulation; the remaining six handle operand access, input centering, pointer advancement, and loop control.

#### Remaining Accumulation Instructions

The accumulation kernels also execute instructions outside the innermost loop.

| Work | Calculation across three projections | Instructions |
|---|---|---:|
| Per-token setup | 3 × 32 × 4 | 384 |
| Per-output setup | 3 × 32 × 128 × 4 | 49,152 |
| Bias checks and output stores | 3 × 32 × 128 × 2 | 24,576 |
| Output-feature updates and branches | 3 × 32 × 128 × 4 | 49,152 |
| Token updates and branches | 3 × 32 × 5 | 480 |
| Function entry and exit | 3 × 24 | 72 |
| **Total outside the inner loops** | | **123,816** |

Combining both regions:

$$
I_{\text{three accumulations}}
=
12{,}582{,}912 + 123{,}816
=
12{,}706{,}728.
$$

The optional bias loads and bias additions do not execute for QKV.

#### Requantization Instructions

The three conversions process 12,288 elements in total.

| Purpose | Total across QKV |
|---|---:|
| Scaling multiplication (`mul`) | 12,288 |
| Rounding-offset additions | 12,288 |
| Output-zero-point additions | 12,288 |
| Arithmetic right shifts | 12,288 |
| INT32 input loads | 12,288 |
| INT8 output stores | 12,288 |
| Input/output pointer updates | 24,576 |
| Clamp comparisons | 24,576 |
| Loop branches | 12,288 |
| Function setup and returns | 27 |
| Data-dependent clamp assignments | C_total |
| **Total** | **135,195 + C_total** |

Here:

$$
C_{\text{total}} = C_Q + C_K + C_V.
$$

The setup-and-return row includes three invocations, each with eight setup instructions and one return instruction.

#### Combined Instruction Accounting

| Region | Instructions |
|---|---:|
| Accumulation inner loops | 12,582,912 |
| Remaining accumulation work | 123,816 |
| Requantization kernels | 135,195 + C_total |
| QKV wrapper | 72 |
| **Complete QKV function including callees** | **12,841,995 + C_total** |

These regions do not overlap. Measurement and caller overhead are excluded and will be reconciled separately.

#### Arithmetic Totals

| Arithmetic purpose | Executed instructions |
|---|---:|
| MAC multiplication | 1,572,864 |
| Requantization multiplication | 12,288 |
| **Total multiplication instructions** | **1,585,152** |
| MAC accumulation addition | 1,572,864 |
| Requantization rounding and zero-point additions | 24,576 |
| **Total data-processing additions listed above** | **1,597,440** |
| Input zero-point subtraction | 1,572,864 |

The addition total above excludes address calculations, pointer updates, loop counters, and setup arithmetic.

These counts identify instruction costs. They do not establish the hardware cycle cost of multiplication, memory access, or branching.


### 5.4 Reconciliation with `minstret`

The calculated count includes the QKV wrapper and all six kernel invocations. The raw counter measurement also includes instructions executed at the profiling call site.

#### Measurement Boundary

The individual QKV measurement uses:

```asm
80001a54: csrr s5,minstret
80001a58: ld   a4,88(sp)
80001a5a: ld   a3,168(sp)
80001a5c: ld   a2,160(sp)
80001a5e: ld   a1,152(sp)
80001a60: ld   a0,144(sp)
80001a62: jal  80000c86 <block0_qkv_forward>
80001a66: csrr a5,minstret
```

The raw delta includes seven instructions outside the QKV function:

| Component | Instructions |
|---|---:|
| First counter read | 1 |
| Argument preparation | 5 |
| Call into the QKV wrapper | 1 |
| **Total outside QKV** | **7** |

The ending counter read's own retirement is not included in the counter value it returns.

The subtraction and storage of the measured delta happen afterward and are outside the measurement interval.

Therefore:

$$
I_{\text{QKV}} = I_{\text{raw}} - 7.
$$

This correction applies to this particular compiled call site.

#### Calculated Count

From the preceding analysis:

$$
I_{\text{QKV}}
=
12{,}706{,}728
+
135{,}195
+
C_{\text{total}}
+
72.
$$

Thus:

$$
I_{\text{QKV}}
=
12{,}841{,}995 + C_{\text{total}},
$$

and:

$$
I_{\text{raw}}
=
12{,}842{,}002 + C_{\text{total}}.
$$

Here, C_total is the combined number of clipping assignments across Q, K, and V requantization.

#### Comparison Across Eight Cases

| Case | Measured raw delta | QKV count after subtracting 7 | Inferred C_total |
|---|---:|---:|---:|
| 0 | 12,842,003 | 12,841,996 | 1 |
| 1 | 12,842,005 | 12,841,998 | 3 |
| 2 | 12,842,004 | 12,841,997 | 2 |
| 3 | 12,842,002 | 12,841,995 | 0 |
| 4 | 12,842,002 | 12,841,995 | 0 |
| 5 | 12,842,005 | 12,841,998 | 3 |
| 6 | 12,842,004 | 12,841,997 | 2 |
| 7 | 12,842,003 | 12,841,996 | 1 |
| **Mean** | **12,842,003.5** | **12,841,996.5** | **1.5** |

The remaining differences are consistent with zero to three clipping assignments per case.

These clipping counts were initially inferred by subtracting the fixed instruction count from the measurements. The latest-ELF semantic replay now verifies their call-by-call attribution; see Section 7 for the Q, K, and V split.

#### Case 0 Accounting

| Component | Instructions |
|---|---:|
| Three accumulation kernels | 12,706,728 |
| Three requantization kernels, including inferred clipping | 135,196 |
| QKV wrapper | 72 |
| **QKV function including callees** | **12,841,996** |
| Measurement/caller overhead | 7 |
| **Raw delta** | **12,842,003** |

The calculated accounting reconciles with the supplied case-0 measurement.

## 6. Findings and Optimization Opportunities

### 6.1 Dominant Instruction Region

QKV accounts for approximately **22.46%** of the mean full-block raw instruction count across the eight fixtures.

Within QKV, the case-0 distribution is:

| Component | Instructions | Share of QKV function count |
|---|---:|---:|
| Accumulation kernels | 12,706,728 | 98.95% |
| Requantization kernels | 135,196 | 1.05% |
| Wrapper | 72 | Approximately 0.00056% |
| **Total** | **12,841,996** | **100%** |

Percentages are rounded. Measurement/caller overhead is excluded from this table.

The accumulation kernels are the main instruction-count hotspot. Requantization and wrapper overhead contribute much smaller shares.

### 6.2 Arithmetic and Supporting Work

The accumulation inner loop executes eight instructions per input/weight pair:

| Work | Instructions per iteration | Share of inner-loop instructions |
|---|---:|---:|
| Multiplication and accumulation | 2 | 25% |
| Input and weight loads | 2 | 25% |
| Pointer updates | 2 | 25% |
| Input centering | 1 | 12.5% |
| Loop branch | 1 | 12.5% |
| **Total** | **8** | **100%** |

The arithmetic MAC consists of one multiplication and one accumulation addition. The complete scalar iteration requires six additional instructions. Therefore, 75% of the inner-loop instructions support the MAC. These instructions perform necessary work. Their share does not mean they can all be removed. Memory loads account for 25% of the loop instructions. Pointer arithmetic, input centering, and branching are separate costs.


### 6.3 Requantization Optimization

QKV requantization contributes approximately 1.05% of QKV instructions.

Its repeated sequence includes a 64-bit multiplication, rounding addition, arithmetic shift, zero-point addition, and clamping.

A requantization instruction could reduce this sequence, but its effect on QKV's total instruction count is limited by this small share. Its contribution in other block stages must be analyzed separately.

Any replacement must preserve the existing numerical behavior.

### 6.4 Interpretation and Limits

The current evidence identifies an **instruction-count hotspot**: the scalar projection accumulation loop.

It does not establish whether hardware execution time is dominated by multiplication latency, load-use dependencies, cache misses, branch behavior, or another pipeline effect.

Spike instruction counts therefore support instruction-reduction analysis. CV-Wally measurements are needed to establish hardware cycle improvements.

The main optimization target is the complete accumulation loop, including arithmetic and supporting instructions.

[Back to block-0 profiling results](Result.md)

## 7. Verification Update for the Supplied Latest ELF

The earlier analysis inferred clipping from the measured residual. A scoped semantic replay of the supplied disassembly and ELF now reproduces all 32,768 final T107 outputs and all eight stage counts. The helper-by-helper clipping counts below identify which projection contains the saturation assignments. They agree with the previously inferred totals.

| Case | Q clipped elements | K clipped elements | V clipped elements | Total |
|---|---|---|---|---|
| 0 | 1 | 0 | 0 | 1 |
| 1 | 1 | 2 | 0 | 3 |
| 2 | 0 | 2 | 0 | 2 |
| 3 | 0 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 | 0 |
| 5 | 1 | 2 | 0 | 3 |
| 6 | 0 | 2 | 0 | 2 |
| 7 | 0 | 1 | 0 | 1 |

This is validation by local instruction-semantic replay, not a new Spike or CV-Wally run. See [method and reproducibility](analysis_method.md) and [results](Result.md).
