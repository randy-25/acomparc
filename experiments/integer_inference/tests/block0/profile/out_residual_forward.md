# Part 4: Attention Output Projection and First Residual — Disassembly Analysis

## 1. Function Overview

This note analyzes `block0_out_residual_forward()` and every helper it actually executes in the **supplied compiled ELF**. Case 0 provides the detailed instruction mix; Section 5.4 reconciles all eight cases with the supplied Spike `minstret` results. Counts describe dynamic executions, not the number of lines in the disassembly. See [counting method](analysis_method.md).

| Parameter | Current block-0 value |
|---|---|
| projection T / K / O | 32 / 128 / 128 |
| bias | present, 128 INT32 elements |
| input zero point | 9 |
| T71 multiplier / shift / zero point | 747,156,156 / 39 / -13 |
| T72 multiplier / shift / input zero point | 939,062,400 / 22 / -13 |
| T0 alignment / zero point | 256 / 0 |
| T73 output | INT32 [32,128], scale 2^-16 |


## 2. Block Wrapper: `block0_out_residual_forward()`

### 2.1 Source and operation order

```c
void block0_out_residual_forward(
    const int8_t *attention_output,
    const int8_t *residual_input,
    int32_t *output,
    int8_t *projection_workspace,
    int32_t *accumulator_workspace
)
{
    const size_t count = BLOCK0_TOKEN_COUNT * BLOCK0_HIDDEN_SIZE;

    projection_accumulate_i8(
        attention_output,
        block0_out_weights,
        block0_out_bias,
        accumulator_workspace,
        BLOCK0_TOKEN_COUNT,
        BLOCK0_HIDDEN_SIZE,
        BLOCK0_HIDDEN_SIZE,
        BLOCK0_OUT_PROJ_INPUT_ZERO_POINT
    );

    /* Preserve the notebook's INT8 T71 boundary. */
    requantize_i32_to_i8(
        accumulator_workspace,
        projection_workspace,
        count,
        BLOCK0_OUT_PROJ_MULTIPLIER,
        BLOCK0_OUT_PROJ_SHIFT,
        BLOCK0_OUT_PROJ_OUTPUT_ZERO_POINT
    );

    /* Op 41 counterpart: T71 -> INT32 T72 at scale 2^-16. */
    requantize_i8_to_i32(
        projection_workspace,
        output,
        count,
        BLOCK0_RESIDUAL1_BRANCH_MULTIPLIER,
        BLOCK0_RESIDUAL1_BRANCH_SHIFT,
        BLOCK0_OUT_PROJ_OUTPUT_ZERO_POINT
    );

    /* Op 42: exact T0 alignment, followed by addition. */
    residual_add_i8_i32(
        residual_input,
        output,
        output,
        count,
        BLOCK0_RESIDUAL1_INPUT_ALIGNMENT,
        BLOCK0_RESIDUAL1_INPUT_ZERO_POINT
    );
}
```


### 2.2 Compiled wrapper

The assembly below includes the wrapper only. Each `jal` or tail `j` transfers execution to a helper whose work is counted separately. `executed` is the case-0 execution frequency of that instruction. A helper reused several times has aggregated frequencies in its own listing. Unexecuted alternate paths are shown with zero.

```asm
0000000080000df2 <block0_out_residual_forward>:
80000df2:	7179                	addi	sp,sp,-48                                                        # executed: 1
80000df4:	08000813          	li	a6,128                                                               # executed: 1
80000df8:	f022                	sd	s0,32(sp)                                                          # executed: 1
80000dfa:	ec26                	sd	s1,24(sp)                                                          # executed: 1
80000dfc:	e84a                	sd	s2,16(sp)                                                          # executed: 1
80000dfe:	e44e                	sd	s3,8(sp)                                                           # executed: 1
80000e00:	8432                	mv	s0,a2                                                              # executed: 1
80000e02:	84b6                	mv	s1,a3                                                              # executed: 1
80000e04:	87c2                	mv	a5,a6                                                              # executed: 1
80000e06:	893a                	mv	s2,a4                                                              # executed: 1
80000e08:	89ae                	mv	s3,a1                                                              # executed: 1
80000e0a:	86ba                	mv	a3,a4                                                              # executed: 1
80000e0c:	48a5                	li	a7,9                                                               # executed: 1
80000e0e:	02000713          	li	a4,32                                                                # executed: 1
80000e12:	00022617          	auipc	a2,0x22                                                           # executed: 1
80000e16:	37e60613          	addi	a2,a2,894 # 80023190 <block0_out_bias>                             # executed: 1
80000e1a:	00022597          	auipc	a1,0x22                                                           # executed: 1
80000e1e:	57658593          	addi	a1,a1,1398 # 80023390 <block0_out_weights>                         # executed: 1
80000e22:	f406                	sd	ra,40(sp)                                                          # executed: 1
80000e24:	cb9ff0ef          	jal	80000adc <projection_accumulate_i8>                                 # executed: 1
80000e28:	2c88b6b7          	lui	a3,0x2c88b                                                          # executed: 1
80000e2c:	854a                	mv	a0,s2                                                              # executed: 1
80000e2e:	85a6                	mv	a1,s1                                                              # executed: 1
80000e30:	2bc68693          	addi	a3,a3,700 # 2c88b2bc <_start-0x53774d44>                           # executed: 1
80000e34:	57cd                	li	a5,-13                                                             # executed: 1
80000e36:	02700713          	li	a4,39                                                                # executed: 1
80000e3a:	6605                	lui	a2,0x1                                                            # executed: 1
80000e3c:	bcbff0ef          	jal	80000a06 <requantize_i32_to_i8>                                     # executed: 1
80000e40:	37f8f6b7          	lui	a3,0x37f8f                                                          # executed: 1
80000e44:	8526                	mv	a0,s1                                                              # executed: 1
80000e46:	85a2                	mv	a1,s0                                                              # executed: 1
80000e48:	48068693          	addi	a3,a3,1152 # 37f8f480 <_start-0x48070b80>                          # executed: 1
80000e4c:	57cd                	li	a5,-13                                                             # executed: 1
80000e4e:	4759                	li	a4,22                                                              # executed: 1
80000e50:	6605                	lui	a2,0x1                                                            # executed: 1
80000e52:	bfbff0ef          	jal	80000a4c <requantize_i8_to_i32>                                     # executed: 1
80000e56:	8622                	mv	a2,s0                                                              # executed: 1
80000e58:	85a2                	mv	a1,s0                                                              # executed: 1
80000e5a:	7402                	ld	s0,32(sp)                                                          # executed: 1
80000e5c:	70a2                	ld	ra,40(sp)                                                          # executed: 1
80000e5e:	64e2                	ld	s1,24(sp)                                                          # executed: 1
80000e60:	6942                	ld	s2,16(sp)                                                          # executed: 1
80000e62:	854e                	mv	a0,s3                                                              # executed: 1
80000e64:	69a2                	ld	s3,8(sp)                                                           # executed: 1
80000e66:	4781                	li	a5,0                                                               # executed: 1
80000e68:	10000713          	li	a4,256                                                               # executed: 1
80000e6c:	6685                	lui	a3,0x1                                                            # executed: 1
80000e6e:	6145                	addi	sp,sp,48                                                         # executed: 1
80000e70:	b339                	j	80000b7e <residual_add_i8_i32>                                      # executed: 1
```


The wrapper itself executes **49 instructions**, excluding all callee bodies. Stack saves retain registers and pointers across calls; the final helper may be tail-called after restoration. A tail call is counted as one wrapper jump followed by the helper's body and return, not a second wrapper epilogue.

#### Incoming registers

| Register on entry | C argument |
|---|---|
| `a0` | const int8_t *attention_output |
| `a1` | const int8_t *residual_input |
| `a2` | int32_t *output |
| `a3` | int8_t *projection_workspace |
| `a4` | int32_t *accumulator_workspace |

#### Retained values and stack frame

| Register | Role in this wrapper |
|---|---|
| `s0` | INT32 output pointer for T72 then T73 |
| `s1` | INT8 projection workspace pointer for T71 |
| `s2` | INT32 projection-accumulator workspace pointer |
| `s3` | Original T0 residual pointer |

All six wrappers analyzed in the new notes allocate 48 bytes. They save `ra` at `40(sp)`, `s0` at `32(sp)`, `s1` at `24(sp)`, `s2` at `16(sp)`, and `s3` at `8(sp)`: 40 bytes of saved values plus eight bytes unused in this frame, keeping the 16-byte stack alignment. These saved slots hold register values, not activation tensors. The callee-saved registers retain values across helper calls, and the wrapper restores the caller's original values before its final tail transfer.

### 2.3 Helper arguments and data boundaries

Pointer arguments select the source tensors and caller-owned scratch shown in the C wrapper. Non-pointer arguments decoded at each actual call are:

| Call in source order | Actual non-pointer arguments |
|---|---|
| output accumulation | T=32, K=128, O=128, input z=9; bias present |
| T71 conversion | N=4096, M=747,156,156, shift=39, output z=-13 |
| T72 alignment | N=4096, M=939,062,400, shift=22, input z=-13 |
| T0 + T72 residual | N=4096, alignment=256, residual z=0 |

## 3. Computation Kernels

### 3.1 `projection_accumulate_i8()` — Biased dense projection

This is the same shared kernel analyzed in [QKV](qkv_forward.md), with a non-NULL bias in this stage. For dimensions T tokens, K inputs, and O outputs:
$$
y_{t,o}=b_o+\sum_{k=0}^{K-1}(x_{t,k}-z_x)w_{o,k}.
$$
The inner loop at `0x80000b1a–0x80000b32` has eight executed instructions per MAC: two `lb`, one `subw`, one `mulw`, one `addw`, two `addi`, and one `bne`.

With B = 1 for a present bias and B = 0 for NULL:
$$
I_{\rm projection}=24+T\{9+O[10+2B+8K]\}.
$$
The 24 covers normal entry/exit, nine covers non-output token work, and ten covers non-inner output work including the bias test. The biased path adds `lw` and `addw` once per output, **not once per MAC**. QKV uses B = 0; this invocation uses B = 1.

#### Corresponding C

```c
void projection_accumulate_i8(
    const int8_t *input,
    const int8_t *weights,
    const int32_t *bias,
    int32_t *output,
    size_t token_count,
    size_t input_features,
    size_t output_features,
    int32_t input_zero_point
)
{
    for (size_t t = 0; t < token_count; ++t) {
        const int8_t *input_row =
            input + t * input_features;

        for (size_t o = 0; o < output_features; ++o) {
            const int8_t *weight_row =
                weights + o * input_features;

            int32_t accumulator = 0;

            for (size_t k = 0; k < input_features; ++k) {
                const int32_t centered_input =
                    (int32_t)input_row[k] - input_zero_point;

                const int32_t weight = (int32_t)weight_row[k];

                accumulator += centered_input * weight;
            }

            if (bias != NULL) {
                accumulator += bias[o];
            }

            output[t * output_features + o] = accumulator;
        }
    }
}
```


#### Compiled assembly

```asm
0000000080000adc <projection_accumulate_i8>:
80000adc:	c345                	beqz	a4,80000b7c <projection_accumulate_i8+0xa0>                      # executed: 1
80000ade:	7179                	addi	sp,sp,-48                                                        # executed: 1
80000ae0:	f422                	sd	s0,40(sp)                                                          # executed: 1
80000ae2:	f026                	sd	s1,32(sp)                                                          # executed: 1
80000ae4:	8442                	mv	s0,a6                                                              # executed: 1
80000ae6:	84ba                	mv	s1,a4                                                              # executed: 1
80000ae8:	08080063          	beqz	a6,80000b68 <projection_accumulate_i8+0x8c>                        # executed: 1
80000aec:	872a                	mv	a4,a0                                                              # executed: 1
80000aee:	00281513          	slli	a0,a6,0x2                                                          # executed: 1
80000af2:	ec4a                	sd	s2,24(sp)                                                          # executed: 1
80000af4:	e84e                	sd	s3,16(sp)                                                          # executed: 1
80000af6:	e452                	sd	s4,8(sp)                                                           # executed: 1
80000af8:	8936                	mv	s2,a3                                                              # executed: 1
80000afa:	00f70f33          	add	t5,a4,a5                                                            # executed: 1
80000afe:	00a606b3          	add	a3,a2,a0                                                            # executed: 1
80000b02:	4981                	li	s3,0                                                               # executed: 1
80000b04:	4a01                	li	s4,0                                                               # executed: 1
80000b06:	00299293          	slli	t0,s3,0x2                                                          # executed: 32
80000b0a:	92ca                	add	t0,t0,s2                                                          # executed: 32
80000b0c:	8fb2                	mv	t6,a2                                                              # executed: 32
80000b0e:	4381                	li	t2,0                                                               # executed: 32
80000b10:	c7a5                	beqz	a5,80000b78 <projection_accumulate_i8+0x9c>                      # executed: 4,096
80000b12:	00758e33          	add	t3,a1,t2                                                            # executed: 4,096
80000b16:	853a                	mv	a0,a4                                                              # executed: 4,096
80000b18:	4e81                	li	t4,0                                                               # executed: 4,096
80000b1a:	00050303          	lb	t1,0(a0)                                                             # executed: 524,288
80000b1e:	000e0803          	lb	a6,0(t3)                                                             # executed: 524,288
80000b22:	0505                	addi	a0,a0,1                                                          # executed: 524,288
80000b24:	4113033b          	subw	t1,t1,a7                                                           # executed: 524,288
80000b28:	0268083b          	mulw	a6,a6,t1                                                           # executed: 524,288
80000b2c:	0e05                	addi	t3,t3,1                                                          # executed: 524,288
80000b2e:	01d80ebb          	addw	t4,a6,t4                                                           # executed: 524,288
80000b32:	ffe514e3          	bne	a0,t5,80000b1a <projection_accumulate_i8+0x3e>                      # executed: 524,288
80000b36:	c609                	beqz	a2,80000b40 <projection_accumulate_i8+0x64>                      # executed: 4,096
80000b38:	000fa503          	lw	a0,0(t6) # 40000000 <_start-0x40000000>                              # executed: 4,096
80000b3c:	01d50ebb          	addw	t4,a0,t4                                                           # executed: 4,096
80000b40:	01d2a023          	sw	t4,0(t0)                                                             # executed: 4,096
80000b44:	0f91                	addi	t6,t6,4                                                          # executed: 4,096
80000b46:	0291                	addi	t0,t0,4                                                          # executed: 4,096
80000b48:	93be                	add	t2,t2,a5                                                          # executed: 4,096
80000b4a:	fcdf93e3          	bne	t6,a3,80000b10 <projection_accumulate_i8+0x34>                      # executed: 4,096
80000b4e:	0a05                	addi	s4,s4,1                                                          # executed: 32
80000b50:	99a2                	add	s3,s3,s0                                                          # executed: 32
80000b52:	973e                	add	a4,a4,a5                                                          # executed: 32
80000b54:	9f3e                	add	t5,t5,a5                                                          # executed: 32
80000b56:	fb4498e3          	bne	s1,s4,80000b06 <projection_accumulate_i8+0x2a>                      # executed: 32
80000b5a:	6962                	ld	s2,24(sp)                                                          # executed: 1
80000b5c:	69c2                	ld	s3,16(sp)                                                          # executed: 1
80000b5e:	6a22                	ld	s4,8(sp)                                                           # executed: 1
80000b60:	7422                	ld	s0,40(sp)                                                          # executed: 1
80000b62:	7482                	ld	s1,32(sp)                                                          # executed: 1
80000b64:	6145                	addi	sp,sp,48                                                         # executed: 1
80000b66:	8082                	ret                                                                   # executed: 1
80000b68:	00140793          	addi	a5,s0,1                                                            # executed: 0
80000b6c:	0409                	addi	s0,s0,2                                                          # executed: 0
80000b6e:	fef489e3          	beq	s1,a5,80000b60 <projection_accumulate_i8+0x84>                      # executed: 0
80000b72:	fe941be3          	bne	s0,s1,80000b68 <projection_accumulate_i8+0x8c>                      # executed: 0
80000b76:	b7ed                	j	80000b60 <projection_accumulate_i8+0x84>                            # executed: 0
80000b78:	4e81                	li	t4,0                                                               # executed: 0
80000b7a:	bf75                	j	80000b36 <projection_accumulate_i8+0x5a>                            # executed: 0
80000b7c:	8082                	ret                                                                   # executed: 0
```


### 3.2 `residual_add_i8_i32()` — First residual addition

Align T0 exactly by computing $256(T0-0)+T72$, then store INT32 T73 at scale $2^{-16}$. The generic helper still executes a `mul` by the runtime alignment value. The ten-instruction loop has two loads (`lb`, `lw`), one centering subtract, one multiply, one numerical add, three pointer updates, one `sw`, and one `bne`. $I=10N+4=40{,}964$. The output may overwrite T72; the original T0 input remains live until this stage completes.

#### Corresponding C

```c
void residual_add_i8_i32(
    const int8_t *residual,
    const int32_t *branch,
    int32_t *output,
    size_t count,
    int64_t alignment,
    int32_t residual_zero_point
)
{
    for (size_t i = 0; i < count; ++i) {
        const int64_t centered =
            (int64_t)residual[i] - residual_zero_point;

        const int64_t aligned = centered * alignment;

        output[i] = (int32_t)((int64_t)branch[i] + aligned);
    }
}
```


#### Compiled assembly

```asm
0000000080000b7e <residual_add_i8_i32>:
80000b7e:	c685                	beqz	a3,80000ba6 <residual_add_i8_i32+0x28>                           # executed: 1
80000b80:	068a                	slli	a3,a3,0x2                                                        # executed: 1
80000b82:	96ae                	add	a3,a3,a1                                                          # executed: 1
80000b84:	00050803          	lb	a6,0(a0)                                                             # executed: 4,096
80000b88:	0005a883          	lw	a7,0(a1)                                                             # executed: 4,096
80000b8c:	0591                	addi	a1,a1,4                                                          # executed: 4,096
80000b8e:	40f80833          	sub	a6,a6,a5                                                            # executed: 4,096
80000b92:	02e80833          	mul	a6,a6,a4                                                            # executed: 4,096
80000b96:	0505                	addi	a0,a0,1                                                          # executed: 4,096
80000b98:	0611                	addi	a2,a2,4                                                          # executed: 4,096
80000b9a:	0108883b          	addw	a6,a7,a6                                                           # executed: 4,096
80000b9e:	ff062e23          	sw	a6,-4(a2)                                                            # executed: 4,096
80000ba2:	feb691e3          	bne	a3,a1,80000b84 <residual_add_i8_i32+0x6>                            # executed: 4,096
80000ba6:	8082                	ret                                                                   # executed: 1
```


## 4. Conversion Kernels

### 4.1 `requantize_i32_to_i8()` — INT32 to INT8 fixed-point conversion

Compute $\operatorname{clamp}_{[-128,127]}(((vM+2^{s-1})\gg s)+z)$, promoting the input before the product. This is the same eleven-instruction ordinary loop as QKV, with `lw` instead of `ld`. With C clipped elements, $I=11N+9+C$. Actual upper/lower assignments are at `0x80000a34` and `0x80000a3c`. The product and fixed-point arithmetic use 64-bit `mul`/`add`/`sra`, even though the input tensor is INT32. The two comparisons execute for every element; an endpoint assignment executes only when a value exceeds a bound.

#### Corresponding C

```c
void requantize_i32_to_i8(
    const int32_t *input,
    int8_t *output,
    size_t element_count,
    int64_t multiplier,
    unsigned int shift,
    int32_t output_zero_point
)
{
    const int64_t rounding_offset =
        INT64_C(1) << (shift - 1);

    for (size_t i = 0; i < element_count; ++i) {
        /* Promote before multiplication. */
        const int64_t product =
            (int64_t)input[i] * multiplier;

        int64_t converted =
            (product + rounding_offset) >> shift;

        converted += (int64_t)output_zero_point;

        if (converted < -128) {
            converted = -128;
        } else if (converted > 127) {
            converted = 127;
        }

        output[i] = (int8_t)converted;
    }
}
```


#### Compiled assembly

```asm
0000000080000a06 <requantize_i32_to_i8>:
80000a06:	c231                	beqz	a2,80000a4a <requantize_i32_to_i8+0x44>                          # executed: 1
80000a08:	060a                	slli	a2,a2,0x2                                                        # executed: 1
80000a0a:	fff7081b          	addiw	a6,a4,-1                                                          # executed: 1
80000a0e:	4885                	li	a7,1                                                               # executed: 1
80000a10:	010898b3          	sll	a7,a7,a6                                                            # executed: 1
80000a14:	962a                	add	a2,a2,a0                                                          # executed: 1
80000a16:	07f00e13          	li	t3,127                                                               # executed: 1
80000a1a:	f8000313          	li	t1,-128                                                              # executed: 1
80000a1e:	00052803          	lw	a6,0(a0)                                                             # executed: 4,096
80000a22:	0511                	addi	a0,a0,4                                                          # executed: 4,096
80000a24:	02d80833          	mul	a6,a6,a3                                                            # executed: 4,096
80000a28:	9846                	add	a6,a6,a7                                                          # executed: 4,096
80000a2a:	40e85833          	sra	a6,a6,a4                                                            # executed: 4,096
80000a2e:	983e                	add	a6,a6,a5                                                          # executed: 4,096
80000a30:	010e5463          	bge	t3,a6,80000a38 <requantize_i32_to_i8+0x32>                          # executed: 4,096
80000a34:	07f00813          	li	a6,127                                                               # executed: 0
80000a38:	00685463          	bge	a6,t1,80000a40 <requantize_i32_to_i8+0x3a>                          # executed: 4,096
80000a3c:	f8000813          	li	a6,-128                                                              # executed: 0
80000a40:	01058023          	sb	a6,0(a1)                                                             # executed: 4,096
80000a44:	0585                	addi	a1,a1,1                                                          # executed: 4,096
80000a46:	fca61ce3          	bne	a2,a0,80000a1e <requantize_i32_to_i8+0x18>                          # executed: 4,096
80000a4a:	8082                	ret                                                                   # executed: 1
```


### 4.2 `requantize_i8_to_i32()` — Align the quantized branch to the residual scale

Compute $(((x-z)M)+2^{s-1})\gg s$ from signed INT8 into INT32. Unlike the INT32-to-INT8 helper, its zero point is an **input** zero point and is subtracted before multiplication. There is no INT8 clamp. The nine-instruction loop `0x80000a5c–0x80000a76` has one load, two pointer updates, one subtract, one multiply, one rounding add, one shift, one store, and one branch. $I=9N+7=36{,}871$ for 4,096 elements.

#### Corresponding C

```c
void requantize_i8_to_i32(
    const int8_t *input,
    int32_t *output,
    size_t count,
    int64_t multiplier,
    unsigned int shift,
    int32_t input_zero_point
)
{
    const int64_t rounding = INT64_C(1) << (shift - 1u);

    for (size_t i = 0; i < count; ++i) {
        const int64_t centered =
            (int64_t)input[i] - input_zero_point;

        output[i] = (int32_t)(
            (centered * multiplier + rounding) >> shift
        );
    }
}
```


#### Compiled assembly

```asm
0000000080000a4c <requantize_i8_to_i32>:
80000a4c:	c61d                	beqz	a2,80000a7a <requantize_i8_to_i32+0x2e>                          # executed: 1
80000a4e:	060a                	slli	a2,a2,0x2                                                        # executed: 1
80000a50:	fff7081b          	addiw	a6,a4,-1                                                          # executed: 1
80000a54:	4885                	li	a7,1                                                               # executed: 1
80000a56:	962e                	add	a2,a2,a1                                                          # executed: 1
80000a58:	010898b3          	sll	a7,a7,a6                                                            # executed: 1
80000a5c:	00050803          	lb	a6,0(a0)                                                             # executed: 4,096
80000a60:	0591                	addi	a1,a1,4                                                          # executed: 4,096
80000a62:	0505                	addi	a0,a0,1                                                          # executed: 4,096
80000a64:	40f80833          	sub	a6,a6,a5                                                            # executed: 4,096
80000a68:	02d80833          	mul	a6,a6,a3                                                            # executed: 4,096
80000a6c:	9846                	add	a6,a6,a7                                                          # executed: 4,096
80000a6e:	40e85833          	sra	a6,a6,a4                                                            # executed: 4,096
80000a72:	ff05ae23          	sw	a6,-4(a1)                                                            # executed: 4,096
80000a76:	feb613e3          	bne	a2,a1,80000a5c <requantize_i8_to_i32+0x10>                          # executed: 4,096
80000a7a:	8082                	ret                                                                   # executed: 1
```


## 5. Complete Instruction Accounting

### 5.1 Loop and computation counts

The projection performs **524,288 mathematical MACs** and produces 4,096 outputs. Its biased formula is

$$
24+32\{9+128[12+8(128)]\}=4,243,768.
$$

| Inner-loop purpose | Instructions per MAC iteration | Executed instructions |
|---|---|---|
| MAC multiplication (`mulw`) | 1 | 524,288 |
| MAC accumulation (`addw`) | 1 | 524,288 |
| Input centering (`subw`) | 1 | 524,288 |
| Input and weight loads (`lb`) | 2 | 1,048,576 |
| Pointer updates (`addi`) | 2 | 1,048,576 |
| Inner-loop branch (`bne`) | 1 | 524,288 |
| **Inner-loop total** | **8** | **4,194,304** |


Outside the inner loop: 24 entry/exit instructions, $32\times9=288$ token instructions, and $4,096\times12=49,152$ output instructions, totaling **49,464**. Of those output instructions, exactly **8,192** are the bias load and bias addition. Thus the accumulation kernel's full count is **4,243,768**.

One mathematical MAC remains one multiplication plus one accumulator addition. Eight describes the complete scalar loop iteration. Its 75% supporting-instruction share does not by itself establish a memory-bandwidth bottleneck.

### 5.2 Requantization counts

| Conversion | Elements | Formula | C, case 0 | Executed instructions |
|---|---|---|---|---|
| T71 conversion | 4,096 | 11N + 9 + C | 0 | 45,065 |
| T72 alignment | 4,096 | 9N + 7 | 0 | 36,871 |
| **Total** |  |  |  | **81,936** |

Conversion kernels account for **1.88%** of this stage body in case 0. Computation kernels, conversion kernels, and wrapper instructions are counted separately.

### 5.3 Function totals and instruction breakdown

| Order | Work | Function | Case-0 instructions | Share of stage body |
|---|---|---|---|---|
| 0 | wrapper | `block0_out_residual_forward` | 49 | 0.0011% |
| 1 | output accumulation | `projection_accumulate_i8` | 4,243,768 | 97.1844% |
| 2 | T71 conversion | `requantize_i32_to_i8` | 45,065 | 1.0320% |
| 3 | T72 alignment | `requantize_i8_to_i32` | 36,871 | 0.8444% |
| 4 | T0 + T72 residual | `residual_add_i8_i32` | 40,964 | 0.9381% |
|  | **Total, including callees** |  | **4,366,717** | **100%** |

Across this stage, multiplication instructions (`mul` + `mulw`) total **536,576**. Register additions (`add` + `addw`) total **553,093**. These opcode totals include address/setup arithmetic where the compiler uses those mnemonics; they are not counts of mathematical MACs. Immediate additions (`addi` + `addiw`) are listed separately below.

| Disassembly mnemonic | Executed, case 0 | Function in this stage |
|---|---|---|
| `beqz` | 8,197 | Zero-size checks or zero-bit tests. |
| `addi` | 1,085,480 | Add an immediate: pointer/loop/stack updates, plus immediate numerical constants. |
| `sd` | 10 | Store INT64 intermediates or save registers. |
| `mv` | 4,145 | Copy a pointer/value or prepare call arguments; also select attention maxima. |
| `mul` | 12,288 | 64-bit low-product arithmetic: requantization, scaling, normalization, and occasional dimension setup. |
| `add` | 20,613 | 64-bit addition: numerical arithmetic and address/index calculations. |
| `slli` | 36 | Immediate left shift: byte offsets, alignment, or integer constants. |
| `li` | 4,143 | Load a constant, initialize a loop/accumulator, or assign a saturation endpoint. |
| `lb` | 1,056,768 | Load a signed INT8 activation or weight. |
| `subw` | 524,288 | 32-bit subtraction: input centering or attention score shifting. |
| `mulw` | 524,288 | 32-bit low-product arithmetic: projection/attention MAC or GELU products. |
| `addw` | 532,480 | 32-bit addition: accumulation, residual/elementwise arithmetic, or rounding-independent sums. |
| `bne` | 540,704 | Continue a counted pointer/index loop. |
| `ld` | 10 | Load INT64 scratch data, pointers, or saved registers. |
| `sw` | 12,288 | Store an INT32 result, intermediate, or probability. |
| `ret` | 4 | Return to the caller; charged to the returning helper. |
| `j` | 1 | Unconditional jump, including a tail call or loop transfer. |
| `lw` | 12,288 | Load a signed INT32 accumulator, probability, parameter, or LUT value. |
| `bge` | 8,192 | Signed comparisons for clipping, maximum selection, or the exponential cutoff. |
| `addiw` | 2 | 32-bit immediate addition: rounding-shift setup or LUT index calculation. |
| `sll` | 2 | Register-controlled left shift: rounding constants or probability scaling. |
| `lui` | 5 | Construct a constant or high part of an address. |
| `sub` | 8,192 | 64-bit subtraction: centering, normalization, interpolation, square-root work, or row addressing. |
| `sra` | 8,192 | Arithmetic right shift: rounded fixed-point conversion. |
| `sb` | 4,096 | Store an INT8 activation. |
| `jal` | 3 | Call a helper; its instruction is charged to the caller. |
| `auipc` | 2 | Construct a PC-relative parameter/table address. |
| **Total** | **4,366,717** |  |

### 5.4 Reconciliation with `minstret`

The supplied Spike counter interval contains **7 instructions outside the stage body**: the first counter read, argument setup, and the call. The ending read has not yet retired when its value is sampled. Instructions that subtract, save, or print the delta are outside this interval. This is specific to the supplied compiled call site.

```asm
80001a8e:	b0202af3          	csrr	s5,minstret
80001a92:	6766                	ld	a4,88(sp)
80001a94:	76ea                	ld	a3,184(sp)
80001a96:	662e                	ld	a2,200(sp)
80001a98:	754a                	ld	a0,176(sp)
80001a9a:	85ca                	mv	a1,s2
80001a9c:	b56ff0ef          	jal	80000df2 <block0_out_residual_forward>
80001aa0:	b02027f3          	csrr	a5,minstret
```


| Case | Supplied raw minstret | Caller/measurement overhead | Counted stage body + callees | Unexplained difference |
|---|---|---|---|---|
| 0 | 4,366,724 | 7 | 4,366,717 | 0 |
| 1 | 4,366,724 | 7 | 4,366,717 | 0 |
| 2 | 4,366,724 | 7 | 4,366,717 | 0 |
| 3 | 4,366,725 | 7 | 4,366,718 | 0 |
| 4 | 4,366,724 | 7 | 4,366,717 | 0 |
| 5 | 4,366,725 | 7 | 4,366,718 | 0 |
| 6 | 4,366,724 | 7 | 4,366,717 | 0 |
| 7 | 4,366,724 | 7 | 4,366,717 | 0 |
| **Mean** | 4,366,724.25 | 7 | 4,366,717.25 | **0** |


Every case reconciles exactly. The body count includes all helper calls and their returns, and each wrapper instruction once per actual execution. The separately run staged and full paths both passed the supplied T107 comparisons.

| Conversion call | C0 clips | C1 clips | C2 clips | C3 clips | C4 clips | C5 clips | C6 clips | C7 clips |
|---|---|---|---|---|---|---|---|---|
| T71 conversion | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |

These are clipping assignments executed during the semantic replay of each helper call, not clipping counts guessed from the stage residual. Each clipped element adds one instruction in this binary.

## 6. Findings

| Region | Case-0 instructions | Share of stage body |
|---|---|---|
| Computation | 4,284,732 | 98.12% |
| Requantization | 81,936 | 1.88% |
| Wrapper | 49 | 0.00% |

The projection accumulation accounts for **97.18%** of this stage. Its complete scalar MAC loop dominates, including loads, centering, pointer updates, and control. The remaining conversions and residual addition have much smaller instruction counts. This stage is about **7.64%** of full-block raw instructions.

Instruction shares identify where instructions are executed. They do not establish load latency, multiplier latency, cache behavior, or CV-Wally cycle speedup. No accelerator or custom-instruction speedup is measured here. The eight fixture passes validate these supplied integer cases; they do not establish equality with the original mixed TFLite graph or unseen-data quality.

[Back to results](Result.md) · [Counting method and validation](analysis_method.md)
