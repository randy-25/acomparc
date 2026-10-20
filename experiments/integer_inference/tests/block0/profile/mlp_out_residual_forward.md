# Part 7: MLP Output Projection and Second Residual — Disassembly Analysis

## 1. Function Overview

This note analyzes `block0_mlp_out_residual_forward()` and every helper it actually executes in the **supplied compiled ELF**. Case 0 provides the detailed instruction mix; Section 5.4 reconciles all eight cases with the supplied Spike `minstret` results. Counts describe dynamic executions, not the number of lines in the disassembly. See [counting method](analysis_method.md).

| Parameter | Current block-0 value |
|---|---|
| projection T / K / O | 32 / 512 / 128 |
| bias / input zero point | 128 INT32 elements / -88 |
| T105 multiplier / shift / zero point | 641,059,315 / 39 / 10 |
| T106 multiplier / shift / input zero point | 720,118,272 / 22 / 10 |
| residual input / final output | T73 / T107, both INT32 [32,128], scale 2^-16 |


## 2. Block Wrapper: `block0_mlp_out_residual_forward()`

### 2.1 Source and operation order

```c
void block0_mlp_out_residual_forward(
    const int8_t *input,
    const int32_t *residual_input,
    int32_t *output,
    int8_t *projection_workspace,
    int32_t *accumulator_workspace
)
{
    const size_t count =
        BLOCK0_TOKEN_COUNT * BLOCK0_MLP_OUT_OUTPUT_FEATURES;

    /* Op 71: 512 -> 128 projection, including bias. */
    projection_accumulate_i8(
        input,
        block0_mlp_out_weights,
        block0_mlp_out_bias,
        accumulator_workspace,
        BLOCK0_TOKEN_COUNT,
        BLOCK0_MLP_OUT_INPUT_FEATURES,
        BLOCK0_MLP_OUT_OUTPUT_FEATURES,
        BLOCK0_MLP_OUT_INPUT_ZERO_POINT
    );

    /* Preserve the INT8 T105 boundary. */
    requantize_i32_to_i8(
        accumulator_workspace,
        projection_workspace,
        count,
        BLOCK0_MLP_OUT_MULTIPLIER,
        BLOCK0_MLP_OUT_SHIFT,
        BLOCK0_MLP_OUT_OUTPUT_ZERO_POINT
    );

    /* Op 72 counterpart: T105 -> INT32 T106, scale 2^-16. */
    requantize_i8_to_i32(
        projection_workspace,
        accumulator_workspace,
        count,
        BLOCK0_RESIDUAL2_BRANCH_MULTIPLIER,
        BLOCK0_RESIDUAL2_BRANCH_SHIFT,
        BLOCK0_MLP_OUT_OUTPUT_ZERO_POINT
    );

    /* Op 73: T106 + retained T73 -> T107. */
    residual_add_i32(
        accumulator_workspace,
        residual_input,
        output,
        count
    );
}
```


### 2.2 Compiled wrapper

The assembly below includes the wrapper only. Each `jal` or tail `j` transfers execution to a helper whose work is counted separately. `executed` is the case-0 execution frequency of that instruction. A helper reused several times has aggregated frequencies in its own listing. Unexecuted alternate paths are shown with zero.

```asm
000000008000113a <block0_mlp_out_residual_forward>:
8000113a:	7179                	addi	sp,sp,-48                                                        # executed: 1
8000113c:	f022                	sd	s0,32(sp)                                                          # executed: 1
8000113e:	ec26                	sd	s1,24(sp)                                                          # executed: 1
80001140:	e84a                	sd	s2,16(sp)                                                          # executed: 1
80001142:	e44e                	sd	s3,8(sp)                                                           # executed: 1
80001144:	843a                	mv	s0,a4                                                              # executed: 1
80001146:	84b6                	mv	s1,a3                                                              # executed: 1
80001148:	89ae                	mv	s3,a1                                                              # executed: 1
8000114a:	8932                	mv	s2,a2                                                              # executed: 1
8000114c:	fa800893          	li	a7,-88                                                               # executed: 1
80001150:	86ba                	mv	a3,a4                                                              # executed: 1
80001152:	08000813          	li	a6,128                                                               # executed: 1
80001156:	20000793          	li	a5,512                                                               # executed: 1
8000115a:	02000713          	li	a4,32                                                                # executed: 1
8000115e:	00001617          	auipc	a2,0x1                                                            # executed: 1
80001162:	df260613          	addi	a2,a2,-526 # 80001f50 <block0_mlp_out_bias>                        # executed: 1
80001166:	00001597          	auipc	a1,0x1                                                            # executed: 1
8000116a:	fea58593          	addi	a1,a1,-22 # 80002150 <block0_mlp_out_weights>                      # executed: 1
8000116e:	f406                	sd	ra,40(sp)                                                          # executed: 1
80001170:	96dff0ef          	jal	80000adc <projection_accumulate_i8>                                 # executed: 1
80001174:	2635d6b7          	lui	a3,0x2635d                                                          # executed: 1
80001178:	85a6                	mv	a1,s1                                                              # executed: 1
8000117a:	8522                	mv	a0,s0                                                              # executed: 1
8000117c:	9f368693          	addi	a3,a3,-1549 # 2635c9f3 <_start-0x59ca360d>                         # executed: 1
80001180:	47a9                	li	a5,10                                                              # executed: 1
80001182:	02700713          	li	a4,39                                                                # executed: 1
80001186:	6605                	lui	a2,0x1                                                            # executed: 1
80001188:	87fff0ef          	jal	80000a06 <requantize_i32_to_i8>                                     # executed: 1
8000118c:	2aec26b7          	lui	a3,0x2aec2                                                          # executed: 1
80001190:	8526                	mv	a0,s1                                                              # executed: 1
80001192:	85a2                	mv	a1,s0                                                              # executed: 1
80001194:	20068693          	addi	a3,a3,512 # 2aec2200 <_start-0x5513de00>                           # executed: 1
80001198:	6605                	lui	a2,0x1                                                            # executed: 1
8000119a:	47a9                	li	a5,10                                                              # executed: 1
8000119c:	4759                	li	a4,22                                                              # executed: 1
8000119e:	8afff0ef          	jal	80000a4c <requantize_i8_to_i32>                                     # executed: 1
800011a2:	8522                	mv	a0,s0                                                              # executed: 1
800011a4:	7402                	ld	s0,32(sp)                                                          # executed: 1
800011a6:	70a2                	ld	ra,40(sp)                                                          # executed: 1
800011a8:	64e2                	ld	s1,24(sp)                                                          # executed: 1
800011aa:	864a                	mv	a2,s2                                                              # executed: 1
800011ac:	85ce                	mv	a1,s3                                                              # executed: 1
800011ae:	6942                	ld	s2,16(sp)                                                          # executed: 1
800011b0:	69a2                	ld	s3,8(sp)                                                           # executed: 1
800011b2:	6685                	lui	a3,0x1                                                            # executed: 1
800011b4:	6145                	addi	sp,sp,48                                                         # executed: 1
800011b6:	bacd                	j	80000ba8 <residual_add_i32>                                         # executed: 1
```


The wrapper itself executes **47 instructions**, excluding all callee bodies. Stack saves retain registers and pointers across calls; the final helper may be tail-called after restoration. A tail call is counted as one wrapper jump followed by the helper's body and return, not a second wrapper epilogue.

#### Incoming registers

| Register on entry | C argument |
|---|---|
| `a0` | const int8_t *input |
| `a1` | const int32_t *residual_input |
| `a2` | int32_t *output |
| `a3` | int8_t *projection_workspace |
| `a4` | int32_t *accumulator_workspace |

#### Retained values and stack frame

| Register | Role in this wrapper |
|---|---|
| `s0` | INT32 accumulator workspace, later T106 |
| `s1` | INT8 projection workspace for T105 |
| `s2` | Final INT32 T107 output pointer |
| `s3` | Retained INT32 T73 residual pointer |

All six wrappers analyzed in the new notes allocate 48 bytes. They save `ra` at `40(sp)`, `s0` at `32(sp)`, `s1` at `24(sp)`, `s2` at `16(sp)`, and `s3` at `8(sp)`: 40 bytes of saved values plus eight bytes unused in this frame, keeping the 16-byte stack alignment. These saved slots hold register values, not activation tensors. The callee-saved registers retain values across helper calls, and the wrapper restores the caller's original values before its final tail transfer.

### 2.3 Helper arguments and data boundaries

Pointer arguments select the source tensors and caller-owned scratch shown in the C wrapper. Non-pointer arguments decoded at each actual call are:

| Call in source order | Actual non-pointer arguments |
|---|---|
| output accumulation | T=32, K=512, O=128, input z=-88; bias present |
| T105 conversion | N=4096, M=641,059,315, shift=39, output z=10 |
| T106 alignment | N=4096, M=720,118,272, shift=22, input z=10 |
| T73 + T106 residual | N=4096 |

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
80000b1a:	00050303          	lb	t1,0(a0)                                                             # executed: 2,097,152
80000b1e:	000e0803          	lb	a6,0(t3)                                                             # executed: 2,097,152
80000b22:	0505                	addi	a0,a0,1                                                          # executed: 2,097,152
80000b24:	4113033b          	subw	t1,t1,a7                                                           # executed: 2,097,152
80000b28:	0268083b          	mulw	a6,a6,t1                                                           # executed: 2,097,152
80000b2c:	0e05                	addi	t3,t3,1                                                          # executed: 2,097,152
80000b2e:	01d80ebb          	addw	t4,a6,t4                                                           # executed: 2,097,152
80000b32:	ffe514e3          	bne	a0,t5,80000b1a <projection_accumulate_i8+0x3e>                      # executed: 2,097,152
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


### 3.2 `residual_add_i32()` — Second residual addition

Add retained INT32 T73 and INT32 T106 at the same scale $2^{-16}$, then store final INT32 T107. There is one numerical `addw` per element and no multiplier or requantization inside this helper. Two loads, three pointer updates, one add, one store, and one branch give eight instructions per element; $I=8N+4=32{,}772$.

#### Corresponding C

```c
void residual_add_i32(
    const int32_t *left,
    const int32_t *right,
    int32_t *output,
    size_t count
)
{
    for (size_t i = 0; i < count; ++i) {
        const int64_t sum = (int64_t)left[i] + right[i];

        output[i] = (int32_t)sum;
    }
}
```


#### Compiled assembly

```asm
0000000080000ba8 <residual_add_i32>:
80000ba8:	ce81                	beqz	a3,80000bc0 <residual_add_i32+0x18>                              # executed: 1
80000baa:	068a                	slli	a3,a3,0x2                                                        # executed: 1
80000bac:	96aa                	add	a3,a3,a0                                                          # executed: 1
80000bae:	4118                	lw	a4,0(a0)                                                           # executed: 4,096
80000bb0:	419c                	lw	a5,0(a1)                                                           # executed: 4,096
80000bb2:	0511                	addi	a0,a0,4                                                          # executed: 4,096
80000bb4:	0591                	addi	a1,a1,4                                                          # executed: 4,096
80000bb6:	9fb9                	addw	a5,a5,a4                                                         # executed: 4,096
80000bb8:	c21c                	sw	a5,0(a2)                                                           # executed: 4,096
80000bba:	0611                	addi	a2,a2,4                                                          # executed: 4,096
80000bbc:	fed519e3          	bne	a0,a3,80000bae <residual_add_i32+0x6>                               # executed: 4,096
80000bc0:	8082                	ret                                                                   # executed: 1
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
80000a34:	07f00813          	li	a6,127                                                               # executed: 2
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

The projection performs **2,097,152 mathematical MACs** and produces 4,096 outputs. Its biased formula is

$$
24+32\{9+128[12+8(512)]\}=16,826,680.
$$

| Inner-loop purpose | Instructions per MAC iteration | Executed instructions |
|---|---|---|
| MAC multiplication (`mulw`) | 1 | 2,097,152 |
| MAC accumulation (`addw`) | 1 | 2,097,152 |
| Input centering (`subw`) | 1 | 2,097,152 |
| Input and weight loads (`lb`) | 2 | 4,194,304 |
| Pointer updates (`addi`) | 2 | 4,194,304 |
| Inner-loop branch (`bne`) | 1 | 2,097,152 |
| **Inner-loop total** | **8** | **16,777,216** |


Outside the inner loop: 24 entry/exit instructions, $32\times9=288$ token instructions, and $4,096\times12=49,152$ output instructions, totaling **49,464**. Of those output instructions, exactly **8,192** are the bias load and bias addition. Thus the accumulation kernel's full count is **16,826,680**.

One mathematical MAC remains one multiplication plus one accumulator addition. Eight describes the complete scalar loop iteration. Its 75% supporting-instruction share does not by itself establish a memory-bandwidth bottleneck.

### 5.2 Requantization counts

| Conversion | Elements | Formula | C, case 0 | Executed instructions |
|---|---|---|---|---|
| T105 conversion | 4,096 | 11N + 9 + C | 2 | 45,067 |
| T106 alignment | 4,096 | 9N + 7 | 0 | 36,871 |
| **Total** |  |  |  | **81,938** |

Conversion kernels account for **0.48%** of this stage body in case 0. Computation kernels, conversion kernels, and wrapper instructions are counted separately.

### 5.3 Function totals and instruction breakdown

| Order | Work | Function | Case-0 instructions | Share of stage body |
|---|---|---|---|---|
| 0 | wrapper | `block0_mlp_out_residual_forward` | 47 | 0.0003% |
| 1 | output accumulation | `projection_accumulate_i8` | 16,826,680 | 99.3226% |
| 2 | T105 conversion | `requantize_i32_to_i8` | 45,067 | 0.2660% |
| 3 | T106 alignment | `requantize_i8_to_i32` | 36,871 | 0.2176% |
| 4 | T73 + T106 residual | `residual_add_i32` | 32,772 | 0.1934% |
|  | **Total, including callees** |  | **16,941,437** | **100%** |

Across this stage, multiplication instructions (`mul` + `mulw`) total **2,105,344**. Register additions (`add` + `addw`) total **2,125,957**. These opcode totals include address/setup arithmetic where the compiler uses those mnemonics; they are not counts of mathematical MACs. Immediate additions (`addi` + `addiw`) are listed separately below.

| Disassembly mnemonic | Executed, case 0 | Function in this stage |
|---|---|---|
| `beqz` | 8,197 | Zero-size checks or zero-bit tests. |
| `addi` | 4,231,208 | Add an immediate: pointer/loop/stack updates, plus immediate numerical constants. |
| `sd` | 10 | Store INT64 intermediates or save registers. |
| `mv` | 4,144 | Copy a pointer/value or prepare call arguments; also select attention maxima. |
| `mul` | 8,192 | 64-bit low-product arithmetic: requantization, scaling, normalization, and occasional dimension setup. |
| `add` | 20,613 | 64-bit addition: numerical arithmetic and address/index calculations. |
| `slli` | 36 | Immediate left shift: byte offsets, alignment, or integer constants. |
| `li` | 4,144 | Load a constant, initialize a loop/accumulator, or assign a saturation endpoint. |
| `lb` | 4,198,400 | Load a signed INT8 activation or weight. |
| `subw` | 2,097,152 | 32-bit subtraction: input centering or attention score shifting. |
| `mulw` | 2,097,152 | 32-bit low-product arithmetic: projection/attention MAC or GELU products. |
| `addw` | 2,105,344 | 32-bit addition: accumulation, residual/elementwise arithmetic, or rounding-independent sums. |
| `bne` | 2,113,568 | Continue a counted pointer/index loop. |
| `ld` | 10 | Load INT64 scratch data, pointers, or saved registers. |
| `sw` | 12,288 | Store an INT32 result, intermediate, or probability. |
| `ret` | 4 | Return to the caller; charged to the returning helper. |
| `j` | 1 | Unconditional jump, including a tail call or loop transfer. |
| `lw` | 16,384 | Load a signed INT32 accumulator, probability, parameter, or LUT value. |
| `bge` | 8,192 | Signed comparisons for clipping, maximum selection, or the exponential cutoff. |
| `addiw` | 2 | 32-bit immediate addition: rounding-shift setup or LUT index calculation. |
| `sll` | 2 | Register-controlled left shift: rounding constants or probability scaling. |
| `lui` | 5 | Construct a constant or high part of an address. |
| `sub` | 4,096 | 64-bit subtraction: centering, normalization, interpolation, square-root work, or row addressing. |
| `sra` | 8,192 | Arithmetic right shift: rounded fixed-point conversion. |
| `sb` | 4,096 | Store an INT8 activation. |
| `jal` | 3 | Call a helper; its instruction is charged to the caller. |
| `auipc` | 2 | Construct a PC-relative parameter/table address. |
| **Total** | **16,941,437** |  |

### 5.4 Reconciliation with `minstret`

The supplied Spike counter interval contains **8 instructions outside the stage body**: the first counter read, argument setup, and the call. The ending read has not yet retired when its value is sampled. Instructions that subtract, save, or print the delta are outside this interval. This is specific to the supplied compiled call site.

```asm
80001ada:	b0202af3          	csrr	s5,minstret
80001ade:	6766                	ld	a4,88(sp)
80001ae0:	76ea                	ld	a3,184(sp)
80001ae2:	65ae                	ld	a1,200(sp)
80001ae4:	650e                	ld	a0,192(sp)
80001ae6:	001bc617          	auipc	a2,0x1bc
80001aea:	56260613          	addi	a2,a2,1378 # 801be048 <product_workspace>
80001aee:	e4cff0ef          	jal	8000113a <block0_mlp_out_residual_forward>
80001af2:	b02027f3          	csrr	a5,minstret
```


| Case | Supplied raw minstret | Caller/measurement overhead | Counted stage body + callees | Unexplained difference |
|---|---|---|---|---|
| 0 | 16,941,445 | 8 | 16,941,437 | 0 |
| 1 | 16,941,445 | 8 | 16,941,437 | 0 |
| 2 | 16,941,443 | 8 | 16,941,435 | 0 |
| 3 | 16,941,443 | 8 | 16,941,435 | 0 |
| 4 | 16,941,443 | 8 | 16,941,435 | 0 |
| 5 | 16,941,443 | 8 | 16,941,435 | 0 |
| 6 | 16,941,443 | 8 | 16,941,435 | 0 |
| 7 | 16,941,444 | 8 | 16,941,436 | 0 |
| **Mean** | 16,941,443.625 | 8 | 16,941,435.625 | **0** |


Every case reconciles exactly. The body count includes all helper calls and their returns, and each wrapper instruction once per actual execution. The separately run staged and full paths both passed the supplied T107 comparisons.

| Conversion call | C0 clips | C1 clips | C2 clips | C3 clips | C4 clips | C5 clips | C6 clips | C7 clips |
|---|---|---|---|---|---|---|---|---|
| T105 conversion | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 1 |

These are clipping assignments executed during the semantic replay of each helper call, not clipping counts guessed from the stage residual. Each clipped element adds one instruction in this binary.

## 6. Findings

| Region | Case-0 instructions | Share of stage body |
|---|---|---|
| Computation | 16,859,452 | 99.52% |
| Requantization | 81,938 | 0.48% |
| Wrapper | 47 | 0.00% |

The projection accumulation accounts for **99.32%** of this stage. Its complete scalar MAC loop dominates, including loads, centering, pointer updates, and control. The remaining conversions and residual addition have much smaller instruction counts. This stage is about **29.63%** of full-block raw instructions.

Instruction shares identify where instructions are executed. They do not establish load latency, multiplier latency, cache behavior, or CV-Wally cycle speedup. No accelerator or custom-instruction speedup is measured here. The eight fixture passes validate these supplied integer cases; they do not establish equality with the original mixed TFLite graph or unseen-data quality.

[Back to results](Result.md) · [Counting method and validation](analysis_method.md)
