# Part 5: Second Layer Normalization — Disassembly Analysis

## 1. Function Overview

This note analyzes `block0_ln2_forward()` and every helper it actually executes in the **supplied compiled ELF**. Case 0 provides the detailed instruction mix; Section 5.4 reconciles all eight cases with the supplied Spike `minstret` results. Counts describe dynamic executions, not the number of lines in the disassembly. See [counting method](analysis_method.md).

| Parameter | Current block-0 value |
|---|---|
| tokens / features / elements | 32 / 128 / 4,096 |
| input / output | INT32 T73 → INT8 T88 |
| square / normalize shift | 23 / 23 |
| epsilon / rsqrt numerator | 10,737 / 2^62 |
| beta alignment | 65,536 |
| final multiplier / shift / output zero point | 1,009,162 / 47 / 2 |


## 2. Block Wrapper: `block0_ln2_forward()`

### 2.1 Source and operation order

```c
void block0_ln2_forward(
    const int32_t *input,
    int8_t *output,
    LayerNormI8Workspace *workspace
)
{
    const LayerNormI8Params *params = &block0_ln2_params;

    const size_t token_count = BLOCK0_TOKEN_COUNT;
    const size_t feature_count = BLOCK0_HIDDEN_SIZE;
    const size_t count = token_count * feature_count;

    /*
     * Op 43-44: mean sum and broadcast indexing.
     * T74/T75: INT32, scale 2^-23.
     */
    layernorm_sum_i32(
        input,
        workspace->mean_sum,
        token_count,
        feature_count
    );

    /*
     * Op 45: fine squared differences.
     * INT64, scale 2^-46.
     */
    layernorm_squared_difference_i32(
        input,
        workspace->mean_sum,
        workspace->fine_i64,
        token_count,
        feature_count
    );

    /* T76: convert to INT32, scale 2^-23. */
    requantize_shift_i64_to_i32(
        workspace->fine_i64,
        workspace->feature_i32,
        count,
        params->square_shift
    );

    /*
     * Op 46: store the feature sum as variance.
     * T77: INT32, scale 2^-30.
     */
    layernorm_variance_i32(
        workspace->feature_i32,
        workspace->variance,
        token_count,
        feature_count
    );

    /* Op 47: T78, variance plus epsilon, scale 2^-30. */
    layernorm_add_epsilon_i32(
        workspace->variance,
        workspace->variance,
        token_count,
        params->epsilon_q
    );

    /* Op 48: T79, inverse standard deviation, scale 2^-16. */
    layernorm_rsqrt_i32(
        workspace->variance,
        workspace->inv_std,
        token_count,
        params->rsqrt_numerator
    );

    /*
     * Op 49-54: normalized features before conversion.
     * Reuse fine_i64; its squared differences are no longer needed.
     * Fine T85: INT64, scale 2^-39.
     */
    layernorm_normalize_i32(
        input,
        workspace->mean_sum,
        workspace->inv_std,
        workspace->fine_i64,
        token_count,
        feature_count
    );

    /*
     * T85: convert to INT32, scale 2^-16.
     * Reuse feature_i32; squared differences are no longer needed.
     */
    requantize_shift_i64_to_i32(
        workspace->fine_i64,
        workspace->feature_i32,
        count,
        params->normalize_shift
    );

    /*
     * Op 55-56: gamma multiplication and aligned beta addition.
     * T87: INT64, scale 2^-32.
     */
    layernorm_affine_i32(
        workspace->feature_i32,
        params->gamma,
        params->beta,
        workspace->fine_i64,
        token_count,
        feature_count,
        params->beta_alignment
    );

    /* Op 57: T88, INT8 MLP input. */
    requantize_i64_to_i8(
        workspace->fine_i64,
        output,
        count,
        params->output_multiplier,
        params->output_shift,
        params->output_zero_point
    );
}
```


### 2.2 Compiled wrapper

The assembly below includes the wrapper only. Each `jal` or tail `j` transfers execution to a helper whose work is counted separately. `executed` is the case-0 execution frequency of that instruction. A helper reused several times has aggregated frequencies in its own listing. Unexecuted alternate paths are shown with zero.

```asm
0000000080000e72 <block0_ln2_forward>:
80000e72:	7179                	addi	sp,sp,-48                                                        # executed: 1
80000e74:	e44e                	sd	s3,8(sp)                                                           # executed: 1
80000e76:	89ae                	mv	s3,a1                                                              # executed: 1
80000e78:	620c                	ld	a1,0(a2)                                                           # executed: 1
80000e7a:	f022                	sd	s0,32(sp)                                                          # executed: 1
80000e7c:	08000693          	li	a3,128                                                               # executed: 1
80000e80:	8432                	mv	s0,a2                                                              # executed: 1
80000e82:	02000613          	li	a2,32                                                                # executed: 1
80000e86:	f406                	sd	ra,40(sp)                                                          # executed: 1
80000e88:	ec26                	sd	s1,24(sp)                                                          # executed: 1
80000e8a:	e84a                	sd	s2,16(sp)                                                          # executed: 1
80000e8c:	892a                	mv	s2,a0                                                              # executed: 1
80000e8e:	a11ff0ef          	jal	8000089e <layernorm_sum_i32>                                        # executed: 1
80000e92:	7010                	ld	a2,32(s0)                                                          # executed: 1
80000e94:	600c                	ld	a1,0(s0)                                                           # executed: 1
80000e96:	08000713          	li	a4,128                                                               # executed: 1
80000e9a:	854a                	mv	a0,s2                                                              # executed: 1
80000e9c:	02000693          	li	a3,32                                                                # executed: 1
80000ea0:	a45ff0ef          	jal	800008e4 <layernorm_squared_difference_i32>                         # executed: 1
80000ea4:	00022497          	auipc	s1,0x22                                                           # executed: 1
80000ea8:	eac48493          	addi	s1,s1,-340 # 80022d50 <block0_ln2_params>                          # executed: 1
80000eac:	6c0c                	ld	a1,24(s0)                                                          # executed: 1
80000eae:	7008                	ld	a0,32(s0)                                                          # executed: 1
80000eb0:	4894                	lw	a3,16(s1)                                                          # executed: 1
80000eb2:	6605                	lui	a2,0x1                                                            # executed: 1
80000eb4:	ae7ff0ef          	jal	8000099a <requantize_shift_i64_to_i32>                              # executed: 1
80000eb8:	640c                	ld	a1,8(s0)                                                           # executed: 1
80000eba:	6c08                	ld	a0,24(s0)                                                          # executed: 1
80000ebc:	08000693          	li	a3,128                                                               # executed: 1
80000ec0:	02000613          	li	a2,32                                                                # executed: 1
80000ec4:	861ff0ef          	jal	80000724 <layernorm_variance_i32>                                   # executed: 1
80000ec8:	640c                	ld	a1,8(s0)                                                           # executed: 1
80000eca:	48d4                	lw	a3,20(s1)                                                          # executed: 1
80000ecc:	02000613          	li	a2,32                                                                # executed: 1
80000ed0:	852e                	mv	a0,a1                                                              # executed: 1
80000ed2:	899ff0ef          	jal	8000076a <layernorm_add_epsilon_i32>                                # executed: 1
80000ed6:	6c94                	ld	a3,24(s1)                                                          # executed: 1
80000ed8:	680c                	ld	a1,16(s0)                                                          # executed: 1
80000eda:	6408                	ld	a0,8(s0)                                                           # executed: 1
80000edc:	02000613          	li	a2,32                                                                # executed: 1
80000ee0:	8a3ff0ef          	jal	80000782 <layernorm_rsqrt_i32>                                      # executed: 1
80000ee4:	7014                	ld	a3,32(s0)                                                          # executed: 1
80000ee6:	6810                	ld	a2,16(s0)                                                          # executed: 1
80000ee8:	600c                	ld	a1,0(s0)                                                           # executed: 1
80000eea:	854a                	mv	a0,s2                                                              # executed: 1
80000eec:	08000793          	li	a5,128                                                               # executed: 1
80000ef0:	02000713          	li	a4,32                                                                # executed: 1
80000ef4:	a49ff0ef          	jal	8000093c <layernorm_normalize_i32>                                  # executed: 1
80000ef8:	5094                	lw	a3,32(s1)                                                          # executed: 1
80000efa:	6c0c                	ld	a1,24(s0)                                                          # executed: 1
80000efc:	7008                	ld	a0,32(s0)                                                          # executed: 1
80000efe:	6605                	lui	a2,0x1                                                            # executed: 1
80000f00:	a9bff0ef          	jal	8000099a <requantize_shift_i64_to_i32>                              # executed: 1
80000f04:	7014                	ld	a3,32(s0)                                                          # executed: 1
80000f06:	6490                	ld	a2,8(s1)                                                           # executed: 1
80000f08:	608c                	ld	a1,0(s1)                                                           # executed: 1
80000f0a:	6c08                	ld	a0,24(s0)                                                          # executed: 1
80000f0c:	0284b803          	ld	a6,40(s1)                                                            # executed: 1
80000f10:	08000793          	li	a5,128                                                               # executed: 1
80000f14:	02000713          	li	a4,32                                                                # executed: 1
80000f18:	917ff0ef          	jal	8000082e <layernorm_affine_i32>                                     # executed: 1
80000f1c:	7008                	ld	a0,32(s0)                                                          # executed: 1
80000f1e:	7402                	ld	s0,32(sp)                                                          # executed: 1
80000f20:	5cdc                	lw	a5,60(s1)                                                          # executed: 1
80000f22:	5c98                	lw	a4,56(s1)                                                          # executed: 1
80000f24:	7894                	ld	a3,48(s1)                                                          # executed: 1
80000f26:	70a2                	ld	ra,40(sp)                                                          # executed: 1
80000f28:	64e2                	ld	s1,24(sp)                                                          # executed: 1
80000f2a:	6942                	ld	s2,16(sp)                                                          # executed: 1
80000f2c:	85ce                	mv	a1,s3                                                              # executed: 1
80000f2e:	69a2                	ld	s3,8(sp)                                                           # executed: 1
80000f30:	6605                	lui	a2,0x1                                                            # executed: 1
80000f32:	6145                	addi	sp,sp,48                                                         # executed: 1
80000f34:	b471                	j	800009c0 <requantize_i64_to_i8>                                     # executed: 1
```


The wrapper itself executes **74 instructions**, excluding all callee bodies. Stack saves retain registers and pointers across calls; the final helper may be tail-called after restoration. A tail call is counted as one wrapper jump followed by the helper's body and return, not a second wrapper epilogue.

#### Incoming registers

| Register on entry | C argument |
|---|---|
| `a0` | const int32_t *input |
| `a1` | int8_t *output |
| `a2` | LayerNormI8Workspace *workspace |

#### Retained values and stack frame

| Register | Role in this wrapper |
|---|---|
| `s0` | LayerNormI8Workspace pointer |
| `s1` | block0_ln2_params pointer, constructed before the first conversion |
| `s2` | Retained INT32 T73 input pointer |
| `s3` | Final INT8 T88 output pointer |

All six wrappers analyzed in the new notes allocate 48 bytes. They save `ra` at `40(sp)`, `s0` at `32(sp)`, `s1` at `24(sp)`, `s2` at `16(sp)`, and `s3` at `8(sp)`: 40 bytes of saved values plus eight bytes unused in this frame, keeping the 16-byte stack alignment. These saved slots hold register values, not activation tensors. The callee-saved registers retain values across helper calls, and the wrapper restores the caller's original values before its final tail transfer.

Workspace pointer fields are at offsets 0 (`mean_sum`), 8 (`variance`), 16 (`inv_std`), 24 (`feature_i32`), and 32 (`fine_i64`) from `s0`. Thus `ld a0,32(s0)` passes the INT64 scratch pointer; it does not load a tensor value. The parameter object similarly supplies gamma/beta pointers and conversion constants.

### 2.3 Helper arguments and data boundaries

Pointer arguments select the source tensors and caller-owned scratch shown in the C wrapper. Non-pointer arguments decoded at each actual call are:

| Call in source order | Actual non-pointer arguments |
|---|---|
| mean sums | T=32, F=128 |
| squared differences | T=32, F=128 |
| square conversion | N=4096, shift=23 |
| variance sums | T=32, F=128 |
| epsilon | T=32, epsilon=10,737 |
| inverse standard deviation | T=32, numerator=4,611,686,018,427,387,904 |
| normalize | T=32, F=128 |
| normalize conversion | N=4096, shift=23 |
| affine transform | T=32, F=128, beta alignment=65,536 |
| T88 conversion | N=4096, M=1,009,162, shift=47, output z=2 |

## 3. Computation Kernels

### 3.1 `layernorm_sum_i32()` — Mean numerator

The INT32 input uses `lw` and 64-bit `add` for its sum. The four-instruction feature loop is `0x800008b6–0x800008bc`. Six entry instructions, six other instructions per token, and one return give $7+T(6+4F)=16{,}583$. The byte stride is four instead of one.

#### Corresponding C

```c
void layernorm_sum_i32(
    const int32_t *input,
    int32_t *output,
    size_t token_count,
    size_t feature_count
)
{
    for (size_t t = 0; t < token_count; ++t) {
        const int32_t *input_row = input + t * feature_count;
        int64_t sum = 0;

        for (size_t k = 0; k < feature_count; ++k) {
            sum += (int64_t)input_row[k];
        }

        output[t] = (int32_t)sum;
    }
}
```


#### Compiled assembly

```asm
000000008000089e <layernorm_sum_i32>:
8000089e:	c615                	beqz	a2,800008ca <layernorm_sum_i32+0x2c>                             # executed: 1
800008a0:	060a                	slli	a2,a2,0x2                                                        # executed: 1
800008a2:	00269813          	slli	a6,a3,0x2                                                          # executed: 1
800008a6:	00c588b3          	add	a7,a1,a2                                                            # executed: 1
800008aa:	01050633          	add	a2,a0,a6                                                            # executed: 1
800008ae:	ce99                	beqz	a3,800008cc <layernorm_sum_i32+0x2e>                             # executed: 1
800008b0:	410607b3          	sub	a5,a2,a6                                                            # executed: 32
800008b4:	4701                	li	a4,0                                                               # executed: 32
800008b6:	4394                	lw	a3,0(a5)                                                           # executed: 4,096
800008b8:	0791                	addi	a5,a5,4                                                          # executed: 4,096
800008ba:	9736                	add	a4,a4,a3                                                          # executed: 4,096
800008bc:	fec79de3          	bne	a5,a2,800008b6 <layernorm_sum_i32+0x18>                             # executed: 4,096
800008c0:	c198                	sw	a4,0(a1)                                                           # executed: 32
800008c2:	0591                	addi	a1,a1,4                                                          # executed: 32
800008c4:	9642                	add	a2,a2,a6                                                          # executed: 32
800008c6:	feb895e3          	bne	a7,a1,800008b0 <layernorm_sum_i32+0x12>                             # executed: 32
800008ca:	8082                	ret                                                                   # executed: 1
800008cc:	0005a023          	sw	zero,0(a1)                                                           # executed: 0
800008d0:	00458793          	addi	a5,a1,4                                                            # executed: 0
800008d4:	fef88be3          	beq	a7,a5,800008ca <layernorm_sum_i32+0x2c>                             # executed: 0
800008d8:	0005a223          	sw	zero,4(a1)                                                           # executed: 0
800008dc:	05a1                	addi	a1,a1,8                                                          # executed: 0
800008de:	feb897e3          	bne	a7,a1,800008cc <layernorm_sum_i32+0x2e>                             # executed: 0
800008e2:	8082                	ret                                                                   # executed: 0
```


### 3.2 `layernorm_squared_difference_i32()` — Aligned squared differences

Compute $d_{t,k}=128x_{t,k}-S_t$, then $d_{t,k}^2$ in INT64. The eight-instruction feature loop is `0x80000906–0x8000091c`. It uses `lw` for the input and `sd` for the result. The count is $8+T(8+8F)=33{,}032$, including two `mul` instructions per feature.

#### Corresponding C

```c
void layernorm_squared_difference_i32(
    const int32_t *input,
    const int32_t *mean_sum,
    int64_t *output,
    size_t token_count,
    size_t feature_count
)
{
    const int64_t alignment = (int64_t)feature_count;

    for (size_t t = 0; t < token_count; ++t) {
        const int32_t *input_row = input + t * feature_count;
        int64_t *output_row = output + t * feature_count;

        const int64_t mean = mean_sum[t];

        for (size_t k = 0; k < feature_count; ++k) {
            const int64_t aligned_input =
                (int64_t)input_row[k] * alignment;

            const int64_t difference = aligned_input - mean;

            output_row[k] = difference * difference;
        }
    }
}
```


#### Compiled assembly

```asm
00000000800008e4 <layernorm_squared_difference_i32>:
800008e4:	c2b9                	beqz	a3,8000092a <layernorm_squared_difference_i32+0x46>              # executed: 1
800008e6:	068a                	slli	a3,a3,0x2                                                        # executed: 1
800008e8:	00d58eb3          	add	t4,a1,a3                                                            # executed: 1
800008ec:	c321                	beqz	a4,8000092c <layernorm_squared_difference_i32+0x48>              # executed: 1
800008ee:	00271e13          	slli	t3,a4,0x2                                                          # executed: 1
800008f2:	01c50833          	add	a6,a0,t3                                                            # executed: 1
800008f6:	4301                	li	t1,0                                                               # executed: 1
800008f8:	0005a883          	lw	a7,0(a1)                                                             # executed: 32
800008fc:	00331513          	slli	a0,t1,0x3                                                          # executed: 32
80000900:	9532                	add	a0,a0,a2                                                          # executed: 32
80000902:	41c806b3          	sub	a3,a6,t3                                                            # executed: 32
80000906:	429c                	lw	a5,0(a3)                                                           # executed: 4,096
80000908:	0691                	addi	a3,a3,4                                                          # executed: 4,096
8000090a:	0521                	addi	a0,a0,8                                                          # executed: 4,096
8000090c:	02e787b3          	mul	a5,a5,a4                                                            # executed: 4,096
80000910:	411787b3          	sub	a5,a5,a7                                                            # executed: 4,096
80000914:	02f787b3          	mul	a5,a5,a5                                                            # executed: 4,096
80000918:	fef53c23          	sd	a5,-8(a0)                                                            # executed: 4,096
8000091c:	ff0695e3          	bne	a3,a6,80000906 <layernorm_squared_difference_i32+0x22>              # executed: 4,096
80000920:	0591                	addi	a1,a1,4                                                          # executed: 32
80000922:	933a                	add	t1,t1,a4                                                          # executed: 32
80000924:	9872                	add	a6,a6,t3                                                          # executed: 32
80000926:	fcbe99e3          	bne	t4,a1,800008f8 <layernorm_squared_difference_i32+0x14>              # executed: 32
8000092a:	8082                	ret                                                                   # executed: 1
8000092c:	00458793          	addi	a5,a1,4                                                            # executed: 0
80000930:	05a1                	addi	a1,a1,8                                                          # executed: 0
80000932:	fefe8ce3          	beq	t4,a5,8000092a <layernorm_squared_difference_i32+0x46>              # executed: 0
80000936:	ffd59be3          	bne	a1,t4,8000092c <layernorm_squared_difference_i32+0x48>              # executed: 0
8000093a:	8082                	ret                                                                   # executed: 0
```


### 3.3 `layernorm_variance_i32()` — Variance numerator

Sum the already converted squared differences over 128 features. The four-instruction feature loop uses `lw`, pointer `addi`, `add`, and `bne`; INT64 accumulation precedes the INT32 store. The count is $7+T(6+4F)=16{,}583$. Its scale is set by the surrounding conversion and LN parameters; it is not an extra floating-point variance calculation.

#### Corresponding C

```c
void layernorm_variance_i32(
    const int32_t *squared_difference,
    int32_t *output,
    size_t token_count,
    size_t feature_count
)
{
    for (size_t t = 0; t < token_count; ++t) {
        const int32_t *row =
            squared_difference + t * feature_count;

        int64_t sum = 0;

        for (size_t k = 0; k < feature_count; ++k) {
            sum += (int64_t)row[k];
        }

        output[t] = (int32_t)sum;
    }
}
```


#### Compiled assembly

```asm
0000000080000724 <layernorm_variance_i32>:
80000724:	c615                	beqz	a2,80000750 <layernorm_variance_i32+0x2c>                        # executed: 1
80000726:	060a                	slli	a2,a2,0x2                                                        # executed: 1
80000728:	00269813          	slli	a6,a3,0x2                                                          # executed: 1
8000072c:	00c588b3          	add	a7,a1,a2                                                            # executed: 1
80000730:	01050633          	add	a2,a0,a6                                                            # executed: 1
80000734:	ce99                	beqz	a3,80000752 <layernorm_variance_i32+0x2e>                        # executed: 1
80000736:	410607b3          	sub	a5,a2,a6                                                            # executed: 32
8000073a:	4701                	li	a4,0                                                               # executed: 32
8000073c:	4394                	lw	a3,0(a5)                                                           # executed: 4,096
8000073e:	0791                	addi	a5,a5,4                                                          # executed: 4,096
80000740:	9736                	add	a4,a4,a3                                                          # executed: 4,096
80000742:	fec79de3          	bne	a5,a2,8000073c <layernorm_variance_i32+0x18>                        # executed: 4,096
80000746:	c198                	sw	a4,0(a1)                                                           # executed: 32
80000748:	0591                	addi	a1,a1,4                                                          # executed: 32
8000074a:	9642                	add	a2,a2,a6                                                          # executed: 32
8000074c:	feb895e3          	bne	a7,a1,80000736 <layernorm_variance_i32+0x12>                        # executed: 32
80000750:	8082                	ret                                                                   # executed: 1
80000752:	0005a023          	sw	zero,0(a1)                                                           # executed: 0
80000756:	00458793          	addi	a5,a1,4                                                            # executed: 0
8000075a:	fef88be3          	beq	a7,a5,80000750 <layernorm_variance_i32+0x2c>                        # executed: 0
8000075e:	0005a223          	sw	zero,4(a1)                                                           # executed: 0
80000762:	05a1                	addi	a1,a1,8                                                          # executed: 0
80000764:	feb897e3          	bne	a7,a1,80000752 <layernorm_variance_i32+0x2e>                        # executed: 0
80000768:	8082                	ret                                                                   # executed: 0
```


### 3.4 `layernorm_add_epsilon_i32()` — Epsilon addition

Add the block-specific integer epsilon to each of 32 token variances. The six-instruction element loop has one `lw`, two pointer updates, `addw`, `sw`, and `bne`; $6T+4=196$. The helper supports in-place output.

#### Corresponding C

```c
void layernorm_add_epsilon_i32(
    const int32_t *variance,
    int32_t *output,
    size_t token_count,
    int32_t epsilon_q
)
{
    for (size_t t = 0; t < token_count; ++t) {
        const int64_t sum =
            (int64_t)variance[t] + (int64_t)epsilon_q;

        output[t] = (int32_t)sum;
    }
}
```


#### Compiled assembly

```asm
000000008000076a <layernorm_add_epsilon_i32>:
8000076a:	ca19                	beqz	a2,80000780 <layernorm_add_epsilon_i32+0x16>                     # executed: 1
8000076c:	060a                	slli	a2,a2,0x2                                                        # executed: 1
8000076e:	962a                	add	a2,a2,a0                                                          # executed: 1
80000770:	411c                	lw	a5,0(a0)                                                           # executed: 32
80000772:	0511                	addi	a0,a0,4                                                          # executed: 32
80000774:	0591                	addi	a1,a1,4                                                          # executed: 32
80000776:	9fb5                	addw	a5,a5,a3                                                         # executed: 32
80000778:	fef5ae23          	sw	a5,-4(a1)                                                            # executed: 32
8000077c:	fec51ae3          	bne	a0,a2,80000770 <layernorm_add_epsilon_i32+0x6>                      # executed: 32
80000780:	8082                	ret                                                                   # executed: 1
```


### 3.5 `layernorm_rsqrt_i32()` — Exact reciprocal-square-root candidate

For each token, calculate $q=\lfloor N_{\rm rsqrt}/V\rfloor$, then the exact floor integer square root $r=\lfloor\sqrt q\rfloor$. The `divu` at `0x80000798` executes once per token. `isqrt_u64()` is inlined here: there is no extra function call.

The initial bit search at `0x800007a0–0x800007a2` shifts a power of four down toward the quotient. The root loop at `0x800007aa–0x800007bc` uses shifts, comparisons, and conditional subtraction/addition. Thus different input values can change its executed instruction count. For these fixtures, let P be total initial bit-search iterations, R total root-loop iterations, and S successful conditional subtracts. The executed path has $7+11T+2P+5R+2S$ instructions. This formula applies to the observed positive quotients and their paths; it is not a formula for every possible argument.

Case 0: P = 388, R = 636, S = 311, giving **4,937 instructions**.

```c
static uint64_t isqrt_u64(uint64_t value)
{
    uint64_t root = 0;
    uint64_t bit = UINT64_C(1) << 62;

    /* Start at the largest power of four not exceeding value. */
    while (bit > value) {
        bit >>= 2;
    }

    while (bit != 0) {
        if (value >= root + bit) {
            value -= root + bit;
            root = (root >> 1) + bit;
        } else {
            root >>= 1;
        }

        bit >>= 2;
    }

    return root;
}
```


#### Corresponding C

```c
void layernorm_rsqrt_i32(
    const int32_t *input,
    int32_t *output,
    size_t token_count,
    uint64_t numerator
)
{
    for (size_t t = 0; t < token_count; ++t) {
        const uint64_t quotient =
            numerator / (uint64_t)input[t];

        output[t] = (int32_t)isqrt_u64(quotient);
    }
}
```


#### Compiled assembly

```asm
0000000080000782 <layernorm_rsqrt_i32>:
80000782:	c621                	beqz	a2,800007ca <layernorm_rsqrt_i32+0x48>                           # executed: 1
80000784:	060a                	slli	a2,a2,0x2                                                        # executed: 1
80000786:	537d                	li	t1,-1                                                              # executed: 1
80000788:	00c508b3          	add	a7,a0,a2                                                            # executed: 1
8000078c:	00235313          	srli	t1,t1,0x2                                                          # executed: 1
80000790:	4e05                	li	t3,1                                                               # executed: 1
80000792:	4110                	lw	a2,0(a0)                                                           # executed: 32
80000794:	03ee1793          	slli	a5,t3,0x3e                                                         # executed: 32
80000798:	02c6d633          	divu	a2,a3,a2                                                           # executed: 32
8000079c:	00c36663          	bltu	t1,a2,800007a8 <layernorm_rsqrt_i32+0x26>                          # executed: 32
800007a0:	8389                	srli	a5,a5,0x2                                                        # executed: 388
800007a2:	fef66fe3          	bltu	a2,a5,800007a0 <layernorm_rsqrt_i32+0x1e>                          # executed: 388
800007a6:	c39d                	beqz	a5,800007cc <layernorm_rsqrt_i32+0x4a>                           # executed: 32
800007a8:	4701                	li	a4,0                                                               # executed: 32
800007aa:	00e78833          	add	a6,a5,a4                                                            # executed: 636
800007ae:	8305                	srli	a4,a4,0x1                                                        # executed: 636
800007b0:	01066563          	bltu	a2,a6,800007ba <layernorm_rsqrt_i32+0x38>                          # executed: 636
800007b4:	41060633          	sub	a2,a2,a6                                                            # executed: 311
800007b8:	973e                	add	a4,a4,a5                                                          # executed: 311
800007ba:	8389                	srli	a5,a5,0x2                                                        # executed: 636
800007bc:	f7fd                	bnez	a5,800007aa <layernorm_rsqrt_i32+0x28>                           # executed: 636
800007be:	2701                	sext.w	a4,a4                                                          # executed: 32
800007c0:	c198                	sw	a4,0(a1)                                                           # executed: 32
800007c2:	0511                	addi	a0,a0,4                                                          # executed: 32
800007c4:	0591                	addi	a1,a1,4                                                          # executed: 32
800007c6:	fca896e3          	bne	a7,a0,80000792 <layernorm_rsqrt_i32+0x10>                           # executed: 32
800007ca:	8082                	ret                                                                   # executed: 1
800007cc:	4701                	li	a4,0                                                               # executed: 0
800007ce:	bfcd                	j	800007c0 <layernorm_rsqrt_i32+0x3e>                                 # executed: 0
```


### 3.6 `layernorm_normalize_i32()` — Fine normalized features

Calculate $(128x_{t,k}-S_t)r_t$ from INT32 input into INT64 scratch. The eight-instruction loop is `0x80000962–0x80000978`; it has two `mul` instructions per feature. The count is $8+T(10+8F)=33{,}096$. The mean and inverse standard deviation are loaded once per token.

#### Corresponding C

```c
void layernorm_normalize_i32(
    const int32_t *input,
    const int32_t *mean_sum,
    const int32_t *inv_std,
    int64_t *output,
    size_t token_count,
    size_t feature_count
)
{
    const int64_t alignment = (int64_t)feature_count;

    for (size_t t = 0; t < token_count; ++t) {
        const int32_t *input_row = input + t * feature_count;
        int64_t *output_row = output + t * feature_count;

        const int64_t mean = mean_sum[t];
        const int64_t inverse_std = inv_std[t];

        for (size_t k = 0; k < feature_count; ++k) {
            const int64_t aligned_input =
                (int64_t)input_row[k] * alignment;

            const int64_t difference = aligned_input - mean;

            output_row[k] = difference * inverse_std;
        }
    }
}
```


#### Compiled assembly

```asm
000000008000093c <layernorm_normalize_i32>:
8000093c:	c731                	beqz	a4,80000988 <layernorm_normalize_i32+0x4c>                       # executed: 1
8000093e:	070a                	slli	a4,a4,0x2                                                        # executed: 1
80000940:	00e58fb3          	add	t6,a1,a4                                                            # executed: 1
80000944:	c3b9                	beqz	a5,8000098a <layernorm_normalize_i32+0x4e>                       # executed: 1
80000946:	00279f13          	slli	t5,a5,0x2                                                          # executed: 1
8000094a:	01e508b3          	add	a7,a0,t5                                                            # executed: 1
8000094e:	4e81                	li	t4,0                                                               # executed: 1
80000950:	0005ae03          	lw	t3,0(a1)                                                             # executed: 32
80000954:	00062303          	lw	t1,0(a2)                                                             # executed: 32
80000958:	003e9813          	slli	a6,t4,0x3                                                          # executed: 32
8000095c:	9836                	add	a6,a6,a3                                                          # executed: 32
8000095e:	41e88533          	sub	a0,a7,t5                                                            # executed: 32
80000962:	4118                	lw	a4,0(a0)                                                           # executed: 4,096
80000964:	0511                	addi	a0,a0,4                                                          # executed: 4,096
80000966:	0821                	addi	a6,a6,8                                                          # executed: 4,096
80000968:	02f70733          	mul	a4,a4,a5                                                            # executed: 4,096
8000096c:	41c70733          	sub	a4,a4,t3                                                            # executed: 4,096
80000970:	02670733          	mul	a4,a4,t1                                                            # executed: 4,096
80000974:	fee83c23          	sd	a4,-8(a6)                                                            # executed: 4,096
80000978:	ff1515e3          	bne	a0,a7,80000962 <layernorm_normalize_i32+0x26>                       # executed: 4,096
8000097c:	0591                	addi	a1,a1,4                                                          # executed: 32
8000097e:	9ebe                	add	t4,t4,a5                                                          # executed: 32
80000980:	98fa                	add	a7,a7,t5                                                          # executed: 32
80000982:	0611                	addi	a2,a2,4                                                          # executed: 32
80000984:	fcbf96e3          	bne	t6,a1,80000950 <layernorm_normalize_i32+0x14>                       # executed: 32
80000988:	8082                	ret                                                                   # executed: 1
8000098a:	00458793          	addi	a5,a1,4                                                            # executed: 0
8000098e:	05a1                	addi	a1,a1,8                                                          # executed: 0
80000990:	feff8ce3          	beq	t6,a5,80000988 <layernorm_normalize_i32+0x4c>                       # executed: 0
80000994:	fff59be3          	bne	a1,t6,8000098a <layernorm_normalize_i32+0x4e>                       # executed: 0
80000998:	8082                	ret                                                                   # executed: 0
```


### 3.7 `layernorm_affine_i32()` — Gamma and beta

Compute $u_{t,k}\gamma_k+\beta_k A$, where $A=65{,}536$. The compiled generic helper still uses a `mul` for beta alignment even though this invocation passes a power of two. The twelve-instruction inner loop at `0x80000856–0x80000878` has three `lw`, four pointer updates, two `mul`, one numerical `add`, one `sd`, and one `bne`. Including setup, token control, and exit gives $15+T(9+12F)=49{,}455$.

#### Corresponding C

```c
void layernorm_affine_i32(
    const int32_t *normalized,
    const int32_t *gamma,
    const int32_t *beta,
    int64_t *output,
    size_t token_count,
    size_t feature_count,
    int64_t beta_alignment
)
{
    for (size_t t = 0; t < token_count; ++t) {
        const size_t offset = t * feature_count;

        for (size_t k = 0; k < feature_count; ++k) {
            const int64_t product =
                (int64_t)normalized[offset + k]
                * (int64_t)gamma[k];

            const int64_t aligned_beta =
                (int64_t)beta[k] * beta_alignment;

            output[offset + k] = product + aligned_beta;
        }
    }
}
```


#### Compiled assembly

```asm
000000008000082e <layernorm_affine_i32>:
8000082e:	c73d                	beqz	a4,8000089c <layernorm_affine_i32+0x6e>                          # executed: 1
80000830:	1141                	addi	sp,sp,-16                                                        # executed: 1
80000832:	e422                	sd	s0,8(sp)                                                           # executed: 1
80000834:	e026                	sd	s1,0(sp)                                                           # executed: 1
80000836:	843e                	mv	s0,a5                                                              # executed: 1
80000838:	84ba                	mv	s1,a4                                                              # executed: 1
8000083a:	cba9                	beqz	a5,8000088c <layernorm_affine_i32+0x5e>                          # executed: 1
8000083c:	00279393          	slli	t2,a5,0x2                                                          # executed: 1
80000840:	93ae                	add	t2,t2,a1                                                          # executed: 1
80000842:	4701                	li	a4,0                                                               # executed: 1
80000844:	4781                	li	a5,0                                                               # executed: 1
80000846:	00279f13          	slli	t5,a5,0x2                                                          # executed: 32
8000084a:	00379e93          	slli	t4,a5,0x3                                                          # executed: 32
8000084e:	9f2a                	add	t5,t5,a0                                                          # executed: 32
80000850:	9eb6                	add	t4,t4,a3                                                          # executed: 32
80000852:	8fb2                	mv	t6,a2                                                              # executed: 32
80000854:	832e                	mv	t1,a1                                                              # executed: 32
80000856:	000f2883          	lw	a7,0(t5)                                                             # executed: 4,096
8000085a:	00032283          	lw	t0,0(t1)                                                             # executed: 4,096
8000085e:	000fae03          	lw	t3,0(t6)                                                             # executed: 4,096
80000862:	0311                	addi	t1,t1,4                                                          # executed: 4,096
80000864:	025888b3          	mul	a7,a7,t0                                                            # executed: 4,096
80000868:	0f11                	addi	t5,t5,4                                                          # executed: 4,096
8000086a:	0f91                	addi	t6,t6,4                                                          # executed: 4,096
8000086c:	0ea1                	addi	t4,t4,8                                                          # executed: 4,096
8000086e:	030e0e33          	mul	t3,t3,a6                                                            # executed: 4,096
80000872:	98f2                	add	a7,a7,t3                                                          # executed: 4,096
80000874:	ff1ebc23          	sd	a7,-8(t4)                                                            # executed: 4,096
80000878:	fc639fe3          	bne	t2,t1,80000856 <layernorm_affine_i32+0x28>                          # executed: 4,096
8000087c:	0705                	addi	a4,a4,1                                                          # executed: 32
8000087e:	97a2                	add	a5,a5,s0                                                          # executed: 32
80000880:	fce493e3          	bne	s1,a4,80000846 <layernorm_affine_i32+0x18>                          # executed: 32
80000884:	6422                	ld	s0,8(sp)                                                           # executed: 1
80000886:	6482                	ld	s1,0(sp)                                                           # executed: 1
80000888:	0141                	addi	sp,sp,16                                                         # executed: 1
8000088a:	8082                	ret                                                                   # executed: 1
8000088c:	00140793          	addi	a5,s0,1                                                            # executed: 0
80000890:	0409                	addi	s0,s0,2                                                          # executed: 0
80000892:	fef489e3          	beq	s1,a5,80000884 <layernorm_affine_i32+0x56>                          # executed: 0
80000896:	fe941be3          	bne	s0,s1,8000088c <layernorm_affine_i32+0x5e>                          # executed: 0
8000089a:	b7ed                	j	80000884 <layernorm_affine_i32+0x56>                                # executed: 0
8000089c:	8082                	ret                                                                   # executed: 0
```


## 4. Conversion Kernels

### 4.1 `requantize_shift_i64_to_i32()` — Rounded shift without multiplication

Compute $(v+2^{s-1})\mathbin{\gg}s$ from INT64 scratch into INT32. The seven-instruction loop `0x800009aa–0x800009ba` has an INT64 load, two pointer updates, a rounding add, arithmetic shift, INT32 store, and branch. Six setup instructions plus one return give $7N+7=28{,}679$ for 4,096 elements. This helper runs twice per LN. There is no clamp and no multiply.

#### Corresponding C

```c
void requantize_shift_i64_to_i32(
    const int64_t *input,
    int32_t *output,
    size_t element_count,
    unsigned int shift
)
{
    const int64_t rounding_offset =
        INT64_C(1) << (shift - 1);

    for (size_t i = 0; i < element_count; ++i) {
        const int64_t rounded =
            (input[i] + rounding_offset) >> shift;

        output[i] = (int32_t)rounded;
    }
}
```


#### Compiled assembly

```asm
000000008000099a <requantize_shift_i64_to_i32>:
8000099a:	c215                	beqz	a2,800009be <requantize_shift_i64_to_i32+0x24>                   # executed: 2
8000099c:	060e                	slli	a2,a2,0x3                                                        # executed: 2
8000099e:	fff6879b          	addiw	a5,a3,-1                                                          # executed: 2
800009a2:	4705                	li	a4,1                                                               # executed: 2
800009a4:	962a                	add	a2,a2,a0                                                          # executed: 2
800009a6:	00f71733          	sll	a4,a4,a5                                                            # executed: 2
800009aa:	611c                	ld	a5,0(a0)                                                           # executed: 8,192
800009ac:	0521                	addi	a0,a0,8                                                          # executed: 8,192
800009ae:	0591                	addi	a1,a1,4                                                          # executed: 8,192
800009b0:	97ba                	add	a5,a5,a4                                                          # executed: 8,192
800009b2:	40d7d7b3          	sra	a5,a5,a3                                                            # executed: 8,192
800009b6:	fef5ae23          	sw	a5,-4(a1)                                                            # executed: 8,192
800009ba:	fec518e3          	bne	a0,a2,800009aa <requantize_shift_i64_to_i32+0x10>                   # executed: 8,192
800009be:	8082                	ret                                                                   # executed: 2
```


### 4.2 `requantize_i64_to_i8()` — INT64 to INT8 fixed-point conversion

Compute $\operatorname{clamp}_{[-128,127]}(((vM+2^{s-1})\gg s)+z)$. The ordinary element path has eleven instructions: one `ld`, two pointer updates, one `mul`, two numerical `add`, one `sra`, two `bge`, one `sb`, and one `bne`. Two optional `li` saturation assignments are mutually exclusive for each element. With C actual clipped elements, $I=11N+9+C$.

The clipping assignments are `0x800009ee` (upper) and `0x800009f6` (lower). A value exactly at an endpoint is not a clipped element. Passing M = 1 for GELU's branch-combination conversion still executes the generic `mul`; the compiler did not specialize the helper call.

#### Corresponding C

```c
void requantize_i64_to_i8(
    const int64_t *input,
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
        const int64_t product = input[i] * multiplier;

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
00000000800009c0 <requantize_i64_to_i8>:
800009c0:	c231                	beqz	a2,80000a04 <requantize_i64_to_i8+0x44>                          # executed: 1
800009c2:	060e                	slli	a2,a2,0x3                                                        # executed: 1
800009c4:	fff7081b          	addiw	a6,a4,-1                                                          # executed: 1
800009c8:	4885                	li	a7,1                                                               # executed: 1
800009ca:	010898b3          	sll	a7,a7,a6                                                            # executed: 1
800009ce:	962a                	add	a2,a2,a0                                                          # executed: 1
800009d0:	07f00e13          	li	t3,127                                                               # executed: 1
800009d4:	f8000313          	li	t1,-128                                                              # executed: 1
800009d8:	00053803          	ld	a6,0(a0)                                                             # executed: 4,096
800009dc:	0521                	addi	a0,a0,8                                                          # executed: 4,096
800009de:	03068833          	mul	a6,a3,a6                                                            # executed: 4,096
800009e2:	9846                	add	a6,a6,a7                                                          # executed: 4,096
800009e4:	40e85833          	sra	a6,a6,a4                                                            # executed: 4,096
800009e8:	983e                	add	a6,a6,a5                                                          # executed: 4,096
800009ea:	010e5463          	bge	t3,a6,800009f2 <requantize_i64_to_i8+0x32>                          # executed: 4,096
800009ee:	07f00813          	li	a6,127                                                               # executed: 0
800009f2:	00685463          	bge	a6,t1,800009fa <requantize_i64_to_i8+0x3a>                          # executed: 4,096
800009f6:	f8000813          	li	a6,-128                                                              # executed: 0
800009fa:	01058023          	sb	a6,0(a1)                                                             # executed: 4,096
800009fe:	0585                	addi	a1,a1,1                                                          # executed: 4,096
80000a00:	fca61ce3          	bne	a2,a0,800009d8 <requantize_i64_to_i8+0x18>                          # executed: 4,096
80000a04:	8082                	ret                                                                   # executed: 1
```


## 5. Complete Instruction Accounting

### 5.1 Loop and computation counts

Let T = 32 tokens, F = 128 features, and N = TF = 4,096. The normal paths in this binary have:

| Helper | Instruction formula | Per invocation |
|---|---|---|
| mean sum | 7 + T(7 + 4F) for INT8; 7 + T(6 + 4F) for INT32 | 16,583 |
| squared difference | 8 + T(8 + 8F) | 33,032 |
| variance sum | 7 + T(6 + 4F) | 16,583 |
| epsilon | 6T + 4 | 196 |
| normalize | 8 + T(10 + 8F) | 33,096 |
| affine | 15 + T(9 + 12F) | 49,455 |

The square and normalize loops each execute 8,192 multiplications (alignment plus square/product). Affine executes another 8,192 (gamma product plus beta alignment). Final conversion executes 4,096. The stage therefore has **28,672 multiplication instructions**, all `mul`, and **32 `divu`**. The two rounded-shift conversions execute no multiplication. The exact integer-square-root loop uses shifts, comparisons, additions, and subtractions.

Layer normalization is not a dense matrix-MAC workload. Its sum reductions, integer square root, per-feature products, and conversions have different semantics.

### 5.2 Requantization counts

| Conversion | Elements | Formula | C, case 0 | Executed instructions |
|---|---|---|---|---|
| square conversion | 4,096 | 7N + 7 | 0 | 28,679 |
| normalize conversion | 4,096 | 7N + 7 | 0 | 28,679 |
| T88 conversion | 4,096 | 11N + 9 + C | 0 | 45,065 |
| **Total** |  |  |  | **102,423** |

Conversion kernels account for **39.95%** of this stage body in case 0. Computation kernels, conversion kernels, and wrapper instructions are counted separately.

### 5.3 Function totals and instruction breakdown

| Order | Work | Function | Case-0 instructions | Share of stage body |
|---|---|---|---|---|
| 0 | wrapper | `block0_ln2_forward` | 74 | 0.0289% |
| 1 | mean sums | `layernorm_sum_i32` | 16,583 | 6.4682% |
| 2 | squared differences | `layernorm_squared_difference_i32` | 33,032 | 12.8841% |
| 3 | square conversion | `requantize_shift_i64_to_i32` | 28,679 | 11.1862% |
| 4 | variance sums | `layernorm_variance_i32` | 16,583 | 6.4682% |
| 5 | epsilon | `layernorm_add_epsilon_i32` | 196 | 0.0764% |
| 6 | inverse standard deviation | `layernorm_rsqrt_i32` | 4,937 | 1.9257% |
| 7 | normalize | `layernorm_normalize_i32` | 33,096 | 12.9090% |
| 8 | normalize conversion | `requantize_shift_i64_to_i32` | 28,679 | 11.1862% |
| 9 | affine transform | `layernorm_affine_i32` | 49,455 | 19.2898% |
| 10 | T88 conversion | `requantize_i64_to_i8` | 45,065 | 17.5775% |
|  | **Total, including callees** |  | **256,379** | **100%** |

Across this stage, multiplication instructions (`mul` + `mulw`) total **28,672**. Register additions (`add` + `addw`) total **30,017**. These opcode totals include address/setup arithmetic where the compiler uses those mnemonics; they are not counts of mathematical MACs. Immediate additions (`addi` + `addiw`) are listed separately below.

| Disassembly mnemonic | Executed, case 0 | Function in this stage |
|---|---|---|
| `beqz` | 47 | Zero-size checks or zero-bit tests. |
| `addi` | 65,861 | Add an immediate: pointer/loop/stack updates, plus immediate numerical constants. |
| `sd` | 12,295 | Store INT64 intermediates or save registers. |
| `mv` | 73 | Copy a pointer/value or prepare call arguments; also select attention maxima. |
| `mul` | 28,672 | 64-bit low-product arithmetic: requantization, scaling, normalization, and occasional dimension setup. |
| `add` | 29,985 | 64-bit addition: numerical arithmetic and address/index calculations. |
| `slli` | 174 | Immediate left shift: byte offsets, alignment, or integer constants. |
| `li` | 119 | Load a constant, initialize a loop/accumulator, or assign a saturation endpoint. |
| `addw` | 32 | 32-bit addition: accumulation, residual/elementwise arithmetic, or rounding-independent sums. |
| `bne` | 32,992 | Continue a counted pointer/index loop. |
| `ld` | 12,318 | Load INT64 scratch data, pointers, or saved registers. |
| `sw` | 8,320 | Store an INT32 result, intermediate, or probability. |
| `ret` | 10 | Return to the caller; charged to the returning helper. |
| `j` | 1 | Unconditional jump, including a tail call or loop transfer. |
| `lw` | 28,837 | Load a signed INT32 accumulator, probability, parameter, or LUT value. |
| `bge` | 8,192 | Signed comparisons for clipping, maximum selection, or the exponential cutoff. |
| `sext.w` | 32 | Sign-extend a 32-bit value to RV64. |
| `addiw` | 3 | 32-bit immediate addition: rounding-shift setup or LUT index calculation. |
| `sll` | 3 | Register-controlled left shift: rounding constants or probability scaling. |
| `lui` | 3 | Construct a constant or high part of an address. |
| `sub` | 8,631 | 64-bit subtraction: centering, normalization, interpolation, square-root work, or row addressing. |
| `sra` | 12,288 | Arithmetic right shift: rounded fixed-point conversion. |
| `srli` | 1,661 | Logical right shift: integer square-root iteration, sign extraction, or division setup. |
| `divu` | 32 | Unsigned integer division before the integer square root. |
| `bltu` | 1,056 | Unsigned tests in the exact integer square root. |
| `bnez` | 636 | Continue an integer square-root loop. |
| `sb` | 4,096 | Store an INT8 activation. |
| `jal` | 9 | Call a helper; its instruction is charged to the caller. |
| `auipc` | 1 | Construct a PC-relative parameter/table address. |
| **Total** | **256,379** |  |

### 5.4 Reconciliation with `minstret`

The supplied Spike counter interval contains **5 instructions outside the stage body**: the first counter read, argument setup, and the call. The ending read has not yet retired when its value is sampled. Instructions that subtract, save, or print the delta are outside this interval. This is specific to the supplied compiled call site.

```asm
80001aaa:	b0202af3          	csrr	s5,minstret
80001aae:	65ca                	ld	a1,144(sp)
80001ab0:	652e                	ld	a0,200(sp)
80001ab2:	0090                	addi	a2,sp,64
80001ab4:	bbeff0ef          	jal	80000e72 <block0_ln2_forward>
80001ab8:	b02027f3          	csrr	a5,minstret
```


| Case | Supplied raw minstret | Caller/measurement overhead | Counted stage body + callees | Unexplained difference |
|---|---|---|---|---|
| 0 | 256,384 | 5 | 256,379 | 0 |
| 1 | 256,406 | 5 | 256,401 | 0 |
| 2 | 256,371 | 5 | 256,366 | 0 |
| 3 | 256,385 | 5 | 256,380 | 0 |
| 4 | 256,393 | 5 | 256,388 | 0 |
| 5 | 256,364 | 5 | 256,359 | 0 |
| 6 | 256,380 | 5 | 256,375 | 0 |
| 7 | 256,375 | 5 | 256,370 | 0 |
| **Mean** | 256,382.25 | 5 | 256,377.25 | **0** |


Every case reconciles exactly. The body count includes all helper calls and their returns, and each wrapper instruction once per actual execution. The separately run staged and full paths both passed the supplied T107 comparisons.

| Case | rsqrt instructions | Final INT8 clipped elements |
|---|---|---|
| 0 | 4,937 | 0 |
| 1 | 4,959 | 0 |
| 2 | 4,924 | 0 |
| 3 | 4,938 | 0 |
| 4 | 4,946 | 0 |
| 5 | 4,917 | 0 |
| 6 | 4,931 | 2 |
| 7 | 4,928 | 0 |

Variation is localized to the exact integer square root and occasional final clipping; all other helper counts are fixed for these shapes.

## 6. Findings

| Region | Case-0 instructions | Share of stage body |
|---|---|---|
| Computation | 153,882 | 60.02% |
| Requantization | 102,423 | 39.95% |
| Wrapper | 74 | 0.03% |

The largest single helper is affine (49,455 instructions), followed by final INT8 conversion (45,065 in case 0). The two shift conversions together use 57,358 instructions. Each LN stage is about 0.45% of full-block raw instructions, so it is a lower-priority target for reducing the full-block instruction count. Input-dependent integer-square-root branches explain most small count differences.

Instruction shares identify where instructions are executed. They do not establish load latency, multiplier latency, cache behavior, or CV-Wally cycle speedup. No accelerator or custom-instruction speedup is measured here. The eight fixture passes validate these supplied integer cases; they do not establish equality with the original mixed TFLite graph or unseen-data quality.

[Back to results](Result.md) · [Counting method and validation](analysis_method.md)
