# Part 1: First Layer Normalization — Disassembly Analysis

## 1. Function Overview

This note analyzes `block0_ln1_forward()` and every helper it actually executes in the **supplied compiled ELF**. Case 0 provides the detailed instruction mix; Section 5.4 reconciles all eight cases with the supplied Spike `minstret` results. Counts describe dynamic executions, not the number of lines in the disassembly. See [counting method](analysis_method.md).

| Parameter | Current block-0 value |
|---|---|
| tokens / features / elements | 32 / 128 / 4,096 |
| input / output | INT8 T0 → INT8 T45 |
| square / normalize shift | 15 / 15 |
| epsilon / rsqrt numerator | 42 / 2^54 |
| beta alignment | 65,536 |
| final multiplier / shift / output zero point | 974,600 / 47 / -17 |


## 2. Block Wrapper: `block0_ln1_forward()`

### 2.1 Source and operation order

```c
void block0_ln1_forward(
    const int8_t *input,
    int8_t *output,
    LayerNormI8Workspace *workspace
)
{
    const size_t token_count = BLOCK0_TOKEN_COUNT;
    const size_t feature_count = BLOCK0_HIDDEN_SIZE;
    const size_t element_count = token_count * feature_count;

    const LayerNormI8Params *params = &block0_ln1_params;

    /* Op 0–1: mean sums; broadcast through indexing. */
    layernorm_sum_i8(
        input, workspace->mean_sum,
        token_count, feature_count
    );

    /* Op 2: fine squared differences, scale 2^-30. */
    layernorm_squared_difference_i8(
        input, workspace->mean_sum, workspace->fine_i64,
        token_count, feature_count
    );

    /* T33: INT64 at 2^-30 -> INT32 at 2^-15. */
    requantize_shift_i64_to_i32(
        workspace->fine_i64, workspace->feature_i32,
        element_count, params->square_shift
    );

    /* Op 3: variance, scale 2^-22. */
    layernorm_variance_i32(
        workspace->feature_i32, workspace->variance,
        token_count, feature_count
    );

    /* Op 4: variance + epsilon; reuse variance buffer. */
    layernorm_add_epsilon_i32(
        workspace->variance, workspace->variance,
        token_count, params->epsilon_q
    );

    /* Op 5: inverse standard deviation, scale 2^-16. */
    layernorm_rsqrt_i32(
        workspace->variance, workspace->inv_std,
        token_count, params->rsqrt_numerator
    );

    /* Op 6–11: fine normalization, scale 2^-31. */
    layernorm_normalize_i8(
        input, workspace->mean_sum, workspace->inv_std,
        workspace->fine_i64, token_count, feature_count
    );

    /* T42: INT64 at 2^-31 -> INT32 at 2^-16. */
    requantize_shift_i64_to_i32(
        workspace->fine_i64, workspace->feature_i32,
        element_count, params->normalize_shift
    );

    /* Op 12–13: affine transform, scale 2^-32. */
    layernorm_affine_i32(
        workspace->feature_i32, params->gamma, params->beta,
        workspace->fine_i64, token_count, feature_count,
        params->beta_alignment
    );

    /* Op 14: T44 -> INT8 T45. */
    requantize_i64_to_i8(
        workspace->fine_i64, output, element_count,
        params->output_multiplier, params->output_shift,
        params->output_zero_point
    );
}
```


### 2.2 Compiled wrapper

The assembly below includes the wrapper only. Each `jal` or tail `j` transfers execution to a helper whose work is counted separately. `executed` is the case-0 execution frequency of that instruction. A helper reused several times has aggregated frequencies in its own listing. Unexecuted alternate paths are shown with zero.

```asm
0000000080000bc2 <block0_ln1_forward>:
80000bc2:	7179                	addi	sp,sp,-48                                                        # executed: 1
80000bc4:	e44e                	sd	s3,8(sp)                                                           # executed: 1
80000bc6:	89ae                	mv	s3,a1                                                              # executed: 1
80000bc8:	620c                	ld	a1,0(a2)                                                           # executed: 1
80000bca:	f022                	sd	s0,32(sp)                                                          # executed: 1
80000bcc:	08000693          	li	a3,128                                                               # executed: 1
80000bd0:	8432                	mv	s0,a2                                                              # executed: 1
80000bd2:	02000613          	li	a2,32                                                                # executed: 1
80000bd6:	f406                	sd	ra,40(sp)                                                          # executed: 1
80000bd8:	ec26                	sd	s1,24(sp)                                                          # executed: 1
80000bda:	e84a                	sd	s2,16(sp)                                                          # executed: 1
80000bdc:	892a                	mv	s2,a0                                                              # executed: 1
80000bde:	aa7ff0ef          	jal	80000684 <layernorm_sum_i8>                                         # executed: 1
80000be2:	7010                	ld	a2,32(s0)                                                          # executed: 1
80000be4:	600c                	ld	a1,0(s0)                                                           # executed: 1
80000be6:	08000713          	li	a4,128                                                               # executed: 1
80000bea:	854a                	mv	a0,s2                                                              # executed: 1
80000bec:	02000693          	li	a3,32                                                                # executed: 1
80000bf0:	addff0ef          	jal	800006cc <layernorm_squared_difference_i8>                          # executed: 1
80000bf4:	00036497          	auipc	s1,0x36                                                           # executed: 1
80000bf8:	7a448493          	addi	s1,s1,1956 # 80037398 <block0_ln1_params>                          # executed: 1
80000bfc:	6c0c                	ld	a1,24(s0)                                                          # executed: 1
80000bfe:	7008                	ld	a0,32(s0)                                                          # executed: 1
80000c00:	4894                	lw	a3,16(s1)                                                          # executed: 1
80000c02:	6605                	lui	a2,0x1                                                            # executed: 1
80000c04:	d97ff0ef          	jal	8000099a <requantize_shift_i64_to_i32>                              # executed: 1
80000c08:	640c                	ld	a1,8(s0)                                                           # executed: 1
80000c0a:	6c08                	ld	a0,24(s0)                                                          # executed: 1
80000c0c:	08000693          	li	a3,128                                                               # executed: 1
80000c10:	02000613          	li	a2,32                                                                # executed: 1
80000c14:	b11ff0ef          	jal	80000724 <layernorm_variance_i32>                                   # executed: 1
80000c18:	640c                	ld	a1,8(s0)                                                           # executed: 1
80000c1a:	48d4                	lw	a3,20(s1)                                                          # executed: 1
80000c1c:	02000613          	li	a2,32                                                                # executed: 1
80000c20:	852e                	mv	a0,a1                                                              # executed: 1
80000c22:	b49ff0ef          	jal	8000076a <layernorm_add_epsilon_i32>                                # executed: 1
80000c26:	6c94                	ld	a3,24(s1)                                                          # executed: 1
80000c28:	680c                	ld	a1,16(s0)                                                          # executed: 1
80000c2a:	6408                	ld	a0,8(s0)                                                           # executed: 1
80000c2c:	02000613          	li	a2,32                                                                # executed: 1
80000c30:	b53ff0ef          	jal	80000782 <layernorm_rsqrt_i32>                                      # executed: 1
80000c34:	7014                	ld	a3,32(s0)                                                          # executed: 1
80000c36:	6810                	ld	a2,16(s0)                                                          # executed: 1
80000c38:	600c                	ld	a1,0(s0)                                                           # executed: 1
80000c3a:	854a                	mv	a0,s2                                                              # executed: 1
80000c3c:	08000793          	li	a5,128                                                               # executed: 1
80000c40:	02000713          	li	a4,32                                                                # executed: 1
80000c44:	b8dff0ef          	jal	800007d0 <layernorm_normalize_i8>                                   # executed: 1
80000c48:	5094                	lw	a3,32(s1)                                                          # executed: 1
80000c4a:	6c0c                	ld	a1,24(s0)                                                          # executed: 1
80000c4c:	7008                	ld	a0,32(s0)                                                          # executed: 1
80000c4e:	6605                	lui	a2,0x1                                                            # executed: 1
80000c50:	d4bff0ef          	jal	8000099a <requantize_shift_i64_to_i32>                              # executed: 1
80000c54:	7014                	ld	a3,32(s0)                                                          # executed: 1
80000c56:	6490                	ld	a2,8(s1)                                                           # executed: 1
80000c58:	608c                	ld	a1,0(s1)                                                           # executed: 1
80000c5a:	6c08                	ld	a0,24(s0)                                                          # executed: 1
80000c5c:	0284b803          	ld	a6,40(s1)                                                            # executed: 1
80000c60:	08000793          	li	a5,128                                                               # executed: 1
80000c64:	02000713          	li	a4,32                                                                # executed: 1
80000c68:	bc7ff0ef          	jal	8000082e <layernorm_affine_i32>                                     # executed: 1
80000c6c:	7008                	ld	a0,32(s0)                                                          # executed: 1
80000c6e:	7402                	ld	s0,32(sp)                                                          # executed: 1
80000c70:	5cdc                	lw	a5,60(s1)                                                          # executed: 1
80000c72:	5c98                	lw	a4,56(s1)                                                          # executed: 1
80000c74:	7894                	ld	a3,48(s1)                                                          # executed: 1
80000c76:	70a2                	ld	ra,40(sp)                                                          # executed: 1
80000c78:	64e2                	ld	s1,24(sp)                                                          # executed: 1
80000c7a:	6942                	ld	s2,16(sp)                                                          # executed: 1
80000c7c:	85ce                	mv	a1,s3                                                              # executed: 1
80000c7e:	69a2                	ld	s3,8(sp)                                                           # executed: 1
80000c80:	6605                	lui	a2,0x1                                                            # executed: 1
80000c82:	6145                	addi	sp,sp,48                                                         # executed: 1
80000c84:	bb35                	j	800009c0 <requantize_i64_to_i8>                                     # executed: 1
```


The wrapper itself executes **74 instructions**, excluding all callee bodies. Stack saves retain registers and pointers across calls; the final helper may be tail-called after restoration. A tail call is counted as one wrapper jump followed by the helper's body and return, not a second wrapper epilogue.

#### Incoming registers

| Register on entry | C argument |
|---|---|
| `a0` | const int8_t *input |
| `a1` | int8_t *output |
| `a2` | LayerNormI8Workspace *workspace |

#### Retained values and stack frame

| Register | Role in this wrapper |
|---|---|
| `s0` | LayerNormI8Workspace pointer |
| `s1` | block0_ln1_params pointer, constructed before the first conversion |
| `s2` | Original INT8 input pointer |
| `s3` | Final INT8 T45 output pointer |

All six wrappers analyzed in the new notes allocate 48 bytes. They save `ra` at `40(sp)`, `s0` at `32(sp)`, `s1` at `24(sp)`, `s2` at `16(sp)`, and `s3` at `8(sp)`: 40 bytes of saved values plus eight bytes unused in this frame, keeping the 16-byte stack alignment. These saved slots hold register values, not activation tensors. The callee-saved registers retain values across helper calls, and the wrapper restores the caller's original values before its final tail transfer.

Workspace pointer fields are at offsets 0 (`mean_sum`), 8 (`variance`), 16 (`inv_std`), 24 (`feature_i32`), and 32 (`fine_i64`) from `s0`. Thus `ld a0,32(s0)` passes the INT64 scratch pointer; it does not load a tensor value. The parameter object similarly supplies gamma/beta pointers and conversion constants.

### 2.3 Helper arguments and data boundaries

Pointer arguments select the source tensors and caller-owned scratch shown in the C wrapper. Non-pointer arguments decoded at each actual call are:

| Call in source order | Actual non-pointer arguments |
|---|---|
| mean sums | T=32, F=128 |
| squared differences | T=32, F=128 |
| square conversion | N=4096, shift=15 |
| variance sums | T=32, F=128 |
| epsilon | T=32, epsilon=42 |
| inverse standard deviation | T=32, numerator=18,014,398,509,481,984 |
| normalize | T=32, F=128 |
| normalize conversion | N=4096, shift=15 |
| affine transform | T=32, F=128, beta alignment=65,536 |
| T45 conversion | N=4096, M=974,600, shift=47, output z=-17 |

## 3. Computation Kernels

### 3.1 `layernorm_sum_i8()` — Mean numerator

For each token, accumulate $S_t=\sum_{k=0}^{127}x_{t,k}$. The sum is retained without a separate division by 128. The inner loop at `0x8000069a–0x800006a2` has four instructions: `lb`, pointer `addi`, `addw`, and `bne`. There are 4,096 feature iterations. Six entry instructions, seven non-inner instructions per token, and one return give $7+T(7+4F)=16{,}615$.

#### Corresponding C

```c
void layernorm_sum_i8(
    const int8_t *input,
    int32_t *output,
    size_t token_count,
    size_t feature_count
)
{
    for (size_t t = 0; t < token_count; ++t) {
        const int8_t *row = input + t * feature_count;
        int32_t sum = 0;

        for (size_t k = 0; k < feature_count; ++k) {
            sum += (int32_t)row[k];
        }

        output[t] = sum;
    }
}
```


#### Compiled assembly

```asm
0000000080000684 <layernorm_sum_i8>:
80000684:	c239                	beqz	a2,800006ca <layernorm_sum_i8+0x46>                              # executed: 1
80000686:	060a                	slli	a2,a2,0x2                                                        # executed: 1
80000688:	00c58333          	add	t1,a1,a2                                                            # executed: 1
8000068c:	c685                	beqz	a3,800006b4 <layernorm_sum_i8+0x30>                              # executed: 1
8000068e:	00d50833          	add	a6,a0,a3                                                            # executed: 1
80000692:	4881                	li	a7,0                                                               # executed: 1
80000694:	011507b3          	add	a5,a0,a7                                                            # executed: 32
80000698:	4701                	li	a4,0                                                               # executed: 32
8000069a:	00078603          	lb	a2,0(a5)                                                             # executed: 4,096
8000069e:	0785                	addi	a5,a5,1                                                          # executed: 4,096
800006a0:	9f31                	addw	a4,a4,a2                                                         # executed: 4,096
800006a2:	ff079ce3          	bne	a5,a6,8000069a <layernorm_sum_i8+0x16>                              # executed: 4,096
800006a6:	c198                	sw	a4,0(a1)                                                           # executed: 32
800006a8:	0591                	addi	a1,a1,4                                                          # executed: 32
800006aa:	98b6                	add	a7,a7,a3                                                          # executed: 32
800006ac:	9836                	add	a6,a6,a3                                                          # executed: 32
800006ae:	feb313e3          	bne	t1,a1,80000694 <layernorm_sum_i8+0x10>                              # executed: 32
800006b2:	8082                	ret                                                                   # executed: 1
800006b4:	0005a023          	sw	zero,0(a1)                                                           # executed: 0
800006b8:	00458793          	addi	a5,a1,4                                                            # executed: 0
800006bc:	00f30763          	beq	t1,a5,800006ca <layernorm_sum_i8+0x46>                              # executed: 0
800006c0:	0005a223          	sw	zero,4(a1)                                                           # executed: 0
800006c4:	05a1                	addi	a1,a1,8                                                          # executed: 0
800006c6:	feb317e3          	bne	t1,a1,800006b4 <layernorm_sum_i8+0x30>                              # executed: 0
800006ca:	8082                	ret                                                                   # executed: 0
```


### 3.2 `layernorm_squared_difference_i8()` — Aligned squared differences

For each feature, compute $d_{t,k}=128x_{t,k}-S_t$, then store $d_{t,k}^2$ in INT64 scratch. `mul` first multiplies the input by the runtime feature count; a second `mul` squares the difference. The eight-instruction inner loop is `0x800006ea–0x80000702`: one load, two pointer updates, two multiplications, one subtraction, one INT64 store, and one branch. The count is $8+T(8+8F)=33{,}032$.

#### Corresponding C

```c
void layernorm_squared_difference_i8(
    const int8_t *input,
    const int32_t *mean_sum,
    int64_t *output,
    size_t token_count,
    size_t feature_count
)
{
    for (size_t t = 0; t < token_count; ++t) {
        const size_t offset = t * feature_count;
        const int64_t mean = (int64_t)mean_sum[t];

        for (size_t k = 0; k < feature_count; ++k) {
            const int64_t aligned_input =
                (int64_t)input[offset + k]
                * (int64_t)feature_count;

            const int64_t difference = aligned_input - mean;

            output[offset + k] = difference * difference;
        }
    }
}
```


#### Compiled assembly

```asm
00000000800006cc <layernorm_squared_difference_i8>:
800006cc:	c2b9                	beqz	a3,80000712 <layernorm_squared_difference_i8+0x46>               # executed: 1
800006ce:	068a                	slli	a3,a3,0x2                                                        # executed: 1
800006d0:	00d58eb3          	add	t4,a1,a3                                                            # executed: 1
800006d4:	c321                	beqz	a4,80000714 <layernorm_squared_difference_i8+0x48>               # executed: 1
800006d6:	00371f13          	slli	t5,a4,0x3                                                          # executed: 1
800006da:	00e508b3          	add	a7,a0,a4                                                            # executed: 1
800006de:	4e01                	li	t3,0                                                               # executed: 1
800006e0:	0005a303          	lw	t1,0(a1)                                                             # executed: 32
800006e4:	01c506b3          	add	a3,a0,t3                                                            # executed: 32
800006e8:	8832                	mv	a6,a2                                                              # executed: 32
800006ea:	00068783          	lb	a5,0(a3)                                                             # executed: 4,096
800006ee:	0685                	addi	a3,a3,1                                                          # executed: 4,096
800006f0:	0821                	addi	a6,a6,8                                                          # executed: 4,096
800006f2:	02e787b3          	mul	a5,a5,a4                                                            # executed: 4,096
800006f6:	406787b3          	sub	a5,a5,t1                                                            # executed: 4,096
800006fa:	02f787b3          	mul	a5,a5,a5                                                            # executed: 4,096
800006fe:	fef83c23          	sd	a5,-8(a6)                                                            # executed: 4,096
80000702:	ff1694e3          	bne	a3,a7,800006ea <layernorm_squared_difference_i8+0x1e>               # executed: 4,096
80000706:	0591                	addi	a1,a1,4                                                          # executed: 32
80000708:	967a                	add	a2,a2,t5                                                          # executed: 32
8000070a:	9e3a                	add	t3,t3,a4                                                          # executed: 32
8000070c:	98ba                	add	a7,a7,a4                                                          # executed: 32
8000070e:	fcbe99e3          	bne	t4,a1,800006e0 <layernorm_squared_difference_i8+0x14>               # executed: 32
80000712:	8082                	ret                                                                   # executed: 1
80000714:	00458793          	addi	a5,a1,4                                                            # executed: 0
80000718:	05a1                	addi	a1,a1,8                                                          # executed: 0
8000071a:	ffd78ce3          	beq	a5,t4,80000712 <layernorm_squared_difference_i8+0x46>               # executed: 0
8000071e:	ffd59be3          	bne	a1,t4,80000714 <layernorm_squared_difference_i8+0x48>               # executed: 0
80000722:	8082                	ret                                                                   # executed: 0
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

Case 0: P = 385, R = 639, S = 321, giving **4,966 instructions**.

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
800007a0:	8389                	srli	a5,a5,0x2                                                        # executed: 385
800007a2:	fef66fe3          	bltu	a2,a5,800007a0 <layernorm_rsqrt_i32+0x1e>                          # executed: 385
800007a6:	c39d                	beqz	a5,800007cc <layernorm_rsqrt_i32+0x4a>                           # executed: 32
800007a8:	4701                	li	a4,0                                                               # executed: 32
800007aa:	00e78833          	add	a6,a5,a4                                                            # executed: 639
800007ae:	8305                	srli	a4,a4,0x1                                                        # executed: 639
800007b0:	01066563          	bltu	a2,a6,800007ba <layernorm_rsqrt_i32+0x38>                          # executed: 639
800007b4:	41060633          	sub	a2,a2,a6                                                            # executed: 321
800007b8:	973e                	add	a4,a4,a5                                                          # executed: 321
800007ba:	8389                	srli	a5,a5,0x2                                                        # executed: 639
800007bc:	f7fd                	bnez	a5,800007aa <layernorm_rsqrt_i32+0x28>                           # executed: 639
800007be:	2701                	sext.w	a4,a4                                                          # executed: 32
800007c0:	c198                	sw	a4,0(a1)                                                           # executed: 32
800007c2:	0511                	addi	a0,a0,4                                                          # executed: 32
800007c4:	0591                	addi	a1,a1,4                                                          # executed: 32
800007c6:	fca896e3          	bne	a7,a0,80000792 <layernorm_rsqrt_i32+0x10>                           # executed: 32
800007ca:	8082                	ret                                                                   # executed: 1
800007cc:	4701                	li	a4,0                                                               # executed: 0
800007ce:	bfcd                	j	800007c0 <layernorm_rsqrt_i32+0x3e>                                 # executed: 0
```


### 3.6 `layernorm_normalize_i8()` — Fine normalized features

Calculate $(128x_{t,k}-S_t)r_t$ in INT64. The eight-instruction inner loop at `0x800007f2–0x8000080a` contains two `mul` instructions, one subtraction, one input load, two pointer updates, one store, and one branch. The count is $8+T(10+8F)=33{,}096$. Conversion back to INT32 remains a separate helper.

#### Corresponding C

```c
void layernorm_normalize_i8(
    const int8_t *input,
    const int32_t *mean_sum,
    const int32_t *inv_std,
    int64_t *output,
    size_t token_count,
    size_t feature_count
)
{
    for (size_t t = 0; t < token_count; ++t) {
        const size_t offset = t * feature_count;
        const int64_t mean = (int64_t)mean_sum[t];
        const int64_t inverse_std = (int64_t)inv_std[t];

        for (size_t k = 0; k < feature_count; ++k) {
            const int64_t aligned_input =
                (int64_t)input[offset + k]
                * (int64_t)feature_count;

            const int64_t difference = aligned_input - mean;

            output[offset + k] = difference * inverse_std;
        }
    }
}
```


#### Compiled assembly

```asm
00000000800007d0 <layernorm_normalize_i8>:
800007d0:	c731                	beqz	a4,8000081c <layernorm_normalize_i8+0x4c>                        # executed: 1
800007d2:	070a                	slli	a4,a4,0x2                                                        # executed: 1
800007d4:	00e58fb3          	add	t6,a1,a4                                                            # executed: 1
800007d8:	c3b9                	beqz	a5,8000081e <layernorm_normalize_i8+0x4e>                        # executed: 1
800007da:	00379293          	slli	t0,a5,0x3                                                          # executed: 1
800007de:	00f50333          	add	t1,a0,a5                                                            # executed: 1
800007e2:	4f01                	li	t5,0                                                               # executed: 1
800007e4:	0005ae83          	lw	t4,0(a1)                                                             # executed: 32
800007e8:	00062e03          	lw	t3,0(a2)                                                             # executed: 32
800007ec:	01e50833          	add	a6,a0,t5                                                            # executed: 32
800007f0:	88b6                	mv	a7,a3                                                              # executed: 32
800007f2:	00080703          	lb	a4,0(a6)                                                             # executed: 4,096
800007f6:	0805                	addi	a6,a6,1                                                          # executed: 4,096
800007f8:	08a1                	addi	a7,a7,8                                                          # executed: 4,096
800007fa:	02f70733          	mul	a4,a4,a5                                                            # executed: 4,096
800007fe:	41d70733          	sub	a4,a4,t4                                                            # executed: 4,096
80000802:	03c70733          	mul	a4,a4,t3                                                            # executed: 4,096
80000806:	fee8bc23          	sd	a4,-8(a7)                                                            # executed: 4,096
8000080a:	fe6814e3          	bne	a6,t1,800007f2 <layernorm_normalize_i8+0x22>                        # executed: 4,096
8000080e:	0591                	addi	a1,a1,4                                                          # executed: 32
80000810:	9696                	add	a3,a3,t0                                                          # executed: 32
80000812:	9f3e                	add	t5,t5,a5                                                          # executed: 32
80000814:	933e                	add	t1,t1,a5                                                          # executed: 32
80000816:	0611                	addi	a2,a2,4                                                          # executed: 32
80000818:	fcbf96e3          	bne	t6,a1,800007e4 <layernorm_normalize_i8+0x14>                        # executed: 32
8000081c:	8082                	ret                                                                   # executed: 1
8000081e:	00458793          	addi	a5,a1,4                                                            # executed: 0
80000822:	05a1                	addi	a1,a1,8                                                          # executed: 0
80000824:	fff78ce3          	beq	a5,t6,8000081c <layernorm_normalize_i8+0x4c>                        # executed: 0
80000828:	febf9be3          	bne	t6,a1,8000081e <layernorm_normalize_i8+0x4e>                        # executed: 0
8000082c:	8082                	ret                                                                   # executed: 0
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
| mean sum | 7 + T(7 + 4F) for INT8; 7 + T(6 + 4F) for INT32 | 16,615 |
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
| T45 conversion | 4,096 | 11N + 9 + C | 0 | 45,065 |
| **Total** |  |  |  | **102,423** |

Conversion kernels account for **39.94%** of this stage body in case 0. Computation kernels, conversion kernels, and wrapper instructions are counted separately.

### 5.3 Function totals and instruction breakdown

| Order | Work | Function | Case-0 instructions | Share of stage body |
|---|---|---|---|---|
| 0 | wrapper | `block0_ln1_forward` | 74 | 0.0289% |
| 1 | mean sums | `layernorm_sum_i8` | 16,615 | 6.4791% |
| 2 | squared differences | `layernorm_squared_difference_i8` | 33,032 | 12.8810% |
| 3 | square conversion | `requantize_shift_i64_to_i32` | 28,679 | 11.1835% |
| 4 | variance sums | `layernorm_variance_i32` | 16,583 | 6.4666% |
| 5 | epsilon | `layernorm_add_epsilon_i32` | 196 | 0.0764% |
| 6 | inverse standard deviation | `layernorm_rsqrt_i32` | 4,966 | 1.9365% |
| 7 | normalize | `layernorm_normalize_i8` | 33,096 | 12.9059% |
| 8 | normalize conversion | `requantize_shift_i64_to_i32` | 28,679 | 11.1835% |
| 9 | affine transform | `layernorm_affine_i32` | 49,455 | 19.2852% |
| 10 | T45 conversion | `requantize_i64_to_i8` | 45,065 | 17.5733% |
|  | **Total, including callees** |  | **256,440** | **100%** |

Across this stage, multiplication instructions (`mul` + `mulw`) total **28,672**. Register additions (`add` + `addw`) total **30,158**. These opcode totals include address/setup arithmetic where the compiler uses those mnemonics; they are not counts of mathematical MACs. Immediate additions (`addi` + `addiw`) are listed separately below.

| Disassembly mnemonic | Executed, case 0 | Function in this stage |
|---|---|---|
| `beqz` | 47 | Zero-size checks or zero-bit tests. |
| `addi` | 65,861 | Add an immediate: pointer/loop/stack updates, plus immediate numerical constants. |
| `sd` | 12,295 | Store INT64 intermediates or save registers. |
| `mv` | 137 | Copy a pointer/value or prepare call arguments; also select attention maxima. |
| `mul` | 28,672 | 64-bit low-product arithmetic: requantization, scaling, normalization, and occasional dimension setup. |
| `add` | 26,030 | 64-bit addition: numerical arithmetic and address/index calculations. |
| `slli` | 109 | Immediate left shift: byte offsets, alignment, or integer constants. |
| `li` | 120 | Load a constant, initialize a loop/accumulator, or assign a saturation endpoint. |
| `lb` | 12,288 | Load a signed INT8 activation or weight. |
| `addw` | 4,128 | 32-bit addition: accumulation, residual/elementwise arithmetic, or rounding-independent sums. |
| `bne` | 32,992 | Continue a counted pointer/index loop. |
| `ld` | 12,318 | Load INT64 scratch data, pointers, or saved registers. |
| `sw` | 8,320 | Store an INT32 result, intermediate, or probability. |
| `ret` | 10 | Return to the caller; charged to the returning helper. |
| `j` | 1 | Unconditional jump, including a tail call or loop transfer. |
| `lw` | 16,549 | Load a signed INT32 accumulator, probability, parameter, or LUT value. |
| `bge` | 8,192 | Signed comparisons for clipping, maximum selection, or the exponential cutoff. |
| `sext.w` | 32 | Sign-extend a 32-bit value to RV64. |
| `addiw` | 3 | 32-bit immediate addition: rounding-shift setup or LUT index calculation. |
| `sll` | 3 | Register-controlled left shift: rounding constants or probability scaling. |
| `lui` | 3 | Construct a constant or high part of an address. |
| `sub` | 8,545 | 64-bit subtraction: centering, normalization, interpolation, square-root work, or row addressing. |
| `sra` | 12,288 | Arithmetic right shift: rounded fixed-point conversion. |
| `srli` | 1,664 | Logical right shift: integer square-root iteration, sign extraction, or division setup. |
| `divu` | 32 | Unsigned integer division before the integer square root. |
| `bltu` | 1,056 | Unsigned tests in the exact integer square root. |
| `bnez` | 639 | Continue an integer square-root loop. |
| `sb` | 4,096 | Store an INT8 activation. |
| `jal` | 9 | Call a helper; its instruction is charged to the caller. |
| `auipc` | 1 | Construct a PC-relative parameter/table address. |
| **Total** | **256,440** |  |

### 5.4 Reconciliation with `minstret`

The supplied Spike counter interval contains **5 instructions outside the stage body**: the first counter read, argument setup, and the call. The ending read has not yet retired when its value is sampled. Instructions that subtract, save, or print the delta are outside this interval. This is specific to the supplied compiled call site.

```asm
80001a3c:	b0202af3          	csrr	s5,minstret
80001a40:	65ca                	ld	a1,144(sp)
80001a42:	0090                	addi	a2,sp,64
80001a44:	854a                	mv	a0,s2
80001a46:	97cff0ef          	jal	80000bc2 <block0_ln1_forward>
80001a4a:	b02027f3          	csrr	a5,minstret
```


| Case | Supplied raw minstret | Caller/measurement overhead | Counted stage body + callees | Unexplained difference |
|---|---|---|---|---|
| 0 | 256,445 | 5 | 256,440 | 0 |
| 1 | 256,447 | 5 | 256,442 | 0 |
| 2 | 256,458 | 5 | 256,453 | 0 |
| 3 | 256,488 | 5 | 256,483 | 0 |
| 4 | 256,485 | 5 | 256,480 | 0 |
| 5 | 256,453 | 5 | 256,448 | 0 |
| 6 | 256,484 | 5 | 256,479 | 0 |
| 7 | 256,449 | 5 | 256,444 | 0 |
| **Mean** | 256,463.625 | 5 | 256,458.625 | **0** |


Every case reconciles exactly. The body count includes all helper calls and their returns, and each wrapper instruction once per actual execution. The separately run staged and full paths both passed the supplied T107 comparisons.

| Case | rsqrt instructions | Final INT8 clipped elements |
|---|---|---|
| 0 | 4,966 | 0 |
| 1 | 4,968 | 0 |
| 2 | 4,979 | 0 |
| 3 | 5,009 | 0 |
| 4 | 5,006 | 0 |
| 5 | 4,974 | 0 |
| 6 | 5,005 | 0 |
| 7 | 4,969 | 1 |

Variation is localized to the exact integer square root and occasional final clipping; all other helper counts are fixed for these shapes.

## 6. Findings

| Region | Case-0 instructions | Share of stage body |
|---|---|---|
| Computation | 153,943 | 60.03% |
| Requantization | 102,423 | 39.94% |
| Wrapper | 74 | 0.03% |

The largest single helper is affine (49,455 instructions), followed by final INT8 conversion (45,065 in case 0). The two shift conversions together use 57,358 instructions. Each LN stage is about 0.45% of full-block raw instructions, so it is a lower-priority target for reducing the full-block instruction count. Input-dependent integer-square-root branches explain most small count differences.

Instruction shares identify where instructions are executed. They do not establish load latency, multiplier latency, cache behavior, or CV-Wally cycle speedup. No accelerator or custom-instruction speedup is measured here. The eight fixture passes validate these supplied integer cases; they do not establish equality with the original mixed TFLite graph or unseen-data quality.

[Back to results](Result.md) · [Counting method and validation](analysis_method.md)
