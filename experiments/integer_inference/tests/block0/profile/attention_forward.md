# Part 3: Attention — Disassembly Analysis

## 1. Function Overview

This note analyzes `block0_attention_forward()` and every helper it actually executes in the **supplied compiled ELF**. Case 0 provides the detailed instruction mix; Section 5.4 reconciles all eight cases with the supplied Spike `minstret` results. Counts describe dynamic executions, not the number of lines in the disassembly. See [counting method](analysis_method.md).

| Parameter | Current block-0 value |
|---|---|
| heads / tokens / head dimension | 16 / 32 / 8 |
| Q / K / V zero points | -12 / -30 / 4 |
| score / probability / context elements | 16,384 / 16,384 / 4,096 |
| probability scale | 2^-16, INT32 |
| exponential coordinate multiplier / shift | 503,030,354 / 24 |
| exponential LUT | 4,097 INT32 entries |
| context conversion | See current parameters and call argument table below |


## 2. Block Wrapper: `block0_attention_forward()`

### 2.1 Source and operation order

```c
void block0_attention_forward(
    const int8_t *query,
    const int8_t *key,
    const int8_t *value,
    int8_t *output,
    int32_t *score_workspace,
    int32_t *context_workspace
)
{
    /* Q × K^T, using the original projection layouts. */
    block0_attention_scores_forward(
        query,
        key,
        score_workspace
    );

    /* Reuse the score buffer for shifted scores. */
    attention_causal_shift_i32(
        score_workspace,
        score_workspace,
        BLOCK0_HEAD_COUNT,
        BLOCK0_TOKEN_COUNT
    );

    /* Reuse it for exponentials. */
    attention_exp_lut_i32(
        score_workspace,
        score_workspace,
        BLOCK0_HEAD_COUNT,
        BLOCK0_TOKEN_COUNT,
        block0_softmax_exp_lut,
        BLOCK0_SOFTMAX_COORD_MULTIPLIER,
        BLOCK0_SOFTMAX_COORD_SHIFT
    );

    /* Reuse it for probabilities. */
    attention_normalize_exp_i32(
        score_workspace,
        score_workspace,
        BLOCK0_HEAD_COUNT,
        BLOCK0_TOKEN_COUNT,
        BLOCK0_SOFTMAX_PROB_FRAC_BITS
    );

    /* Probability × V; write directly in token-major order. */
    attention_context_i8(
        score_workspace,
        value,
        context_workspace,
        BLOCK0_TOKEN_COUNT,
        BLOCK0_HEAD_COUNT,
        BLOCK0_HEAD_DIM,
        BLOCK0_V_OUTPUT_ZERO_POINT
    );

    /* Separate requantization produces the final T70 layout. */
    requantize_i32_to_i8(
        context_workspace,
        output,
        BLOCK0_TOKEN_COUNT * BLOCK0_HIDDEN_SIZE,
        BLOCK0_ATTN_CONTEXT_MULTIPLIER,
        BLOCK0_ATTN_CONTEXT_SHIFT,
        BLOCK0_ATTN_CONTEXT_OUTPUT_ZERO_POINT
    );
}
```


### 2.2 Compiled wrapper

The assembly below includes the wrapper only. Each `jal` or tail `j` transfers execution to a helper whose work is counted separately. `executed` is the case-0 execution frequency of that instruction. A helper reused several times has aggregated frequencies in its own listing. Unexecuted alternate paths are shown with zero.

```asm
0000000080000d58 <block0_attention_forward>:
80000d58:	7179                	addi	sp,sp,-48                                                        # executed: 1
80000d5a:	f022                	sd	s0,32(sp)                                                          # executed: 1
80000d5c:	ec26                	sd	s1,24(sp)                                                          # executed: 1
80000d5e:	e84a                	sd	s2,16(sp)                                                          # executed: 1
80000d60:	e44e                	sd	s3,8(sp)                                                           # executed: 1
80000d62:	843a                	mv	s0,a4                                                              # executed: 1
80000d64:	5889                	li	a7,-30                                                             # executed: 1
80000d66:	84be                	mv	s1,a5                                                              # executed: 1
80000d68:	89b2                	mv	s3,a2                                                              # executed: 1
80000d6a:	8936                	mv	s2,a3                                                              # executed: 1
80000d6c:	863a                	mv	a2,a4                                                              # executed: 1
80000d6e:	5851                	li	a6,-12                                                             # executed: 1
80000d70:	47a1                	li	a5,8                                                               # executed: 1
80000d72:	4741                	li	a4,16                                                              # executed: 1
80000d74:	02000693          	li	a3,32                                                                # executed: 1
80000d78:	f406                	sd	ra,40(sp)                                                          # executed: 1
80000d7a:	bbcff0ef          	jal	80000136 <attention_scores_i8>                                      # executed: 1
80000d7e:	85a2                	mv	a1,s0                                                              # executed: 1
80000d80:	8522                	mv	a0,s0                                                              # executed: 1
80000d82:	02000693          	li	a3,32                                                                # executed: 1
80000d86:	4641                	li	a2,16                                                              # executed: 1
80000d88:	cceff0ef          	jal	80000256 <attention_causal_shift_i32>                               # executed: 1
80000d8c:	1dfba7b7          	lui	a5,0x1dfba                                                          # executed: 1
80000d90:	85a2                	mv	a1,s0                                                              # executed: 1
80000d92:	8522                	mv	a0,s0                                                              # executed: 1
80000d94:	25278793          	addi	a5,a5,594 # 1dfba252 <_start-0x62045dae>                           # executed: 1
80000d98:	4861                	li	a6,24                                                              # executed: 1
80000d9a:	00026717          	auipc	a4,0x26                                                           # executed: 1
80000d9e:	5f670713          	addi	a4,a4,1526 # 80027390 <block0_softmax_exp_lut>                     # executed: 1
80000da2:	02000693          	li	a3,32                                                                # executed: 1
80000da6:	4641                	li	a2,16                                                              # executed: 1
80000da8:	d72ff0ef          	jal	8000031a <attention_exp_lut_i32>                                    # executed: 1
80000dac:	4741                	li	a4,16                                                              # executed: 1
80000dae:	85a2                	mv	a1,s0                                                              # executed: 1
80000db0:	8522                	mv	a0,s0                                                              # executed: 1
80000db2:	863a                	mv	a2,a4                                                              # executed: 1
80000db4:	02000693          	li	a3,32                                                                # executed: 1
80000db8:	e6aff0ef          	jal	80000422 <attention_normalize_exp_i32>                              # executed: 1
80000dbc:	85ce                	mv	a1,s3                                                              # executed: 1
80000dbe:	8522                	mv	a0,s0                                                              # executed: 1
80000dc0:	8626                	mv	a2,s1                                                              # executed: 1
80000dc2:	47a1                	li	a5,8                                                               # executed: 1
80000dc4:	4741                	li	a4,16                                                              # executed: 1
80000dc6:	02000693          	li	a3,32                                                                # executed: 1
80000dca:	4811                	li	a6,4                                                               # executed: 1
80000dcc:	f1cff0ef          	jal	800004e8 <attention_context_i8>                                     # executed: 1
80000dd0:	7402                	ld	s0,32(sp)                                                          # executed: 1
80000dd2:	70a2                	ld	ra,40(sp)                                                          # executed: 1
80000dd4:	69a2                	ld	s3,8(sp)                                                           # executed: 1
80000dd6:	85ca                	mv	a1,s2                                                              # executed: 1
80000dd8:	8526                	mv	a0,s1                                                              # executed: 1
80000dda:	6942                	ld	s2,16(sp)                                                          # executed: 1
80000ddc:	64e2                	ld	s1,24(sp)                                                          # executed: 1
80000dde:	425a66b7          	lui	a3,0x425a6                                                          # executed: 1
80000de2:	f8068693          	addi	a3,a3,-128 # 425a5f80 <_start-0x3da5a080>                          # executed: 1
80000de6:	47a5                	li	a5,9                                                               # executed: 1
80000de8:	02e00713          	li	a4,46                                                                # executed: 1
80000dec:	6605                	lui	a2,0x1                                                            # executed: 1
80000dee:	6145                	addi	sp,sp,48                                                         # executed: 1
80000df0:	b919                	j	80000a06 <requantize_i32_to_i8>                                     # executed: 1
```


The wrapper itself executes **60 instructions**, excluding all callee bodies. Stack saves retain registers and pointers across calls; the final helper may be tail-called after restoration. A tail call is counted as one wrapper jump followed by the helper's body and return, not a second wrapper epilogue.

`block0_attention_scores_forward()` is present as a separate symbol, but it is **inlined in this wrapper**. The executed instruction at the call site calls `attention_scores_i8()` directly. Do not charge the uncalled standalone wrapper as additional work.

#### Incoming registers

| Register on entry | C argument |
|---|---|
| `a0` | const int8_t *query |
| `a1` | const int8_t *key |
| `a2` | const int8_t *value |
| `a3` | int8_t *output |
| `a4` | int32_t *score_workspace |
| `a5` | int32_t *context_workspace |

#### Retained values and stack frame

| Register | Role in this wrapper |
|---|---|
| `s0` | Score/exponential/probability workspace pointer |
| `s1` | INT32 context workspace pointer |
| `s2` | INT8 T70 output pointer |
| `s3` | INT8 V input pointer retained until context computation |

All six wrappers analyzed in the new notes allocate 48 bytes. They save `ra` at `40(sp)`, `s0` at `32(sp)`, `s1` at `24(sp)`, `s2` at `16(sp)`, and `s3` at `8(sp)`: 40 bytes of saved values plus eight bytes unused in this frame, keeping the 16-byte stack alignment. These saved slots hold register values, not activation tensors. The callee-saved registers retain values across helper calls, and the wrapper restores the caller's original values before its final tail transfer.

### 2.3 Helper arguments and data boundaries

Pointer arguments select the source tensors and caller-owned scratch shown in the C wrapper. Non-pointer arguments decoded at each actual call are:

| Call in source order | Actual non-pointer arguments |
|---|---|
| Q × Kᵀ scores | T=32, H=16, D=8, Q z=-12, K z=-30 |
| causal maximum and shift | H=16, T=32 |
| exponential LUT | H=16, T=32, coordinate M=503,030,354, shift=24 |
| probability normalization | H=16, T=32, probability fractional bits=16 |
| probability × V context | T=32, H=16, D=8, V z=4 |
| T70 conversion | N=4096, M=1,113,218,944, shift=46, output z=9 |

## 3. Computation Kernels

### 3.1 `attention_scores_i8()` — Q × Kᵀ

For each head, query token, and key token, compute
$$
S_{h,t,j}=\sum_{d=0}^{7}(Q_{t,8h+d}+12)(K_{j,8h+d}+30).
$$
Q and K stay token-major. The pointer calculations implement the head selection and transpose view without allocating a transposed tensor. Four loops give $16\times32\times32\times8=131{,}072$ mathematical MACs.

The inner loop at `0x800001b6–0x800001d2` has **nine** instructions: two signed byte loads, two zero-point subtractions, one `mulw`, one accumulation `addw`, two pointer updates, and one `bne`. It executes 1,179,648 instructions. Setup, outer-loop control, stores, and exit add 137,646, giving **1,317,294**. Two additional `mul` instructions construct dimensions; they are not tensor MACs.

The compiled code computes the entire 32 × 32 score matrix per head. Causal masking occurs afterward, so future-key score MACs are included in this baseline.

#### Corresponding C

```c
void attention_scores_i8(
    const int8_t *query,
    const int8_t *key,
    int32_t *scores,
    size_t token_count,
    size_t head_count,
    size_t head_dim,
    int32_t query_zero_point,
    int32_t key_zero_point
)
{
    const size_t hidden_size = head_count * head_dim;

    for (size_t h = 0; h < head_count; ++h) {
        const size_t head_offset = h * head_dim;

        for (size_t t = 0; t < token_count; ++t) {
            const int8_t *query_row =
                query + t * hidden_size + head_offset;

            int32_t *score_row =
                scores + (h * token_count + t) * token_count;

            for (size_t j = 0; j < token_count; ++j) {
                const int8_t *key_row =
                    key + j * hidden_size + head_offset;

                int32_t accumulator = 0;

                for (size_t d = 0; d < head_dim; ++d) {
                    const int32_t q_centered =
                        (int32_t)query_row[d] - query_zero_point;

                    const int32_t k_centered =
                        (int32_t)key_row[d] - key_zero_point;

                    accumulator += q_centered * k_centered;
                }

                score_row[j] = accumulator;
            }
        }
    }
}
```


#### Compiled assembly

```asm
0000000080000136 <attention_scores_i8>:
80000136:	10070f63          	beqz	a4,80000254 <attention_scores_i8+0x11e>                            # executed: 1
8000013a:	7159                	addi	sp,sp,-112                                                       # executed: 1
8000013c:	f4a2                	sd	s0,104(sp)                                                         # executed: 1
8000013e:	8f3a                	mv	t5,a4                                                              # executed: 1
80000140:	8436                	mv	s0,a3                                                              # executed: 1
80000142:	10068163          	beqz	a3,80000244 <attention_scores_i8+0x10e>                            # executed: 1
80000146:	f0a6                	sd	s1,96(sp)                                                          # executed: 1
80000148:	f85e                	sd	s7,48(sp)                                                          # executed: 1
8000014a:	84be                	mv	s1,a5                                                              # executed: 1
8000014c:	02d68bb3          	mul	s7,a3,a3                                                            # executed: 1
80000150:	82aa                	mv	t0,a0                                                              # executed: 1
80000152:	00950eb3          	add	t4,a0,s1                                                            # executed: 1
80000156:	ecca                	sd	s2,88(sp)                                                          # executed: 1
80000158:	00269913          	slli	s2,a3,0x2                                                          # executed: 1
8000015c:	e8ce                	sd	s3,80(sp)                                                          # executed: 1
8000015e:	e4d2                	sd	s4,72(sp)                                                          # executed: 1
80000160:	e0d6                	sd	s5,64(sp)                                                          # executed: 1
80000162:	fc5a                	sd	s6,56(sp)                                                          # executed: 1
80000164:	f462                	sd	s8,40(sp)                                                          # executed: 1
80000166:	02970533          	mul	a0,a4,s1                                                            # executed: 1
8000016a:	f066                	sd	s9,32(sp)                                                          # executed: 1
8000016c:	ec6a                	sd	s10,24(sp)                                                         # executed: 1
8000016e:	e86e                	sd	s11,16(sp)                                                         # executed: 1
80000170:	89b2                	mv	s3,a2                                                              # executed: 1
80000172:	01260b33          	add	s6,a2,s2                                                            # executed: 1
80000176:	4a81                	li	s5,0                                                               # executed: 1
80000178:	4a01                	li	s4,0                                                               # executed: 1
8000017a:	002b9693          	slli	a3,s7,0x2                                                          # executed: 1
8000017e:	87de                	mv	a5,s7                                                              # executed: 1
80000180:	8e2e                	mv	t3,a1                                                              # executed: 1
80000182:	83ba                	mv	t2,a4                                                              # executed: 1
80000184:	4f01                	li	t5,0                                                               # executed: 1
80000186:	01ee8fb3          	add	t6,t4,t5                                                            # executed: 16
8000018a:	832a                	mv	t1,a0                                                              # executed: 16
8000018c:	8c16                	mv	s8,t0                                                              # executed: 16
8000018e:	01e28533          	add	a0,t0,t5                                                            # executed: 16
80000192:	8c9e                	mv	s9,t2                                                              # executed: 16
80000194:	e452                	sd	s4,8(sp)                                                           # executed: 16
80000196:	83f6                	mv	t2,t4                                                              # executed: 16
80000198:	8bd6                	mv	s7,s5                                                              # executed: 16
8000019a:	8ebe                	mv	t4,a5                                                              # executed: 16
8000019c:	85da                	mv	a1,s6                                                              # executed: 16
8000019e:	82b6                	mv	t0,a3                                                              # executed: 16
800001a0:	4781                	li	a5,0                                                               # executed: 16
800001a2:	002a9693          	slli	a3,s5,0x2                                                          # executed: 512
800001a6:	96ce                	add	a3,a3,s3                                                          # executed: 512
800001a8:	c0d1                	beqz	s1,8000022c <attention_scores_i8+0xf6>                           # executed: 512
800001aa:	867a                	mv	a2,t5                                                              # executed: 512
800001ac:	e02e                	sd	a1,0(sp)                                                           # executed: 512
800001ae:	00ce0d33          	add	s10,t3,a2                                                           # executed: 16,384
800001b2:	8a2a                	mv	s4,a0                                                              # executed: 16,384
800001b4:	4d81                	li	s11,0                                                              # executed: 16,384
800001b6:	000a0583          	lb	a1,0(s4)                                                             # executed: 131,072
800001ba:	000d0703          	lb	a4,0(s10)                                                            # executed: 131,072
800001be:	0a05                	addi	s4,s4,1                                                          # executed: 131,072
800001c0:	410585bb          	subw	a1,a1,a6                                                           # executed: 131,072
800001c4:	4117073b          	subw	a4,a4,a7                                                           # executed: 131,072
800001c8:	02b7073b          	mulw	a4,a4,a1                                                           # executed: 131,072
800001cc:	0d05                	addi	s10,s10,1                                                        # executed: 131,072
800001ce:	01b70dbb          	addw	s11,a4,s11                                                         # executed: 131,072
800001d2:	fffa12e3          	bne	s4,t6,800001b6 <attention_scores_i8+0x80>                           # executed: 131,072
800001d6:	6702                	ld	a4,0(sp)                                                           # executed: 16,384
800001d8:	01b6a023          	sw	s11,0(a3)                                                            # executed: 16,384
800001dc:	0691                	addi	a3,a3,4                                                          # executed: 16,384
800001de:	961a                	add	a2,a2,t1                                                          # executed: 16,384
800001e0:	fcd717e3          	bne	a4,a3,800001ae <attention_scores_i8+0x78>                           # executed: 16,384
800001e4:	6582                	ld	a1,0(sp)                                                           # executed: 512
800001e6:	0785                	addi	a5,a5,1                                                          # executed: 512
800001e8:	9aa2                	add	s5,s5,s0                                                          # executed: 512
800001ea:	95ca                	add	a1,a1,s2                                                          # executed: 512
800001ec:	951a                	add	a0,a0,t1                                                          # executed: 512
800001ee:	9f9a                	add	t6,t6,t1                                                          # executed: 512
800001f0:	faf419e3          	bne	s0,a5,800001a2 <attention_scores_i8+0x6c>                           # executed: 512
800001f4:	6a22                	ld	s4,8(sp)                                                           # executed: 16
800001f6:	87f6                	mv	a5,t4                                                              # executed: 16
800001f8:	8696                	mv	a3,t0                                                              # executed: 16
800001fa:	0a05                	addi	s4,s4,1                                                          # executed: 16
800001fc:	8e9e                	mv	t4,t2                                                              # executed: 16
800001fe:	851a                	mv	a0,t1                                                              # executed: 16
80000200:	82e2                	mv	t0,s8                                                              # executed: 16
80000202:	83e6                	mv	t2,s9                                                              # executed: 16
80000204:	9f26                	add	t5,t5,s1                                                          # executed: 16
80000206:	00fb8ab3          	add	s5,s7,a5                                                            # executed: 16
8000020a:	9b36                	add	s6,s6,a3                                                          # executed: 16
8000020c:	f74c9de3          	bne	s9,s4,80000186 <attention_scores_i8+0x50>                           # executed: 16
80000210:	7486                	ld	s1,96(sp)                                                          # executed: 1
80000212:	6966                	ld	s2,88(sp)                                                          # executed: 1
80000214:	69c6                	ld	s3,80(sp)                                                          # executed: 1
80000216:	6a26                	ld	s4,72(sp)                                                          # executed: 1
80000218:	6a86                	ld	s5,64(sp)                                                          # executed: 1
8000021a:	7b62                	ld	s6,56(sp)                                                          # executed: 1
8000021c:	7bc2                	ld	s7,48(sp)                                                          # executed: 1
8000021e:	7c22                	ld	s8,40(sp)                                                          # executed: 1
80000220:	7c82                	ld	s9,32(sp)                                                          # executed: 1
80000222:	6d62                	ld	s10,24(sp)                                                         # executed: 1
80000224:	6dc2                	ld	s11,16(sp)                                                         # executed: 1
80000226:	7426                	ld	s0,104(sp)                                                         # executed: 1
80000228:	6165                	addi	sp,sp,112                                                        # executed: 1
8000022a:	8082                	ret                                                                   # executed: 1
8000022c:	0006a023          	sw	zero,0(a3)                                                           # executed: 0
80000230:	00468713          	addi	a4,a3,4                                                            # executed: 0
80000234:	fab709e3          	beq	a4,a1,800001e6 <attention_scores_i8+0xb0>                           # executed: 0
80000238:	0006a223          	sw	zero,4(a3)                                                           # executed: 0
8000023c:	06a1                	addi	a3,a3,8                                                          # executed: 0
8000023e:	fed597e3          	bne	a1,a3,8000022c <attention_scores_i8+0xf6>                           # executed: 0
80000242:	b755                	j	800001e6 <attention_scores_i8+0xb0>                                 # executed: 0
80000244:	00140793          	addi	a5,s0,1                                                            # executed: 0
80000248:	0409                	addi	s0,s0,2                                                          # executed: 0
8000024a:	fcff0ee3          	beq	t5,a5,80000226 <attention_scores_i8+0xf0>                           # executed: 0
8000024e:	fe8f1be3          	bne	t5,s0,80000244 <attention_scores_i8+0x10e>                          # executed: 0
80000252:	bfd1                	j	80000226 <attention_scores_i8+0xf0>                                 # executed: 0
80000254:	8082                	ret                                                                   # executed: 0
```


### 3.2 `attention_causal_shift_i32()` — Causal maximum, shift, and zero fill

Each row finds a maximum over keys $0\ldots t$, subtracts it from the allowed scores, and writes zero placeholders for future keys. There are $A=16\sum_{t=0}^{31}(t+1)=8{,}448$ allowed positions and $M=7{,}936$ masked positions. Maximum scans compare 7,936 additional candidates after each row's initial value.

The allowed-score loop `0x800002a0–0x800002ac` uses six instructions per position. The zero-fill loop `0x800002b8–0x800002be` uses three. The maximum loop `0x800002d4–0x800002e4` uses six fixed instructions per candidate, plus `mv a1,a4` at `0x800002de` when the candidate is smaller than the retained maximum. The compiler first copies the candidate, then conditionally restores the previous maximum.

Let L count those conditional restores. The count is $7{,}721+6A+3M+6(7{,}936)+L=129{,}833+L$. This conditional move explains almost all inter-case variation in attention.

#### Corresponding C

```c
void attention_causal_shift_i32(
    const int32_t *scores,
    int32_t *shifted,
    size_t head_count,
    size_t token_count
)
{
    for (size_t h = 0; h < head_count; ++h) {
        for (size_t t = 0; t < token_count; ++t) {
            const size_t row_offset =
                (h * token_count + t) * token_count;

            const int32_t *score_row = scores + row_offset;
            int32_t *shifted_row = shifted + row_offset;

            /* Only allowed keys participate in the maximum. */
            int32_t maximum = score_row[0];

            for (size_t j = 1; j <= t; ++j) {
                if (score_row[j] > maximum) {
                    maximum = score_row[j];
                }
            }

            /* Promote before subtraction; contract requires INT32 fit. */
            for (size_t j = 0; j <= t; ++j) {
                const int64_t difference =
                    (int64_t)score_row[j] - (int64_t)maximum;

                shifted_row[j] = (int32_t)difference;
            }

            /* These placeholders must be ignored by softmax. */
            for (size_t j = t + 1; j < token_count; ++j) {
                shifted_row[j] = 0;
            }
        }
    }
}
```


#### Compiled assembly

```asm
0000000080000256 <attention_causal_shift_i32>:
80000256:	c269                	beqz	a2,80000318 <attention_causal_shift_i32+0xc2>                    # executed: 1
80000258:	7179                	addi	sp,sp,-48                                                        # executed: 1
8000025a:	e84e                	sd	s3,16(sp)                                                          # executed: 1
8000025c:	83b6                	mv	t2,a3                                                              # executed: 1
8000025e:	89b2                	mv	s3,a2                                                              # executed: 1
80000260:	c6d1                	beqz	a3,800002ec <attention_causal_shift_i32+0x96>                    # executed: 1
80000262:	02d687b3          	mul	a5,a3,a3                                                            # executed: 1
80000266:	00269f93          	slli	t6,a3,0x2                                                          # executed: 1
8000026a:	f422                	sd	s0,40(sp)                                                          # executed: 1
8000026c:	f026                	sd	s1,32(sp)                                                          # executed: 1
8000026e:	ec4a                	sd	s2,24(sp)                                                          # executed: 1
80000270:	e452                	sd	s4,8(sp)                                                           # executed: 1
80000272:	862a                	mv	a2,a0                                                              # executed: 1
80000274:	00458413          	addi	s0,a1,4                                                            # executed: 1
80000278:	004f8293          	addi	t0,t6,4                                                            # executed: 1
8000027c:	ffcf8a13          	addi	s4,t6,-4                                                           # executed: 1
80000280:	4481                	li	s1,0                                                               # executed: 1
80000282:	00279913          	slli	s2,a5,0x2                                                          # executed: 1
80000286:	4214                	lw	a3,0(a2)                                                           # executed: 16
80000288:	ffc40e93          	addi	t4,s0,-4                                                           # executed: 16
8000028c:	008a08b3          	add	a7,s4,s0                                                            # executed: 16
80000290:	00460813          	addi	a6,a2,4                                                            # executed: 16
80000294:	8532                	mv	a0,a2                                                              # executed: 16
80000296:	8f22                	mv	t5,s0                                                              # executed: 16
80000298:	8332                	mv	t1,a2                                                              # executed: 16
8000029a:	4e01                	li	t3,0                                                               # executed: 16
8000029c:	85f6                	mv	a1,t4                                                              # executed: 512
8000029e:	879a                	mv	a5,t1                                                              # executed: 512
800002a0:	4398                	lw	a4,0(a5)                                                           # executed: 8,448
800002a2:	0791                	addi	a5,a5,4                                                          # executed: 8,448
800002a4:	0591                	addi	a1,a1,4                                                          # executed: 8,448
800002a6:	9f15                	subw	a4,a4,a3                                                         # executed: 8,448
800002a8:	fee5ae23          	sw	a4,-4(a1)                                                            # executed: 8,448
800002ac:	fef81ae3          	bne	a6,a5,800002a0 <attention_causal_shift_i32+0x4a>                    # executed: 8,448
800002b0:	0e05                	addi	t3,t3,1                                                          # executed: 512
800002b2:	05c38763          	beq	t2,t3,80000300 <attention_causal_shift_i32+0xaa>                    # executed: 512
800002b6:	87fa                	mv	a5,t5                                                              # executed: 496
800002b8:	0007a023          	sw	zero,0(a5)                                                           # executed: 7,936
800002bc:	0791                	addi	a5,a5,4                                                          # executed: 7,936
800002be:	fef89de3          	bne	a7,a5,800002b8 <attention_causal_shift_i32+0x62>                    # executed: 7,936
800002c2:	937e                	add	t1,t1,t6                                                          # executed: 496
800002c4:	00032703          	lw	a4,0(t1)                                                             # executed: 496
800002c8:	9f16                	add	t5,t5,t0                                                          # executed: 496
800002ca:	9efe                	add	t4,t4,t6                                                          # executed: 496
800002cc:	98fe                	add	a7,a7,t6                                                          # executed: 496
800002ce:	9816                	add	a6,a6,t0                                                          # executed: 496
800002d0:	9516                	add	a0,a0,t0                                                          # executed: 496
800002d2:	879a                	mv	a5,t1                                                              # executed: 496
800002d4:	43d4                	lw	a3,4(a5)                                                           # executed: 7,936
800002d6:	0791                	addi	a5,a5,4                                                          # executed: 7,936
800002d8:	85b6                	mv	a1,a3                                                              # executed: 7,936
800002da:	00e6d363          	bge	a3,a4,800002e0 <attention_causal_shift_i32+0x8a>                    # executed: 7,936
800002de:	85ba                	mv	a1,a4                                                              # executed: 6,812
800002e0:	0005871b          	sext.w	a4,a1                                                            # executed: 7,936
800002e4:	fea798e3          	bne	a5,a0,800002d4 <attention_causal_shift_i32+0x7e>                    # executed: 7,936
800002e8:	86ba                	mv	a3,a4                                                              # executed: 496
800002ea:	bf4d                	j	8000029c <attention_causal_shift_i32+0x46>                          # executed: 496
800002ec:	00138793          	addi	a5,t2,1                                                            # executed: 0
800002f0:	0389                	addi	t2,t2,2                                                          # executed: 0
800002f2:	00f98463          	beq	s3,a5,800002fa <attention_causal_shift_i32+0xa4>                    # executed: 0
800002f6:	fe799be3          	bne	s3,t2,800002ec <attention_causal_shift_i32+0x96>                    # executed: 0
800002fa:	69c2                	ld	s3,16(sp)                                                          # executed: 0
800002fc:	6145                	addi	sp,sp,48                                                         # executed: 0
800002fe:	8082                	ret                                                                   # executed: 0
80000300:	0485                	addi	s1,s1,1                                                          # executed: 16
80000302:	964a                	add	a2,a2,s2                                                          # executed: 16
80000304:	944a                	add	s0,s0,s2                                                          # executed: 16
80000306:	f89990e3          	bne	s3,s1,80000286 <attention_causal_shift_i32+0x30>                    # executed: 16
8000030a:	7422                	ld	s0,40(sp)                                                          # executed: 1
8000030c:	7482                	ld	s1,32(sp)                                                          # executed: 1
8000030e:	6962                	ld	s2,24(sp)                                                          # executed: 1
80000310:	6a22                	ld	s4,8(sp)                                                           # executed: 1
80000312:	69c2                	ld	s3,16(sp)                                                          # executed: 1
80000314:	6145                	addi	sp,sp,48                                                         # executed: 1
80000316:	8082                	ret                                                                   # executed: 1
80000318:	8082                	ret                                                                   # executed: 0
```


### 3.3 `attention_exp_lut_i32()` — Softmax exponential approximation

Only the 8,448 allowed positions calculate a LUT coordinate. The helper multiplies the shifted-score magnitude by 503,030,354, rounds, and shifts by 24. Coordinates have 16 fractional bits; the 4,097-entry exponential LUT has spacing 1/256. Two neighboring INT32 values and eight interpolation bits produce each exponential. Coordinates at or above $16\cdot2^{16}$ return zero.

The compiled allowed-position loop `0x80000384–0x800003be` has 14 always-executed instructions and seven interpolation instructions when below the cutoff. Future positions execute a three-instruction zero-fill loop. With J interpolated positions, the count is $7{,}277+14A+7J+3M$. Here J = A = 8,448 in all eight cases: no cutoff is taken, giving **208,493**. There are 8,448 coordinate and 8,448 interpolation multiplications, plus one dimension multiplication.

#### Corresponding C

```c
void attention_exp_lut_i32(
    const int32_t *shifted,
    int32_t *exponentials,
    size_t head_count,
    size_t token_count,
    const int32_t *exp_lut,
    int64_t coordinate_multiplier,
    unsigned int coordinate_shift
)
{
    const int64_t coordinate_rounding =
        INT64_C(1) << (coordinate_shift - 1u);

    /* 16 expressed at coordinate scale 2^-16. */
    const int64_t cutoff = INT64_C(16) << 16;

    for (size_t h = 0; h < head_count; ++h) {
        for (size_t t = 0; t < token_count; ++t) {
            const size_t row_offset =
                (h * token_count + t) * token_count;

            const int32_t *input_row = shifted + row_offset;
            int32_t *output_row = exponentials + row_offset;

            for (size_t j = 0; j <= t; ++j) {
                /* Promote before negation, including INT32_MIN. */
                const int64_t magnitude =
                    -(int64_t)input_row[j];

                const int64_t coordinate =
                    (magnitude * coordinate_multiplier
                     + coordinate_rounding)
                    >> coordinate_shift;

                if (coordinate >= cutoff) {
                    output_row[j] = 0;
                    continue;
                }

                /*
                 * Coordinate has 16 fractional bits.
                 * LUT spacing is 1/256: eight interpolation bits.
                 */
                const size_t index =
                    (size_t)(coordinate >> 8);

                const int64_t fraction =
                    coordinate & INT64_C(255);

                const int64_t lower = exp_lut[index];
                const int64_t upper = exp_lut[index + 1u];

                /* Nearest rounding, ties toward +infinity. */
                const int64_t interpolated =
                    lower
                    + (((upper - lower) * fraction
                        + INT64_C(128)) >> 8);

                output_row[j] = (int32_t)interpolated;
            }

            for (size_t j = t + 1; j < token_count; ++j) {
                output_row[j] = 0;
            }
        }
    }
}
```


#### Compiled assembly

```asm
000000008000031a <attention_exp_lut_i32>:
8000031a:	10060363          	beqz	a2,80000420 <attention_exp_lut_i32+0x106>                          # executed: 1
8000031e:	7159                	addi	sp,sp,-112                                                       # executed: 1
80000320:	f0a6                	sd	s1,96(sp)                                                          # executed: 1
80000322:	e0d6                	sd	s5,64(sp)                                                          # executed: 1
80000324:	84b6                	mv	s1,a3                                                              # executed: 1
80000326:	8ab2                	mv	s5,a2                                                              # executed: 1
80000328:	cecd                	beqz	a3,800003e2 <attention_exp_lut_i32+0xc8>                         # executed: 1
8000032a:	e86e                	sd	s11,16(sp)                                                         # executed: 1
8000032c:	02d68db3          	mul	s11,a3,a3                                                           # executed: 1
80000330:	e8ce                	sd	s3,80(sp)                                                          # executed: 1
80000332:	fff8069b          	addiw	a3,a6,-1                                                          # executed: 1
80000336:	4f85                	li	t6,1                                                               # executed: 1
80000338:	00249993          	slli	s3,s1,0x2                                                          # executed: 1
8000033c:	ecca                	sd	s2,88(sp)                                                          # executed: 1
8000033e:	e4d2                	sd	s4,72(sp)                                                          # executed: 1
80000340:	fc5a                	sd	s6,56(sp)                                                          # executed: 1
80000342:	f85e                	sd	s7,48(sp)                                                          # executed: 1
80000344:	f462                	sd	s8,40(sp)                                                          # executed: 1
80000346:	f066                	sd	s9,32(sp)                                                          # executed: 1
80000348:	ec6a                	sd	s10,24(sp)                                                         # executed: 1
8000034a:	f4a2                	sd	s0,104(sp)                                                         # executed: 1
8000034c:	8a2a                	mv	s4,a0                                                              # executed: 1
8000034e:	00df9fb3          	sll	t6,t6,a3                                                            # executed: 1
80000352:	8b2e                	mv	s6,a1                                                              # executed: 1
80000354:	00450c93          	addi	s9,a0,4                                                            # executed: 1
80000358:	0991                	addi	s3,s3,4                                                          # executed: 1
8000035a:	00249913          	slli	s2,s1,0x2                                                          # executed: 1
8000035e:	4c01                	li	s8,0                                                               # executed: 1
80000360:	4b81                	li	s7,0                                                               # executed: 1
80000362:	002d9d13          	slli	s10,s11,0x2                                                        # executed: 1
80000366:	001002b7          	lui	t0,0x100                                                            # executed: 1
8000036a:	004b0413          	addi	s0,s6,4                                                            # executed: 16
8000036e:	01690333          	add	t1,s2,s6                                                            # executed: 16
80000372:	83e2                	mv	t2,s8                                                              # executed: 16
80000374:	8e66                	mv	t3,s9                                                              # executed: 16
80000376:	8f5a                	mv	t5,s6                                                              # executed: 16
80000378:	4e81                	li	t4,0                                                               # executed: 16
8000037a:	00239513          	slli	a0,t2,0x2                                                          # executed: 512
8000037e:	9552                	add	a0,a0,s4                                                          # executed: 512
80000380:	88fa                	mv	a7,t5                                                              # executed: 512
80000382:	e47a                	sd	t5,8(sp)                                                           # executed: 512
80000384:	4114                	lw	a3,0(a0)                                                           # executed: 8,448
80000386:	4601                	li	a2,0                                                               # executed: 8,448
80000388:	0511                	addi	a0,a0,4                                                          # executed: 8,448
8000038a:	02f686b3          	mul	a3,a3,a5                                                            # executed: 8,448
8000038e:	40df86b3          	sub	a3,t6,a3                                                            # executed: 8,448
80000392:	4106d6b3          	sra	a3,a3,a6                                                            # executed: 8,448
80000396:	4086d593          	srai	a1,a3,0x8                                                          # executed: 8,448
8000039a:	058a                	slli	a1,a1,0x2                                                        # executed: 8,448
8000039c:	95ba                	add	a1,a1,a4                                                          # executed: 8,448
8000039e:	0ff6ff13          	zext.b	t5,a3                                                            # executed: 8,448
800003a2:	0056db63          	bge	a3,t0,800003b8 <attention_exp_lut_i32+0x9e>                         # executed: 8,448
800003a6:	4194                	lw	a3,0(a1)                                                           # executed: 8,448
800003a8:	41d0                	lw	a2,4(a1)                                                           # executed: 8,448
800003aa:	8e15                	sub	a2,a2,a3                                                          # executed: 8,448
800003ac:	03e60633          	mul	a2,a2,t5                                                            # executed: 8,448
800003b0:	08060613          	addi	a2,a2,128                                                          # executed: 8,448
800003b4:	8621                	srai	a2,a2,0x8                                                        # executed: 8,448
800003b6:	9e35                	addw	a2,a2,a3                                                         # executed: 8,448
800003b8:	00c8a023          	sw	a2,0(a7)                                                             # executed: 8,448
800003bc:	0891                	addi	a7,a7,4                                                          # executed: 8,448
800003be:	fdc513e3          	bne	a0,t3,80000384 <attention_exp_lut_i32+0x6a>                         # executed: 8,448
800003c2:	0e85                	addi	t4,t4,1                                                          # executed: 512
800003c4:	6f22                	ld	t5,8(sp)                                                           # executed: 512
800003c6:	03d48963          	beq	s1,t4,800003f8 <attention_exp_lut_i32+0xde>                         # executed: 512
800003ca:	86a2                	mv	a3,s0                                                              # executed: 496
800003cc:	0006a023          	sw	zero,0(a3)                                                           # executed: 7,936
800003d0:	0691                	addi	a3,a3,4                                                          # executed: 7,936
800003d2:	fe669de3          	bne	a3,t1,800003cc <attention_exp_lut_i32+0xb2>                         # executed: 7,936
800003d6:	944e                	add	s0,s0,s3                                                          # executed: 496
800003d8:	9f4a                	add	t5,t5,s2                                                          # executed: 496
800003da:	934a                	add	t1,t1,s2                                                          # executed: 496
800003dc:	9e4e                	add	t3,t3,s3                                                          # executed: 496
800003de:	93a6                	add	t2,t2,s1                                                          # executed: 496
800003e0:	bf69                	j	8000037a <attention_exp_lut_i32+0x60>                               # executed: 496
800003e2:	00148793          	addi	a5,s1,1                                                            # executed: 0
800003e6:	0489                	addi	s1,s1,2                                                          # executed: 0
800003e8:	00fa8463          	beq	s5,a5,800003f0 <attention_exp_lut_i32+0xd6>                         # executed: 0
800003ec:	fe9a9be3          	bne	s5,s1,800003e2 <attention_exp_lut_i32+0xc8>                         # executed: 0
800003f0:	7486                	ld	s1,96(sp)                                                          # executed: 0
800003f2:	6a86                	ld	s5,64(sp)                                                          # executed: 0
800003f4:	6165                	addi	sp,sp,112                                                        # executed: 0
800003f6:	8082                	ret                                                                   # executed: 0
800003f8:	0b85                	addi	s7,s7,1                                                          # executed: 16
800003fa:	9b6a                	add	s6,s6,s10                                                         # executed: 16
800003fc:	9cea                	add	s9,s9,s10                                                         # executed: 16
800003fe:	9c6e                	add	s8,s8,s11                                                         # executed: 16
80000400:	f77a95e3          	bne	s5,s7,8000036a <attention_exp_lut_i32+0x50>                         # executed: 16
80000404:	7426                	ld	s0,104(sp)                                                         # executed: 1
80000406:	6966                	ld	s2,88(sp)                                                          # executed: 1
80000408:	69c6                	ld	s3,80(sp)                                                          # executed: 1
8000040a:	6a26                	ld	s4,72(sp)                                                          # executed: 1
8000040c:	7b62                	ld	s6,56(sp)                                                          # executed: 1
8000040e:	7bc2                	ld	s7,48(sp)                                                          # executed: 1
80000410:	7c22                	ld	s8,40(sp)                                                          # executed: 1
80000412:	7c82                	ld	s9,32(sp)                                                          # executed: 1
80000414:	6d62                	ld	s10,24(sp)                                                         # executed: 1
80000416:	6dc2                	ld	s11,16(sp)                                                         # executed: 1
80000418:	7486                	ld	s1,96(sp)                                                          # executed: 1
8000041a:	6a86                	ld	s5,64(sp)                                                          # executed: 1
8000041c:	6165                	addi	sp,sp,112                                                        # executed: 1
8000041e:	8082                	ret                                                                   # executed: 1
80000420:	8082                	ret                                                                   # executed: 0
```


### 3.4 `attention_normalize_exp_i32()` — INT32 probability normalization

For each row, sum the allowed exponentials before overwriting the buffer. Then compute
$$
p_j=\left\lfloor\frac{e_j2^{16}+\lfloor D/2\rfloor}{D}\right\rfloor,\qquad D=\sum_{j=0}^{t}e_j.
$$
Multiplication by $2^{16}$ becomes the register shift `sll` at `0x8000048a`. Rounding `D/2` is compiled into shifts and an addition, not another division. The `div` at `0x80000490` executes **8,448** times, once per allowed probability. Masked positions are zero-filled.

The sum loop has four instructions per allowed position, the probability loop eight, and the masked zero-fill loop three. Including 8,268 other setup/row/head/exit instructions gives $8{,}268+4A+8A+3M=133{,}452$. One setup `mul` calculates a dimension.

#### Corresponding C

```c
void attention_normalize_exp_i32(
    const int32_t *exponentials,
    int32_t *probabilities,
    size_t head_count,
    size_t token_count,
    unsigned int probability_frac_bits
)
{
    const int64_t probability_one =
        INT64_C(1) << probability_frac_bits;

    for (size_t h = 0; h < head_count; ++h) {
        for (size_t t = 0; t < token_count; ++t) {
            const size_t row_offset =
                (h * token_count + t) * token_count;

            const int32_t *exp_row = exponentials + row_offset;
            int32_t *prob_row = probabilities + row_offset;

            int64_t denominator = 0;

            /* Finish reading the sum before overwriting any values. */
            for (size_t j = 0; j <= t; ++j) {
                denominator += (int64_t)exp_row[j];
            }

            const int64_t rounding = denominator / 2;

            for (size_t j = 0; j <= t; ++j) {
                const int64_t numerator =
                    (int64_t)exp_row[j] * probability_one;

                /* Positive division: nearest, ties toward +infinity. */
                prob_row[j] = (int32_t)(
                    (numerator + rounding) / denominator
                );
            }

            for (size_t j = t + 1; j < token_count; ++j) {
                prob_row[j] = 0;
            }
        }
    }
}
```


#### Compiled assembly

```asm
0000000080000422 <attention_normalize_exp_i32>:
80000422:	7179                	addi	sp,sp,-48                                                        # executed: 1
80000424:	ec4a                	sd	s2,24(sp)                                                          # executed: 1
80000426:	8932                	mv	s2,a2                                                              # executed: 1
80000428:	c245                	beqz	a2,800004c8 <attention_normalize_exp_i32+0xa6>                   # executed: 1
8000042a:	8636                	mv	a2,a3                                                              # executed: 1
8000042c:	c6d9                	beqz	a3,800004ba <attention_normalize_exp_i32+0x98>                   # executed: 1
8000042e:	02d687b3          	mul	a5,a3,a3                                                            # executed: 1
80000432:	00269393          	slli	t2,a3,0x2                                                          # executed: 1
80000436:	0391                	addi	t2,t2,4                                                          # executed: 1
80000438:	f422                	sd	s0,40(sp)                                                          # executed: 1
8000043a:	f026                	sd	s1,32(sp)                                                          # executed: 1
8000043c:	e84e                	sd	s3,16(sp)                                                          # executed: 1
8000043e:	e452                	sd	s4,8(sp)                                                           # executed: 1
80000440:	e056                	sd	s5,0(sp)                                                           # executed: 1
80000442:	00458413          	addi	s0,a1,4                                                            # executed: 1
80000446:	84aa                	mv	s1,a0                                                              # executed: 1
80000448:	ff838a93          	addi	s5,t2,-8                                                           # executed: 1
8000044c:	00269293          	slli	t0,a3,0x2                                                          # executed: 1
80000450:	4981                	li	s3,0                                                               # executed: 1
80000452:	00279a13          	slli	s4,a5,0x2                                                          # executed: 1
80000456:	ffc40f13          	addi	t5,s0,-4                                                           # executed: 16
8000045a:	008a8333          	add	t1,s5,s0                                                            # executed: 16
8000045e:	00448813          	addi	a6,s1,4                                                            # executed: 16
80000462:	8e26                	mv	t3,s1                                                              # executed: 16
80000464:	8fa2                	mv	t6,s0                                                              # executed: 16
80000466:	4e81                	li	t4,0                                                               # executed: 16
80000468:	87f2                	mv	a5,t3                                                              # executed: 512
8000046a:	4581                	li	a1,0                                                               # executed: 512
8000046c:	4394                	lw	a3,0(a5)                                                           # executed: 8,448
8000046e:	0791                	addi	a5,a5,4                                                          # executed: 8,448
80000470:	95b6                	add	a1,a1,a3                                                          # executed: 8,448
80000472:	ff079de3          	bne	a5,a6,8000046c <attention_normalize_exp_i32+0x4a>                   # executed: 8,448
80000476:	03f5d893          	srli	a7,a1,0x3f                                                         # executed: 512
8000047a:	98ae                	add	a7,a7,a1                                                          # executed: 512
8000047c:	4018d893          	srai	a7,a7,0x1                                                          # executed: 512
80000480:	857a                	mv	a0,t5                                                              # executed: 512
80000482:	86f2                	mv	a3,t3                                                              # executed: 512
80000484:	429c                	lw	a5,0(a3)                                                           # executed: 8,448
80000486:	0691                	addi	a3,a3,4                                                          # executed: 8,448
80000488:	0511                	addi	a0,a0,4                                                          # executed: 8,448
8000048a:	00e797b3          	sll	a5,a5,a4                                                            # executed: 8,448
8000048e:	97c6                	add	a5,a5,a7                                                          # executed: 8,448
80000490:	02b7c7b3          	div	a5,a5,a1                                                            # executed: 8,448
80000494:	fef52e23          	sw	a5,-4(a0)                                                            # executed: 8,448
80000498:	ff0696e3          	bne	a3,a6,80000484 <attention_normalize_exp_i32+0x62>                   # executed: 8,448
8000049c:	0e85                	addi	t4,t4,1                                                          # executed: 512
8000049e:	03d60863          	beq	a2,t4,800004ce <attention_normalize_exp_i32+0xac>                   # executed: 512
800004a2:	87fe                	mv	a5,t6                                                              # executed: 496
800004a4:	0007a023          	sw	zero,0(a5)                                                           # executed: 7,936
800004a8:	0791                	addi	a5,a5,4                                                          # executed: 7,936
800004aa:	fe679de3          	bne	a5,t1,800004a4 <attention_normalize_exp_i32+0x82>                   # executed: 7,936
800004ae:	9f9e                	add	t6,t6,t2                                                          # executed: 496
800004b0:	9f16                	add	t5,t5,t0                                                          # executed: 496
800004b2:	9316                	add	t1,t1,t0                                                          # executed: 496
800004b4:	9e16                	add	t3,t3,t0                                                          # executed: 496
800004b6:	981e                	add	a6,a6,t2                                                          # executed: 496
800004b8:	bf45                	j	80000468 <attention_normalize_exp_i32+0x46>                         # executed: 496
800004ba:	00160793          	addi	a5,a2,1                                                            # executed: 0
800004be:	0609                	addi	a2,a2,2                                                          # executed: 0
800004c0:	00f90463          	beq	s2,a5,800004c8 <attention_normalize_exp_i32+0xa6>                   # executed: 0
800004c4:	fec91be3          	bne	s2,a2,800004ba <attention_normalize_exp_i32+0x98>                   # executed: 0
800004c8:	6962                	ld	s2,24(sp)                                                          # executed: 0
800004ca:	6145                	addi	sp,sp,48                                                         # executed: 0
800004cc:	8082                	ret                                                                   # executed: 0
800004ce:	0985                	addi	s3,s3,1                                                          # executed: 16
800004d0:	9452                	add	s0,s0,s4                                                          # executed: 16
800004d2:	94d2                	add	s1,s1,s4                                                          # executed: 16
800004d4:	f93911e3          	bne	s2,s3,80000456 <attention_normalize_exp_i32+0x34>                   # executed: 16
800004d8:	7422                	ld	s0,40(sp)                                                          # executed: 1
800004da:	7482                	ld	s1,32(sp)                                                          # executed: 1
800004dc:	69c2                	ld	s3,16(sp)                                                          # executed: 1
800004de:	6a22                	ld	s4,8(sp)                                                           # executed: 1
800004e0:	6a82                	ld	s5,0(sp)                                                           # executed: 1
800004e2:	6962                	ld	s2,24(sp)                                                          # executed: 1
800004e4:	6145                	addi	sp,sp,48                                                         # executed: 1
800004e6:	8082                	ret                                                                   # executed: 1
```


### 3.5 `attention_context_i8()` — Probability × V

For each head, query token, and channel, compute
$$
C_{t,8h+d}=\sum_{j=0}^{31}p_{h,t,j}(V_{j,8h+d}-4).
$$
This is **INT32 probability × centered INT8 value** arithmetic. A signed INT8 × INT8 packed MAC cannot automatically replace it with unchanged numerical semantics.

The 16 × 32 × 8 × 32 loops perform 131,072 MACs. The eight-instruction loop `0x8000053e–0x80000556` has one `lb`, one `lw`, one centering `subw`, one `mulw`, one `addw`, two pointer updates (`addi` and register `add`), and one branch. Its count is 1,048,576; remaining work adds 32,978, giving **1,081,554**. Two setup `mul` instructions are dimension calculations. Every key is visited, including masked keys whose probabilities are zero. Context is written directly in token-major order.

#### Corresponding C

```c
void attention_context_i8(
    const int32_t *probabilities,
    const int8_t *value,
    int32_t *context,
    size_t token_count,
    size_t head_count,
    size_t head_dim,
    int32_t value_zero_point
)
{
    const size_t hidden_size = head_count * head_dim;

    for (size_t h = 0; h < head_count; ++h) {
        const size_t head_offset = h * head_dim;

        for (size_t t = 0; t < token_count; ++t) {
            const int32_t *probability_row =
                probabilities
                + (h * token_count + t) * token_count;

            int32_t *context_head =
                context + t * hidden_size + head_offset;

            for (size_t d = 0; d < head_dim; ++d) {
                int32_t accumulator = 0;

                for (size_t j = 0; j < token_count; ++j) {
                    const int32_t centered_value =
                        (int32_t)value[
                            j * hidden_size + head_offset + d
                        ] - value_zero_point;

                    accumulator +=
                        probability_row[j] * centered_value;
                }

                context_head[d] = accumulator;
            }
        }
    }
}
```


#### Compiled assembly

```asm
00000000800004e8 <attention_context_i8>:
800004e8:	c755                	beqz	a4,80000594 <attention_context_i8+0xac>                          # executed: 1
800004ea:	715d                	addi	sp,sp,-80                                                        # executed: 1
800004ec:	fc4a                	sd	s2,56(sp)                                                          # executed: 1
800004ee:	e85e                	sd	s7,16(sp)                                                          # executed: 1
800004f0:	893a                	mv	s2,a4                                                              # executed: 1
800004f2:	8bbe                	mv	s7,a5                                                              # executed: 1
800004f4:	02d687b3          	mul	a5,a3,a3                                                            # executed: 1
800004f8:	e4a2                	sd	s0,72(sp)                                                          # executed: 1
800004fa:	e0a6                	sd	s1,64(sp)                                                          # executed: 1
800004fc:	f84e                	sd	s3,48(sp)                                                          # executed: 1
800004fe:	f452                	sd	s4,40(sp)                                                          # executed: 1
80000500:	f056                	sd	s5,32(sp)                                                          # executed: 1
80000502:	ec5a                	sd	s6,24(sp)                                                          # executed: 1
80000504:	e462                	sd	s8,8(sp)                                                           # executed: 1
80000506:	8a32                	mv	s4,a2                                                              # executed: 1
80000508:	84b6                	mv	s1,a3                                                              # executed: 1
8000050a:	03790fb3          	mul	t6,s2,s7                                                            # executed: 1
8000050e:	8aaa                	mv	s5,a0                                                              # executed: 1
80000510:	00269413          	slli	s0,a3,0x2                                                          # executed: 1
80000514:	875e                	mv	a4,s7                                                              # executed: 1
80000516:	4981                	li	s3,0                                                               # executed: 1
80000518:	4c01                	li	s8,0                                                               # executed: 1
8000051a:	00279b13          	slli	s6,a5,0x2                                                          # executed: 1
8000051e:	c8b1                	beqz	s1,80000572 <attention_context_i8+0x8a>                          # executed: 16
80000520:	060b8b63          	beqz	s7,80000596 <attention_context_i8+0xae>                            # executed: 16
80000524:	01540f33          	add	t5,s0,s5                                                            # executed: 16
80000528:	86ce                	mv	a3,s3                                                              # executed: 16
8000052a:	87d6                	mv	a5,s5                                                              # executed: 16
8000052c:	4601                	li	a2,0                                                               # executed: 16
8000052e:	00269393          	slli	t2,a3,0x2                                                          # executed: 512
80000532:	93d2                	add	t2,t2,s4                                                          # executed: 512
80000534:	82ce                	mv	t0,s3                                                              # executed: 512
80000536:	00558e33          	add	t3,a1,t0                                                            # executed: 4,096
8000053a:	853e                	mv	a0,a5                                                              # executed: 4,096
8000053c:	4e81                	li	t4,0                                                               # executed: 4,096
8000053e:	000e0303          	lb	t1,0(t3)                                                             # executed: 131,072
80000542:	00052883          	lw	a7,0(a0)                                                             # executed: 131,072
80000546:	0511                	addi	a0,a0,4                                                          # executed: 131,072
80000548:	4103033b          	subw	t1,t1,a6                                                           # executed: 131,072
8000054c:	026888bb          	mulw	a7,a7,t1                                                           # executed: 131,072
80000550:	9e7e                	add	t3,t3,t6                                                          # executed: 131,072
80000552:	01d88ebb          	addw	t4,a7,t4                                                           # executed: 131,072
80000556:	ffe514e3          	bne	a0,t5,8000053e <attention_context_i8+0x56>                          # executed: 131,072
8000055a:	01d3a023          	sw	t4,0(t2)                                                             # executed: 4,096
8000055e:	0285                	addi	t0,t0,1 # 100001 <_start-0x7fefffff>                             # executed: 4,096
80000560:	0391                	addi	t2,t2,4                                                          # executed: 4,096
80000562:	fce29ae3          	bne	t0,a4,80000536 <attention_context_i8+0x4e>                          # executed: 4,096
80000566:	0605                	addi	a2,a2,1                                                          # executed: 512
80000568:	96fe                	add	a3,a3,t6                                                          # executed: 512
8000056a:	97a2                	add	a5,a5,s0                                                          # executed: 512
8000056c:	9f22                	add	t5,t5,s0                                                          # executed: 512
8000056e:	fcc490e3          	bne	s1,a2,8000052e <attention_context_i8+0x46>                          # executed: 512
80000572:	0c05                	addi	s8,s8,1                                                          # executed: 16
80000574:	975e                	add	a4,a4,s7                                                          # executed: 16
80000576:	99de                	add	s3,s3,s7                                                          # executed: 16
80000578:	9ada                	add	s5,s5,s6                                                          # executed: 16
8000057a:	fb8912e3          	bne	s2,s8,8000051e <attention_context_i8+0x36>                          # executed: 16
8000057e:	6426                	ld	s0,72(sp)                                                          # executed: 1
80000580:	6486                	ld	s1,64(sp)                                                          # executed: 1
80000582:	7962                	ld	s2,56(sp)                                                          # executed: 1
80000584:	79c2                	ld	s3,48(sp)                                                          # executed: 1
80000586:	7a22                	ld	s4,40(sp)                                                          # executed: 1
80000588:	7a82                	ld	s5,32(sp)                                                          # executed: 1
8000058a:	6b62                	ld	s6,24(sp)                                                          # executed: 1
8000058c:	6bc2                	ld	s7,16(sp)                                                          # executed: 1
8000058e:	6c22                	ld	s8,8(sp)                                                           # executed: 1
80000590:	6161                	addi	sp,sp,80                                                         # executed: 1
80000592:	8082                	ret                                                                   # executed: 1
80000594:	8082                	ret                                                                   # executed: 0
80000596:	4781                	li	a5,0                                                               # executed: 0
80000598:	00178693          	addi	a3,a5,1                                                            # executed: 0
8000059c:	0789                	addi	a5,a5,2                                                          # executed: 0
8000059e:	fcd48ae3          	beq	s1,a3,80000572 <attention_context_i8+0x8a>                          # executed: 0
800005a2:	fef49be3          	bne	s1,a5,80000598 <attention_context_i8+0xb0>                          # executed: 0
800005a6:	0c05                	addi	s8,s8,1                                                          # executed: 0
800005a8:	975e                	add	a4,a4,s7                                                          # executed: 0
800005aa:	99de                	add	s3,s3,s7                                                          # executed: 0
800005ac:	9ada                	add	s5,s5,s6                                                          # executed: 0
800005ae:	f78918e3          	bne	s2,s8,8000051e <attention_context_i8+0x36>                          # executed: 0
800005b2:	b7f1                	j	8000057e <attention_context_i8+0x96>                                # executed: 0
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


## 5. Complete Instruction Accounting

### 5.1 Loop and computation counts

| Region | Mathematical MACs | Instructions per MAC-loop iteration | MAC-loop instructions | Other helper instructions | Helper total |
|---|---|---|---|---|---|
| Q × Kᵀ scores | 131,072 | 9 | 1,179,648 | 137,646 | 1,317,294 |
| probability × V context | 131,072 | 8 | 1,048,576 | 32,978 | 1,081,554 |
| **Total** | **262,144** |  | **2,228,224** | **170,624** | **2,398,848** |

The other softmax helper counts are derived separately in Section 3: causal shift $129{,}833+L$, exponentials $7{,}277+14A+7J+3M$, and probability normalization $8{,}268+12A+3M$, with A = 8,448 allowed and M = 7,936 masked positions. Case 0 has L = 6,812 and J = 8,448.

Besides the **262,144 `mulw` MAC multiplications**, the stage executes **20,999 `mul`**: 16,896 exponential-coordinate/interpolation products, 4,096 context requantization products, and seven dimension-setup products. Total multiplication instructions: **283,143**. The 8,448 `div` instructions are softmax normalization, not MACs.

### 5.2 Requantization counts

| Conversion | Elements | Formula | C, case 0 | Executed instructions |
|---|---|---|---|---|
| T70 conversion | 4,096 | 11N + 9 + C | 0 | 45,065 |
| **Total** |  |  |  | **45,065** |

Conversion kernels account for **1.54%** of this stage body in case 0. Computation kernels, conversion kernels, and wrapper instructions are counted separately.

### 5.3 Function totals and instruction breakdown

| Order | Work | Function | Case-0 instructions | Share of stage body |
|---|---|---|---|---|
| 0 | wrapper | `block0_attention_forward` | 60 | 0.0021% |
| 1 | Q × Kᵀ scores | `attention_scores_i8` | 1,317,294 | 45.0732% |
| 2 | causal maximum and shift | `attention_causal_shift_i32` | 136,645 | 4.6755% |
| 3 | exponential LUT | `attention_exp_lut_i32` | 208,493 | 7.1339% |
| 4 | probability normalization | `attention_normalize_exp_i32` | 133,452 | 4.5663% |
| 5 | probability × V context | `attention_context_i8` | 1,081,554 | 37.0070% |
| 6 | T70 conversion | `requantize_i32_to_i8` | 45,065 | 1.5420% |
|  | **Total, including callees** |  | **2,922,563** | **100%** |

Across this stage, multiplication instructions (`mul` + `mulw`) total **283,143**. Register additions (`add` + `addw`) total **485,939**. These opcode totals include address/setup arithmetic where the compiler uses those mnemonics; they are not counts of mathematical MACs. Immediate additions (`addi` + `addiw`) are listed separately below.

| Disassembly mnemonic | Executed, case 0 | Function in this stage |
|---|---|---|
| `beqz` | 554 | Zero-size checks or zero-bit tests. |
| `addi` | 528,055 | Add an immediate: pointer/loop/stack updates, plus immediate numerical constants. |
| `sd` | 1,089 | Store INT64 intermediates or save registers. |
| `mv` | 42,229 | Copy a pointer/value or prepare call arguments; also select attention maxima. |
| `mul` | 20,999 | 64-bit low-product arithmetic: requantization, scaling, normalization, and occasional dimension setup. |
| `add` | 215,347 | 64-bit addition: numerical arithmetic and address/index calculations. |
| `slli` | 9,997 | Immediate left shift: byte offsets, alignment, or integer constants. |
| `li` | 29,551 | Load a constant, initialize a loop/accumulator, or assign a saturation endpoint. |
| `lb` | 393,216 | Load a signed INT8 activation or weight. |
| `subw` | 401,664 | 32-bit subtraction: input centering or attention score shifting. |
| `mulw` | 262,144 | 32-bit low-product arithmetic: projection/attention MAC or GELU products. |
| `addw` | 270,592 | 32-bit addition: accumulation, residual/elementwise arithmetic, or rounding-independent sums. |
| `bne` | 353,360 | Continue a counted pointer/index loop. |
| `ld` | 17,473 | Load INT64 scratch data, pointers, or saved registers. |
| `sw` | 69,632 | Store an INT32 result, intermediate, or probability. |
| `ret` | 6 | Return to the caller; charged to the returning helper. |
| `beq` | 1,536 | Loop-end or dimension comparisons. |
| `j` | 1,489 | Unconditional jump, including a tail call or loop transfer. |
| `lw` | 194,304 | Load a signed INT32 accumulator, probability, parameter, or LUT value. |
| `bge` | 24,576 | Signed comparisons for clipping, maximum selection, or the exponential cutoff. |
| `sext.w` | 7,936 | Sign-extend a 32-bit value to RV64. |
| `addiw` | 2 | 32-bit immediate addition: rounding-shift setup or LUT index calculation. |
| `sll` | 8,450 | Register-controlled left shift: rounding constants or probability scaling. |
| `lui` | 4 | Construct a constant or high part of an address. |
| `sub` | 16,896 | 64-bit subtraction: centering, normalization, interpolation, square-root work, or row addressing. |
| `sra` | 12,544 | Arithmetic right shift: rounded fixed-point conversion. |
| `srai` | 17,408 | Immediate arithmetic right shift: fixed-point conversion, interpolation, or signed division setup. |
| `zext.b` | 8,448 | Extract the 8-bit interpolation fraction. |
| `srli` | 512 | Logical right shift: integer square-root iteration, sign extraction, or division setup. |
| `div` | 8,448 | Signed integer division for softmax probability normalization. |
| `sb` | 4,096 | Store an INT8 activation. |
| `jal` | 5 | Call a helper; its instruction is charged to the caller. |
| `auipc` | 1 | Construct a PC-relative parameter/table address. |
| **Total** | **2,922,563** |  |

### 5.4 Reconciliation with `minstret`

The supplied Spike counter interval contains **8 instructions outside the stage body**: the first counter read, argument setup, and the call. The ending read has not yet retired when its value is sampled. Instructions that subtract, save, or print the delta are outside this interval. This is specific to the supplied compiled call site.

```asm
80001a70:	b0202af3          	csrr	s5,minstret
80001a74:	67e6                	ld	a5,88(sp)
80001a76:	672e                	ld	a4,200(sp)
80001a78:	76ca                	ld	a3,176(sp)
80001a7a:	762a                	ld	a2,168(sp)
80001a7c:	758a                	ld	a1,160(sp)
80001a7e:	656a                	ld	a0,152(sp)
80001a80:	ad8ff0ef          	jal	80000d58 <block0_attention_forward>
80001a84:	b02027f3          	csrr	a5,minstret
```


| Case | Supplied raw minstret | Caller/measurement overhead | Counted stage body + callees | Unexplained difference |
|---|---|---|---|---|
| 0 | 2,922,571 | 8 | 2,922,563 | 0 |
| 1 | 2,922,550 | 8 | 2,922,542 | 0 |
| 2 | 2,922,686 | 8 | 2,922,678 | 0 |
| 3 | 2,922,376 | 8 | 2,922,368 | 0 |
| 4 | 2,922,492 | 8 | 2,922,484 | 0 |
| 5 | 2,922,449 | 8 | 2,922,441 | 0 |
| 6 | 2,922,523 | 8 | 2,922,515 | 0 |
| 7 | 2,922,370 | 8 | 2,922,362 | 0 |
| **Mean** | 2,922,502.125 | 8 | 2,922,494.125 | **0** |


Every case reconciles exactly. The body count includes all helper calls and their returns, and each wrapper instruction once per actual execution. The separately run staged and full paths both passed the supplied T107 comparisons.

| Case | Conditional maximum restores (L) | Causal-shift instructions | T70 clipped elements |
|---|---|---|---|
| 0 | 6,812 | 136,645 | 0 |
| 1 | 6,791 | 136,624 | 0 |
| 2 | 6,927 | 136,760 | 0 |
| 3 | 6,617 | 136,450 | 0 |
| 4 | 6,733 | 136,566 | 0 |
| 5 | 6,689 | 136,522 | 1 |
| 6 | 6,764 | 136,597 | 0 |
| 7 | 6,611 | 136,444 | 0 |

The exponential cutoff is never taken in these fixtures. The only other observed variable instruction is one context-output clipping assignment in case 5.

## 6. Findings

| Region | Case-0 instructions | Share of stage body |
|---|---|---|
| Computation | 2,877,438 | 98.46% |
| Requantization | 45,065 | 1.54% |
| Wrapper | 60 | 0.00% |

The two attention matrix-product helpers account for **82.08%** of this stage. Their full-square loops include work later masked or multiplied by zero. This is a measured property of the existing scalar implementation. Changing causal loop bounds would be a separate software optimization requiring output verification. Softmax has 8,448 integer divisions; their cycle impact cannot be ranked from instruction count alone. Attention is about **5.11%** of full-block raw instructions.

Instruction shares identify where instructions are executed. They do not establish load latency, multiplier latency, cache behavior, or CV-Wally cycle speedup. No accelerator or custom-instruction speedup is measured here. The eight fixture passes validate these supplied integer cases; they do not establish equality with the original mixed TFLite graph or unseen-data quality.

[Back to results](Result.md) · [Counting method and validation](analysis_method.md)
