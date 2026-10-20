# TinyStories-3M Block 0: Integer Inference Reference

**Baseline date:** 2026-10-13. **Scope:** the fixed block-0 export and the Python integer candidate validated in this session.

This document specifies the integer computations from the block input through the final residual output. It records the selected representations, parameters, and conversion points in execution order. Original exports and previously verified candidates remain unchanged. The selected final GELU calculation uses the confirmed two-stage Op 70 rounding; its single-rounding comparison remains preserved.

## Contents

1. [Scope and source artifacts](#1-scope-and-source-artifacts)
2. [Block-level equations](#2-block-level-equations)
3. [Integer representation and arithmetic conventions](#3-integer-representation-and-arithmetic-conventions)
4. [Seven-part implementation](#4-seven-part-implementation)
5. [Consolidated conversion and format reference](#5-consolidated-conversion-and-format-reference)
6. [Validation and complete-block results](#6-validation-and-complete-block-results)
7. [Workload and storage facts](#7-workload-and-storage-facts)
8. [Completion status and reproduction](#8-completion-status-and-reproduction)
9. [Appendix: original graph and layout constants](#9-appendix-original-graph-and-layout-constants)

## 1. Scope and source artifacts

### 1.1 Model, dimensions, and target

| Item | Selected configuration |
|---|---|
| Model | TinyStories-3M / GPT-Neo |
| Block | 0 of 8 |
| Batch / sequence | 1 / 32 tokens |
| Hidden size | 128 |
| Attention heads / head dimension | 16 / 8 |
| MLP dimensions | 128 → 512 → 128 |
| Target | Bare-metal RV64IMC, without floating-point hardware |
| Working ISA / ABI | `rv64imc_zicsr_zifencei` / `lp64` |
| Current validation | Python integer candidates; one benchmark and seven calibration sequences |
| Planned architectural extensions | Packed MAC and requantization; not implemented for this candidate |

Dense weights are INT8. Normalization constants, probabilities, accumulators, and residuals use wider integer formats where specified. “W8A8” does not mean every intermediate or learned normalization constant is stored in INT8.

### 1.2 Fixed sources

| Item | Path or identifier |
|---|---|
| Project directory | `/home/randy/Documents/ACompArc/model/project` |
| FP32 checkpoint | `artifacts/models/tinystories_3m_fp32` |
| Fixed export directory | `artifacts/models/tinystories_3m_blocks_w8a8/20261011T134954617091Z` |
| Block file | `block0_w8a8.tflite` in the fixed export directory |
| Block-0 SHA-256 | `06481f6c5151a83e3c01896d314d902ee9608773d94834848777ccc7282d87a9` |
| Saved test inputs | `validation_inputs.npz` in the fixed export directory |
| Analysis source | Attached `analyze_model.ipynb` plus the subsequently executed cells and reported measurements in this session |

The benchmark uses `benchmark_input_ids` and saved `block0_input_fp32`. Seven calibration sequences use `calibration_input_ids`; their embedding inputs are reconstructed with the saved checkpoint. Each input has shape `[1,32,128]`. These are the existing benchmark/calibration cases; a separate unseen validation set has not been tested.

### 1.3 Reference and numeric provenance

The accuracy reference is the **fixed mixed-precision TFLite graph**, including its INT8 rounding and saturation. Its block input and final output are FP32. Candidate tensor IDs refer to corresponding positions in that graph; the candidate does not replace graph tensor types in the saved file.

Most scales below were reported at full precision. A few GELU scales were available only in rounded NumPy output. They are explicitly marked `≈` and kept symbolic in scale-ratio equations. **Do not recompute conversion constants from those rounded displays.** The verified integer multipliers, shifts, zero points, constant codes, and selected rounding algorithms are recorded exactly and define the tested candidate. Section 8 includes a read-only snippet to obtain full-precision metadata from the export.

This document specifies formats and equations, not the weight-array payloads. Read weight/bias arrays from the identified tensors in the fixed export. The actual exported scale metadata remains authoritative.

## 2. Block-level equations

Let $X\in\mathbb{R}^{32\times128}$ be the block input. Weights are stored as `[output_features,input_features]`, so a token-row matrix multiplies the transposed weight matrix.

### 2.1 LayerNorm

For token $t$ and $N=128$ features:

$$
\mu_t=\frac1N\sum_k X_{t,k},\qquad
v_t=\frac1N\sum_k(X_{t,k}-\mu_t)^2
$$

$$
\operatorname{LN}(X)_{t,k}
=\gamma_k(X_{t,k}-\mu_t)(v_t+\epsilon)^{-1/2}+\beta_k.
$$

The two LayerNorms use independent gamma/beta parameters. Their selected integer input and variance formats differ.

### 2.2 Attention and first residual

$$
H_0=\operatorname{LN}_1(X),\quad
Q=H_0W_Q^\mathsf{T},\quad K=H_0W_K^\mathsf{T},\quad V=H_0W_V^\mathsf{T}.
$$

Reshape Q/K/V into heads. For allowed causal entries:

$$
S_h=Q_hK_h^\mathsf{T},\qquad
P_{h,t,j}=\frac{\exp(S_{h,t,j}-m_{h,t})}
{\sum_{\ell\le t}\exp(S_{h,t,\ell}-m_{h,t})},\quad
m_{h,t}=\max_{j\le t}S_{h,t,j}.
$$

Masked probabilities are zero. **The inspected export has no additional attention score scaling; no $1/\sqrt8$ factor is inserted.**

$$
C_h=P_hV_h,\quad A=\operatorname{MergeHeads}(C),\quad
R=X+AW_O^\mathsf{T}+b_O.
$$

### 2.3 MLP and final residual

$$
H_1=\operatorname{LN}_2(R),\qquad
U=H_1W_1^\mathsf{T}+b_1,\qquad
Y=R+\operatorname{GELU}(U)W_2^\mathsf{T}+b_2.
$$

The exported GELU follows the half-input / cubic / tanh structure, with intermediate INT8 conversions. Its conceptual expression is:

$$
\frac{x}{2}\left[1+\tanh\left(\sqrt{2/\pi}\,(x+0.044715x^3)\right)\right].
$$

The candidate follows the saved intermediate formats and **saved quantized coefficient values**, rather than replacing that sequence with a direct evaluation of the conceptual expression.

## 3. Integer representation and arithmetic conventions

### 3.1 Stored values and scales

$$
x_{\mathrm{represented}}=(q_x-z_x)s_x.
$$

Here $q_x$ is the stored integer, $z_x$ the integer zero point, and $s_x$ the tensor scale. `INT32 @ 2^-16` means signed INT32 storage with $s=2^{-16}$; it does not prescribe a multiplication instruction at every use. For power-of-two scales, $F$ denotes the scale exponent: $s=2^{-F}$.

Input/parameter encoding performed offline is:

$$
q_x=\operatorname{round}_{\mathrm{even}}(x/s_x)+z_x,
$$

with clipping when the candidate specifies it. T0 is clipped to INT8. Reference FP32 values encoded for local residual/LayerNorm tests use their stated integer formats with range checks.

### 3.2 Storage width versus computation width

The dtype in a tensor table is its selected **stored representation**. Python often calculates a sum or product in INT64, checks its range, and casts to the chosen stored dtype. Promotion occurs **before** zero-point subtraction and multiplication. A centered INT8 value can exceed signed INT8 range; for example, T104 code 127 with zero point -88 represents centered integer 215.

Bounds described as “full domain” cover the specified input-code domains. Bounds checked on tested LayerNorm/residual values do not prove safety for arbitrary INT32 inputs.

### 3.3 Standard single-stage conversion

For centered source integer $a$:

$$
M=\frac{s_{\mathrm{source}}}{s_{\mathrm{destination}}}
\approx M_q2^{-r}.
$$

Define the tested signed rounding operation:

```text
RQ(a, Mq, r) = (int64(a) * Mq + 2^(r-1)) >> r
```

`>>` is an arithmetic signed right shift. This is nearest rounding with halfway values toward positive infinity. A right-shift-only conversion is `RQ(a,1,r)`.

An INT8 output is:

```text
q_out = clip(RQ(a, Mq, r) + z_out, -128, 127)
```

**The multiplier encoding $2^{-r}$ is separate from the destination tensor's scale.** The destination scale remains the value in its tensor-format table.

### 3.4 Execution action labels

| Label | Meaning |
|---|---|
| **ENCODE** | Offline conversion of a real input/parameter to a selected integer representation |
| **KEEP** | Retain integer values and representation; includes reshape/transpose |
| **ALIGN** | Exact multiplication by a power of two to put integers at a finer shared scale |
| **REQUANTIZE** | Convert between integer representations, with the specified multiplier/rounding/shift and optional clamp |
| **COMPUTE** | A mathematical operation with a derived output scale, such as multiplication, mean-through-scale, division, or integer square root |

A multiplication naturally produces a product scale $s_as_b$. It is not itself an additional requantization. Storing a feature sum at scale $s/128$ represents a mean without shifting that sum. Nonlinear computations have their own specified output formats.

### 3.5 Rounding and nonlinear rules

| Location | Selected rule |
|---|---|
| Offline inputs, gamma/beta, LUT entries, multiplier constants | Nearest, ties to even (`np.rint`) unless explicitly stated |
| Standard `RQ` and square conversion | Nearest, ties toward positive infinity |
| LayerNorm reciprocal square root | Integer division and `isqrt`, floor |
| Softmax probability division | Positive nearest rounding: `(numerator + denominator//2)//denominator` |
| Final GELU Op 70 | Rounded Q31 multiplication, then nearest power-of-two division with ties away from zero |

Op 70 has **one scale conversion implemented with two rounding stages**. Its tested result is specified separately from standard `RQ`.

Floating-point calculations in the analysis perform offline preparation or diagnostic reconstruction. The candidate calculations use integer arithmetic and integer table lookup. No new C implementation is supplied here.

## 4. Seven-part implementation

### 4.1 Part 1 — First LayerNorm, Op 0–14

#### A. Inputs, outputs, and parameters

Input T0 is encoded from the benchmark/calibration FP32 block input. Output T45 feeds all three projections.

| Tensor | Meaning | Shape | Candidate storage | Scale | Zero point |
|---|---|---|---|---|---:|
| T0 | Encoded block input | `[1,32,128]` | INT8 | `2^-8` | 0 |
| T31 | Per-token means, stored feature sums | `[32]` | INT32 | `2^-15` | 0 |
| T32 | Broadcast means | `[1,32,1]` | INT32; wide temporary used for subtraction | `2^-15` | 0 |
| T33 fine | Squared differences | `[1,32,128]` | INT64 temporary | `2^-30` | 0 |
| T33 | Converted squared differences | `[1,32,128]` | INT32 | `2^-15` | 0 |
| T34/T35 | Variance / variance plus epsilon | `[32]` | INT32 | `2^-22` | 0 |
| T36/T39 | Inverse standard deviation / broadcast | `[32]` / `[1,32,1]` | INT32 | `2^-16` | 0 |
| T37/T38/T41 | Mean correction / negative correction / broadcast | `[32]` / `[32]` / `[1,32,1]` | INT64 | `2^-31` | 0 |
| T40 | Input times inverse standard deviation | `[1,32,128]` | INT64 | `2^-24` | 0 |
| T42 fine | Normalized features | `[1,32,128]` | INT64 temporary | `2^-31` | 0 |
| T42 | Converted normalized features | `[1,32,128]` | INT32 | `2^-16` | 0 |
| T6/T7 | Gamma / beta | `[128]` each | INT32 constants | `2^-16` | 0 |
| T43/T44 | Affine product / output | `[1,32,128]` | INT64 | `2^-32` | 0 |
| T45 | Projection input | `[1,32,128]` | INT8 | `0.03362198546528816` | -17 |

T6/T7 are FP32 in the saved graph and converted offline. Original gamma range: `0.922644913` to `1.139880538`; beta range: `-0.242233545` to `0.219664216`. Maximum conversion errors were `7.629394531e-6` and `7.601454854e-6` respectively.

#### B. Mathematical and integer equations

Use the LayerNorm equation in Section 2 with the T6/T7 parameters. The graph's reduction axes are `(0,2)`; batch is one, so this is a per-token 128-feature reduction.

```text
# Before Op 0: ENCODE, nearest ties to even; clip to INT8.
q0 = clip(round_even(X * 256), -128, 127)

# Op 0–1: COMPUTE mean through its scale, then KEEP/reshape.
q31[t] = sum_k q0[0,t,k]          # scale 2^-15 = 2^-8 / 128
q32 = reshape(q31, [1,32,1])

# Op 2: ALIGN input to mean scale; COMPUTE the square.
d = int64(q0) * 128 - int64(q32) # scale 2^-15
q33_fine = d * d                 # scale 2^-30
q33 = RQ(q33_fine, 1, 15)        # REQUANTIZE HERE; store INT32 @ 2^-15

# Op 3: COMPUTE variance through output scale; no rounded division.
q34[t] = sum_k q33[0,t,k]        # scale 2^-22 = 2^-15 / 128

# Op 4–5: COMPUTE epsilon addition and inverse square root.
q35 = q34 + 42                   # same scale 2^-22
q36 = isqrt((1 << 54) // q35)    # floor; output scale 2^-16

# Op 6–11: follow the graph decomposition.
q37 = int64(q36) * int64(q31)    # scale 2^-31
q38 = -q37
q39 = reshape(q36, [1,32,1])
q40 = int64(q0) * int64(q39)     # scale 2^-24
q41 = reshape(q38, [1,32,1])
q42_fine = q40 * 128 + q41      # ALIGN q40 to 2^-31, then ADD
q42 = RQ(q42_fine, 1, 15)       # REQUANTIZE HERE; store INT32 @ 2^-16

# Op 12–13: COMPUTE affine product; ALIGN beta exactly before addition.
q43 = int64(q42) * int64(gamma_q) # scale 2^-32
q44 = q43 + int64(beta_q) * 65536 # scale 2^-32

# Op 14: REQUANTIZE HERE into the projection input.
q45 = clip(RQ(q44, 974600, 47) - 17, -128, 127)
```

The saved epsilon is approximately `9.999999747378752e-6`. At scale `2^-22`, integer epsilon 42 represents `1.0013580322265625e-5`. Inverse-square-root numerator exponent is $22+2(16)=54$.

The graph normalization and fused expression were checked equal:

$$
q_{42,\mathrm{fine}}=(128q_0-q_{32})q_{39}.
$$

#### C. Explicit conversion points and next consumers

| Location | Raw source | Action at this point | Stored result | Next consumer |
|---|---|---|---|---|
| Before Op 0 | FP32 T0 input | ENCODE at `2^-8`, clip | T0 INT8 | Op 0 and residual branch |
| During Op 2 | T0 at `2^-8` | ALIGN ×128 before subtraction | Centered difference at `2^-15` | Square in Op 2 |
| **After Op 2 square** | INT64 T33 fine at `2^-30` | **REQUANTIZE: round, shift 15** | T33 INT32 at `2^-15` | **Op 3** |
| Op 3 mean | Feature sum | COMPUTE output scale `2^-22`; no extra shift | T34 INT32 | Op 4 |
| Before Op 11 add | T40 at `2^-24` | ALIGN ×128 to `2^-31` | Aligned T40 | Op 11 add |
| **After Op 11 add** | INT64 T42 fine at `2^-31` | **REQUANTIZE: round, shift 15** | T42 INT32 at `2^-16` | **Op 12** |
| Before Op 13 add | Beta at `2^-16` | ALIGN ×65536 to `2^-32` | Aligned beta | Op 13 add |
| **Op 14** | INT64 T44 at `2^-32` | **REQUANTIZE: Mq=974600, shift 47, add -17, clip** | T45 INT8 | **Op 15–17** |

The Op 14 real conversion ratio is $2^{-32}/s_{45}$, approximated by $974600\,2^{-47}$. This is not T45's tensor scale.

#### D. Width checks and measurement coverage

Feature sums were checked before INT32 storage. Squares, fine normalization, and affine products use wide temporaries. Epsilon produces positive reciprocal-square-root inputs. `isqrt` floor identities were checked with Python integers. Affine addition and `q44 * 974600 + 2^46` were checked before evaluation. These checks cover the tested cases; they do not assert safety for arbitrary INT64 affine inputs.

#### E. Measurements and interpretation

Part 1 has the selected INT8-input candidate versus the FP32-input reference; its differences already include T0 encoding.

| Measurement | Benchmark | Calibration |
|---|---:|---:|
| T44 maximum absolute error | 0.0262759 | 0.0281122 |
| T44 mean absolute error | 0.0101911 | 0.0101453 |
| T45 exact matches | 2854/4096 (69.678%) | 20020/28672 (69.824%) |
| Maximum T45 code difference | 1 | 1 |
| Fixed conversion versus ideal conversion mismatches | 0 | 0 |
| Saturation count | 0 | 1 |

Earlier input/normalization differences dominate the affine parameter-conversion error. INT16 T0 has not been tested and does not replace this baseline.

### 4.2 Part 2 — Q/K/V projections, Op 15–17

#### A. Inputs, outputs, and weights

All inputs are T45 INT8 `[1,32,128]`, scale `0.03362198546528816`, zero point -17. All weight matrices are INT8 `[128,128]`, `[output,input]`, **per-tensor** scales, zero point 0. These three projections have no bias. All outputs have shape `[1,32,128]`.

| Projection / Op | Weight ID | Weight scale | Output ID | Output scale | Output zero point |
|---|---|---|---|---|---:|
| Q / 15 | T27 | `0.0017367515247315168` | T46 | `0.025061121210455894` | -12 |
| K / 16 | T25 | `0.0017351132119074464` | T47 | `0.018255505710840225` | -30 |
| V / 17 | T26 | `0.0006367104360833764` | T48 | `0.006938841659575701` | 4 |

#### B. Equations and ordered execution

For projection $p$:

$$
A_p[t,o]=\sum_{k=0}^{127}(q_{45}[t,k]+17)q_{W_p}[o,k],\qquad
s_{A_p}=s_{45}s_{W_p}.
$$

```text
centered_input = int64(q45) + 17
accumulator = centered_input @ int64(weight).T
# Check accumulator, then store INT32 at scale s45 * s_weight.
# REQUANTIZE HERE, immediately after each FC dot product:
output = clip(RQ(accumulator, Mq_p, 39) + z_p, -128, 127)
```

| Projection | Scale ratio `s45*s_weight/s_output` | Mq | Shift | Next consumer |
|---|---|---:|---:|---|
| Q | `0.0023300248233497876` | 1280944693 | 39 | Op 18 Q reshape |
| K | `0.003195635996911343` | 1756819468 | 39 | Op 20 K reshape |
| V | `0.003085164653966409` | 1696087205 | 39 | Op 21 V reshape |

The conversion occurs **before head layout operations**, not after QK multiplication. Each output keeps its own exported scale and zero point.

#### C. Width checks

INT64 dot-product checks precede INT32 accumulator storage. The multiplier-product bounds include the full signed INT32 accumulator range and rounding offset. Centered inputs are not stored as INT8.

#### D. Measurements

Local uses reference T45; chained uses Part 1's integer T45.

| Projection | Local matches, benchmark / calibration | Chained matches, benchmark / calibration | Maximum chained difference, benchmark / calibration |
|---|---|---|---|
| Q | 4096/4096 / 28672/28672 | 79.980% / 80.720% | 2 / 3 steps |
| K | 4096/4096 / 28672/28672 | 73.267% / 74.090% | 2 / 2 steps |
| V | 4096/4096 / 28672/28672 | 65.234% / 65.726% | 2 / 2 steps |

All 98,304 local Q/K/V values matched. Chained changes reflect the changed LayerNorm input. Q/K/V total dense MACs: 1,572,864 per sequence.

### 4.3 Part 3 — Attention, Op 18–39

#### A. Inputs, outputs, and intermediate formats

The Q/K/V input scales and zero points are those in Part 2.

| Tensor | Meaning | Shape | Candidate storage / scale | Zero point |
|---|---|---|---|---:|
| T49/T51/T52 | Q/K/V reshaped before head permutation | `[1,32,16,8]` | INT8; corresponding Q/K/V scale | Corresponding Q/K/V |
| T50/T53 | Q/V head layouts | `[1,16,32,8]` | INT8; Q/V scale | -12 / 4 |
| T54 | Transposed K head layout | `[1,16,8,32]` | INT8; K scale | -30 |
| T55 | Q matrices | `[16,32,8]` | INT8; Q scale | -12 |
| T57 | K-transpose matrices | `[16,8,32]` | INT8; K scale | -30 |
| T56/T58 positions | Original graph dequantized Q/K | Same head shapes | Candidate retains Q/K codes and applies zero points in dot product | As above |
| T59 | Raw QK scores | `[16,32,32]` | INT32; `0.0004575034413775367` | 0 |
| T60 | Reshaped scores | `[1,16,32,32]` | INT32; same score scale | 0 |
| T21 | Causal mask | `[1,1,32,32]` | BOOL | Not applicable |
| T61 position | Masked-score stage | `[1,16,32,32]` | Candidate carries allowed offsets INT32 at score scale plus Boolean mask | 0 for represented offsets |
| Exponentials | LUT exponential outputs | `[1,16,32,32]` | INT32; `2^-30` | 0 |
| Exponential sums | Row denominators | `[1,16,32,1]` | INT64; `2^-30` | 0 |
| T62 | Probabilities | `[1,16,32,32]` | INT32; `2^-16` | 0 |
| T63 | Probability matrices | `[16,32,32]` | INT32; `2^-16` | 0 |
| T64 | V matrices | `[16,32,8]` | INT8; `0.006938841659575701` | 4 |
| T65 position | Original graph dequantized V | `[16,32,8]` | Candidate subtracts V zero point in integer arithmetic | — |
| T66 | Context | `[16,32,8]` | INT32; `1.0587832122155305e-07` | 0 |
| T67 | Quantized context | `[16,32,8]` | INT8; `0.006692775525152683` | 9 |
| T68 | Batch/head context | `[1,16,32,8]` | INT8; same as T67 | 9 |
| T69 | Token/head context | `[1,32,16,8]` | INT8; same as T67 | 9 |
| T70 | Merged context | `[1,32,128]` | INT8; same as T67 | 9 |

The candidate's max-subtracted offsets are not literal replacements for the original raw masked FP32 T61 values. Masked entries have a placeholder zero and remain ignored by the separate mask. The candidate does not represent negative infinity as an integer.

#### B. Op 18–28: layout and scores

RESHAPE and TRANSPOSE keep each projection's integer values, scale, and zero point. Original DEQUANTIZE operations are bypassed by using metadata in integer dot products.

$$
q_{59}[h,t,j]=\sum_{d=0}^{7}(q_Q[h,t,d]+12)(q_K[h,d,j]+30).
$$

$$
s_{59}=s_Qs_K=0.0004575034413775367.
$$

```text
# KEEP: execute saved reshapes/permutations on projection codes.
score_wide = (int64(Q_heads) + 12) @ (int64(K_transposed_heads) + 30)
q59 = checked_int32(score_wide)
```

**No score requantization occurs after Op 28.** T59 stores the dot products directly at the product scale. The next consumer is Op 29 reshape, then masking/softmax. No extra attention scaling is applied.

#### C. Op 29–32: mask, exponential lookup, and probability division

T21 is lower-triangular inclusive: 528 allowed and 496 masked entries per head. Select the maximum only among allowed entries, then subtract it using wide arithmetic:

```text
q60 = reshape(q59, [1,16,32,32])
row_max = max(q60 over allowed key positions)
shifted_q = checked_int32(int64(q60) - int64(row_max))
shifted_q = 0 for masked entries  # placeholder; mask controls evaluation
```

The shifted score scale remains $s_{59}$. The table is generated offline as:

$$
L[i]=\operatorname{round}_{\mathrm{even}}(e^{-i/256}2^{30}),\quad 0\le i\le4096.
$$

| Exponential-table item | Selected value |
|---|---|
| Function / interval | `exp(-u)`, `u` in `[0,16]` |
| Spacing / entries | `1/256` / 4097 |
| Storage / bytes | INT32 / 16,388 |
| Output scale | `2^-30` |
| Coordinate fractional bits | 16 |
| Coordinate multiplier / shift | 503030354 / 24 |
| Interpolation fractional bits | 8 |
| Cutoff coordinate | `16 << 16` |

Ordered integer evaluation:

```text
# REQUANTIZE COORDINATE HERE, before table indexing.
coord = RQ(-int64(shifted_q), 503030354, 24)
active = allowed AND (coord < (16 << 16))
safe_coord = clip(coord, 0, (16 << 16)-1)
index = safe_coord >> 8
fraction = safe_coord & 255
lower = int64(L[index])
upper = int64(L[index+1])
exp_q = lower + (((upper-lower)*fraction + 128) >> 8)
exp_q = 0 for inactive entries

# COMPUTE normalized probabilities; denominator is INT64.
sum_exp_q = sum(exp_q over keys)
q62 = (int64(exp_q)*65536 + sum_exp_q//2) // sum_exp_q
q63 = reshape(q62, [16,32,32])    # KEEP, no requantization
```

Coordinates represent $u=-s_{59}\,\mathrm{shifted\_q}$ at scale `2^-16`. The coordinate conversion is for table addressing, not the score or exponential tensor scale. `exp_q` remains at `2^-30`.

Probability normalization directly computes INT32 T62 at `2^-16`; its integer one is 65536. Independent rounding means row sums need not be exactly 65536. Maximum observed deviation was 5 steps (`7.62939453125e-5`). Masked probabilities are exactly zero. No allowed exponential reached the cutoff in tested cases.

#### D. Op 33–39: probability × V and head merging

$$
q_{66}[h,t,d]=\sum_{j=0}^{31}q_{63}[h,t,j](q_{64}[h,j,d]-4),\qquad
s_{66}=2^{-16}s_V=1.0587832122155305\times10^{-7}.
$$

```text
q64 = reshape(V_head_codes, [16,32,8])   # Op 33: KEEP
V_centered = int64(q64) - 4              # Op 34 counterpart
context_wide = int64(q63) @ V_centered   # Op 35: COMPUTE
q66 = checked_int32(context_wide)

# Op 36: REQUANTIZE HERE before head merging.
q67 = clip(RQ(q66, 1113218944, 46) + 9, -128, 127)

q68 = reshape(q67, [1,16,32,8])
q69 = transpose(q68, [0,2,1,3])
q70 = reshape(q69, [1,32,128])           # Op 37–39: KEEP
```

The Op 36 ratio is $s_{66}/s_{67}=1.581979267400241\times10^{-5}$. Multiplier relative error is `3.408504590e-10`. Head merging has no numeric conversion: T67–T70 keep scale `0.006692775525152683` and zero point 9. Features `8*h` through `8*h+7` in T70 belong to head `h`.

#### E. Conversion points and next consumers

| Location | Action | Output | Next consumer |
|---|---|---|---|
| Op 18–27 | KEEP layouts; bypass FP32 dequantization | Integer Q/K/V head codes | Op 28 / 33 |
| Op 28 | COMPUTE scores at product scale; **no requantization** | T59 INT32 | Op 29–31 |
| Before exponential lookup | REQUANTIZE offset magnitude to coordinate; Mq=503030354, shift 24 | Coordinate integer at `2^-16` | LUT indexing/interpolation |
| Op 31 probability normalization | COMPUTE rounded ratio into `2^-16` | T62 INT32 | Op 32 / 35 |
| Op 35 | COMPUTE context at probability × V scale | T66 INT32 | Op 36 |
| **Op 36** | **REQUANTIZE Mq=1113218944, shift 46, +9, clip** | T67 INT8 | **Op 37–39** |
| Op 37–39 | KEEP shape/permutation only | T70 INT8 | Op 40 projection |

#### F. Width checks

| Quantity | Bound / check | Coverage |
|---|---|---|
| QK score | `8 * 139 * 157 = 174584`, fits INT32 | Full INT8 Q/K domains |
| Score-coordinate product | Up to `2^31 * 503030354` plus offset, fits INT64 | Any nonpositive INT32 offset |
| Exponential denominator | At most `32 * 2^30` | Full row length |
| Probability numerator | At most `2^46` before denominator rounding offset | Exponential range |
| Context dot product | `32 * 65536 * 132 = 276824064`, fits INT32 | Full probability/code bounds |
| Context conversion product | `2^31 * 1113218944 + 2^45` fits INT64 | Full INT32 accumulator domain |

Local layouts, masked zeros, table monotonicity, probability ranges, first-query context identity, and head-to-feature mapping checks passed. Query token zero has only one allowed key; its context is exactly `65536 * centered_V_token0` in integer representation.

#### G. Measurements

Local attention begins from reference-matching local Q/K/V projections and carries its integer softmax approximation. Chained attention begins from the integer LayerNorm/QKV path.

| Measurement | Benchmark | Calibration |
|---|---:|---:|
| Local score maximum absolute error | 5.6697e-7 | 9.9516e-7 |
| Chained score maximum absolute error | 0.195812 | 0.231954 |
| Local probability maximum absolute error | 8.5831e-6 | 9.2387e-6 |
| Chained probability maximum absolute error | 0.025901 | 0.034797 |
| Local context maximum / mean absolute error | 2.2438e-5 / 2.4410e-6 | 3.2888e-5 / 2.4953e-6 |
| Chained context maximum / mean absolute error | 0.010257 / 0.001346 | 0.015926 / 0.001308 |
| Local T70 exact matches | 4093/4096 (99.927%) | 28661/28672 (99.962%) |
| Local maximum T70 code difference | 1 | 1 |
| Chained T70 exact matches | 3282/4096 (80.127%) | 23065/28672 (80.444%) |
| Chained maximum T70 code difference | 2 | 3 |
| Op 36 fixed versus ideal conversion mismatches | 0 | 0 |
| T67 saturation, local / chained | 0 / 0 | 2 / 1 |

Across all cases, 14 of 32,768 local T70 values differed, each by one step. The fixed context conversion and head merging introduced no observed additional discrepancies relative to their diagnostic controls.

### 4.4 Part 4 — Attention output projection and residual, Op 40–42

#### A. Inputs, outputs, and parameters

| Tensor | Meaning | Shape | Candidate storage | Scale | Zero point |
|---|---|---|---|---|---:|
| T70 | Merged attention input | `[1,32,128]` | INT8 | `0.006692775525152683` | 9 |
| T28 | Output-projection weights | `[128,128]` | INT8; per-tensor | `0.0006937290891073644` | 0 |
| T24 | Bias | `[128]` | INT32 | `4.6429730686642335e-06` | 0 |
| Accumulator | Dot products plus bias | `[32,128]` | INT32 | `4.6429730686642335e-06` | 0 |
| T71 | Projection output | `[1,32,128]` | INT8 | `0.003416289109736681` | -13 |
| T72 | Projection aligned for residual | `[1,32,128]` | INT32 | `2^-16` | 0 |
| T0 | Existing input residual branch | `[1,32,128]` | INT8 | `2^-8` | 0 |
| T73 | First residual output | `[1,32,128]` | INT32 | `2^-16` | 0 |

Original T24 bias integer range: `-13036` to `8460`. Bias scale equals `s70*s28`.

#### B. Equations and ordered execution

$$
A[t,o]=\sum_{k=0}^{127}(q_{70}[t,k]-9)q_{28}[o,k]+q_{24}[o].
$$

```text
# Op 40: COMPUTE FC accumulator, then REQUANTIZE before residual conversion.
acc = (int64(q70)-9) @ int64(W28).T + int64(b24)
q71 = clip(RQ(checked_int32(acc), 747156156, 39)-13, -128, 127)

# Op 41 counterpart: REQUANTIZE centered projection into residual scale.
q72 = checked_int32(RQ(int64(q71)+13, 939062400, 22))

# Op 42: ALIGN the existing T0 branch exactly, then COMPUTE addition.
aligned_q0 = int64(q0) * 256
q73 = checked_int32(int64(q72) + aligned_q0)
```

| Conversion | Source scale → target scale | Ratio | Mq / shift | Next consumer |
|---|---|---|---|---|
| Op 40 accumulator → T71 | `s70*s28` → `s71` | `0.0013590691301363842` | 747156156 / 39 | Op 41 |
| **Op 41 T71 → T72** | `s71` → `2^-16` | `s71 / 2^-16` | 939062400 / 22 | **Op 42 add** |
| T0 residual alignment | `2^-8` → `2^-16` | Exact ×256 | No rounding | Op 42 add |

Zero-point removal for the residual conversion is `q71 + 13`. T73 is not clipped to INT8. Its stored scale remains `2^-16` and its next consumer is Op 43 second LayerNorm; T73 also remains live for the final residual at Op 73.

#### C. Width checks

Bias compatibility and observed dot products were checked before INT32 storage. Projection multiplier bounds cover all signed INT32 accumulators. T71 conversion is bounded over the full INT8 code domain. Residual sums were checked on tested values before INT32 storage.

#### D. Measurements

Local projection uses reference T70; chained projection uses chained integer T70. **Both Part 4 residual paths use the existing INT8 T0 candidate.** Local residual error therefore includes T0 input quantization.

| Measurement | Benchmark | Calibration |
|---|---:|---:|
| Local T71 matches | 4096/4096 (100%) | 28672/28672 (100%) |
| Chained T71 matches | 3428/4096 (83.691%) | 24479/28672 (85.376%) |
| Chained maximum T71 difference | 2 | 2 |
| Shared T0 maximum encoding error | 1.9524e-3 | 1.9530e-3 |
| Local maximum T73 error | 0.001959 | 0.001959 |
| Chained maximum / mean T73 error | 0.005364 / 0.001368 | 0.007440 / 0.001333 |
| Maximum residual conversion/addition own error | 7.5707e-6 | 7.5707e-6 |
| Fixed projection conversion mismatches versus ideal | 0 | 0 |
| Projection saturation, local / chained | 0 / 0 | 2 / 2 |

The residual operation adds comparatively little error. Its finer output format preserves the sum but cannot recover detail already lost when T0 was encoded to INT8.

### 4.5 Part 5 — Second LayerNorm, Op 43–57

#### A. Inputs, outputs, and parameters

| Tensor | Meaning | Shape | Candidate storage | Scale | Zero point |
|---|---|---|---|---|---:|
| T73 | Second LayerNorm input | `[1,32,128]` | INT32 | `2^-16` | 0 |
| T74/T75 | Means / broadcast means | `[32]` / `[1,32,1]` | INT32 | `2^-23` | 0 |
| T76 fine | Squared differences | `[1,32,128]` | INT64 temporary | `2^-46` | 0 |
| T76 | Converted squared differences | `[1,32,128]` | INT32 | `2^-23` | 0 |
| T77/T78 | Variance / variance plus epsilon | `[32]` | INT32 | `2^-30` | 0 |
| T79/T82 | Inverse standard deviation / broadcast | `[32]` / `[1,32,1]` | INT32 | `2^-16` | 0 |
| T80/T81/T84 | Correction / negative / broadcast | `[32]` / `[32]` / `[1,32,1]` | INT64 | `2^-39` | 0 |
| T83 | Input times inverse standard deviation | `[1,32,128]` | INT64 | `2^-32` | 0 |
| T85 fine | Normalized features | `[1,32,128]` | INT64 temporary | `2^-39` | 0 |
| T85 | Converted normalized features | `[1,32,128]` | INT32 | `2^-16` | 0 |
| T8/T9 | Gamma / beta | `[128]` each | INT32 constants | `2^-16` | 0 |
| T86/T87 | Affine product / output | `[1,32,128]` | INT64 | `2^-32` | 0 |
| T88 | MLP input | `[1,32,128]` | INT8 | `0.03247051313519478` | 2 |

T8/T9 are FP32 in the saved graph and converted offline. Gamma range: `1.017637` to `1.1421613`; beta range: `-0.035599805` to `0.05044351`. Maximum parameter conversion errors: `7.510185242e-6` and `7.528811693e-6`.

#### B. Ordered integer equations

```text
# Op 43–44: COMPUTE mean using scale 2^-16 / 128 = 2^-23; KEEP reshape.
q74[t] = checked_int32(sum_k int64(q73[0,t,k]))
q75 = reshape(q74, [1,32,1])

# Op 45: ALIGN input to mean scale, COMPUTE square, then REQUANTIZE.
d = int64(q73)*128 - int64(q75) # scale 2^-23
q76_fine = d*d                   # scale 2^-46
q76 = checked_int32(RQ(q76_fine, 1, 23)) # store at 2^-23 before Op 46

# Op 46: COMPUTE variance through scale; no extra rounded division.
q77[t] = checked_int32(sum_k int64(q76[0,t,k])) # scale 2^-30

# Op 47–48: COMPUTE epsilon and RSQRT at independently chosen scales.
q78 = q77 + 10737                # scale 2^-30
q79 = isqrt((1 << 62) // q78)    # floor; scale 2^-16

# Op 49–54: follow graph normalization and align products before ADD.
q80 = int64(q79)*int64(q74)      # scale 2^-39
q81 = -q80
q82 = reshape(q79, [1,32,1])
q83 = int64(q73)*int64(q82)      # scale 2^-32
q84 = reshape(q81, [1,32,1])
q85_fine = q83*128 + q84         # scale 2^-39
q85 = checked_int32(RQ(q85_fine, 1, 23)) # REQUANTIZE before Op 55

# Op 55–56: COMPUTE affine product, ALIGN beta exactly, then ADD.
q86 = int64(q85)*int64(gamma2_q) # scale 2^-32
q87 = q86 + int64(beta2_q)*65536 # scale 2^-32

# Op 57: REQUANTIZE HERE into MLP input.
q88 = clip(RQ(q87, 1009162, 47)+2, -128, 127)
```

The independently encoded epsilon is:

| Item | Value |
|---|---|
| Original epsilon | `9.999999747378752e-6` |
| Integer at `2^-30` | 10737 |
| Represented epsilon | `9.999610483646393e-6` |
| Absolute epsilon conversion error | `3.892637324e-10` |

The RSQRT numerator exponent is $30+2(16)=62$, fitting signed INT64. T79 floor error is less than one `2^-16` step. Graph decomposition exactly matched:

$$
q_{85,\mathrm{fine}}=(128q_{73}-q_{75})q_{82}.
$$

#### C. Explicit conversion points

| Location | Raw source | Action | Stored result | Next consumer |
|---|---|---|---|---|
| Before Op 45 subtraction | T73 at `2^-16` | ALIGN ×128 to mean scale | Difference at `2^-23` | Square |
| **After Op 45 square** | INT64 T76 fine at `2^-46` | **REQUANTIZE round, shift 23** | T76 INT32 at `2^-23` | **Op 46** |
| Op 46 | Sum of q76 | COMPUTE scale `2^-30`; no integer division | T77 INT32 | Op 47 |
| Before Op 54 ADD | T83 at `2^-32` | ALIGN ×128 to `2^-39` | Aligned product | Op 54 |
| **After Op 54 ADD** | INT64 T85 fine at `2^-39` | **REQUANTIZE round, shift 23** | T85 INT32 at `2^-16` | **Op 55** |
| Before Op 56 ADD | Beta at `2^-16` | ALIGN ×65536 | Beta at `2^-32` | Op 56 |
| **Op 57** | INT64 T87 at `2^-32` | **REQUANTIZE Mq=1009162, shift 47, +2, clip** | T88 INT8 | **Op 58** |

The final ratio is $2^{-32}/s_{88}$. Multiplier relative error: `2.434246653e-7`.

#### D. Width checks

Observed mean sums were checked before INT32 storage. Each centered difference was bounded before squaring; fine squares and sums fit INT64 on tested inputs. Converted squared sums fit INT32. Epsilon inputs were positive, the numerator fit INT64, and exact floor-square-root identities passed. Normalization product/alignment bounds and affine conversion products were checked before arithmetic. The selected T73 input range, not every possible INT32 code, grounds the observed LayerNorm bounds.

#### E. Measurements

Local starts from **reference T73 encoded directly at `2^-16`**; chained starts from Part 4's chained integer T73. This differs from Part 4's local residual test, which retained the INT8 T0 branch.

| Measurement | Benchmark | Calibration |
|---|---:|---:|
| Local maximum mean error | 8.5874e-7 | 1.0780e-6 |
| Chained maximum mean error | 0.000351 | 0.000412 |
| Local maximum variance error | 2.0023e-7 | 2.6822e-7 |
| Chained maximum variance error | 2.4885e-4 | 2.4885e-4 |
| Local maximum inverse-std error | 6.1035e-5 | 1.1635e-4 |
| Chained maximum inverse-std error | 0.042701 | 0.065284 |
| Local maximum T85 error | 8.1897e-5 | 9.1314e-5 |
| Chained maximum T85 error | 0.052997 | 0.053643 |
| Local maximum / mean T87 error | 9.8902e-5 / 3.4706e-5 | 1.0516e-4 / 3.5469e-5 |
| Chained maximum / mean T87 error | 0.057087 / 0.012255 | 0.057442 / 0.012244 |
| Local T88 matches | 4094/4096 (99.951%) | 28644/28672 (99.902%) |
| Chained T88 matches | 2587/4096 (63.159%) | 18257/28672 (63.675%) |
| Maximum T88 difference, local / chained | 1 / 2 | 1 / 2 |
| Fixed output conversion mismatches versus ideal | 0 | 0 |
| Saturation, local / chained | 0 / 0 | 2 / 2 |

Thirty of 32,768 local T88 values differed by one step. Normalization can magnify earlier residual differences because inverse standard deviation is approximately 6.5–9.8 on these cases. The conversion to T85 adds at most half a `2^-16` step.

### 4.6 Part 6 — MLP expansion and GELU, Op 58–70

#### A. Inputs, outputs, parameters, and formats

| Tensor | Meaning | Shape | Candidate storage | Scale | Zero point |
|---|---|---|---|---|---:|
| T88 | Expansion input | `[1,32,128]` | INT8 | `0.03247051313519478` | 2 |
| T29 | Expansion weights | `[512,128]` | INT8; per-tensor | `0.000803247734438628` | 0 |
| T23 | Bias | `[512]` | INT32 | `2.6081866111904916e-05` | 0 |
| FC accumulator | Expansion dot products plus bias | `[32,512]` | INT32 | `2.6081866111904916e-05` | 0 |
| T89 | Expansion output | `[1,32,512]` | INT8 | `0.011252585798501968` | 37 |
| T90 position | Original dequantized input | `[1,32,512]` | Candidate retains centered T89 integers | `s89` | 0 after centering |
| T91 | Half-factor constant | Scalar | INT8 code 127 | `s91 ≈ 0.00196078` | -128 |
| T92 | Half-input branch | `[1,32,512]` | INT8 | `s92 ≈ 0.00562629` | 37 |
| T93 | Integer cube | `[1,32,512]` | INT32 | `s89^3 = 1.4248101460504496e-6` | 0 |
| T94 | Quantized cube | `[1,32,512]` | INT8 | `s94 ≈ 0.03000131` | 85 |
| T95 | Cubic-coefficient constant | Scalar | INT8 code 55 | `s95 ≈ 0.00024414` | -128 |
| T96 | Weighted cubic term | `[1,32,512]` | INT8 | `s96 ≈ 0.00134151` | 85 |
| T97 | Linear plus cubic term | `[1,32,512]` | INT8 | `0.012609509751200676` | 37 |
| T98 | Argument coefficient | Scalar | INT8 code 127 | `s98 ≈ 0.00312896` | -128 |
| T99 | Tanh argument | `[1,32,512]` | INT8 | `0.010060933418571949` | 37 |
| T100 position | Original dequantized argument | `[1,32,512]` | Candidate uses T99 codes for lookup | `s99` and z99 | 37 |
| T101 | Tanh output | `[1,32,512]` | INT32 | `2^-24` | 0 |
| T102 | One plus tanh | `[1,32,512]` | INT32 | `2^-24` | 0 |
| T103 | Quantized factor | `[1,32,512]` | INT8 | `0.0066078659147024155` | -128 |
| T104 | Final GELU output | `[1,32,512]` | INT8 | `0.004273270256817341` | -88 |

T3 is the exponent 3; T4 is the offset one. They are FP32 scalar constants in the original graph, implemented as cubing and exact integer-one addition in the candidate. Represented quantized constants: T91 approximately `0.50000003`, T95 approximately `0.04467773`, and T98 approximately `0.79788457`. Centered constant integers are respectively **255, 183, 255**. T23 bias range: `-3909` to `1823`.

#### B. Op 58: expansion FC

$$
A[t,o]=\sum_{k=0}^{127}(q_{88}[t,k]-2)q_{29}[o,k]+q_{23}[o],\qquad
s_A=s_{88}s_{29}.
$$

```text
acc = (int64(q88)-2) @ int64(W29).T + int64(b23)
# REQUANTIZE HERE before either GELU branch consumes x.
q89 = clip(RQ(checked_int32(acc), 1274254450, 39)+37, -128, 127)
```

The ratio is `0.0023178553426695166`. Multiplier relative error: `3.012972094e-10`. The accumulator shape is `[32,512]`.

#### C. Op 59–65: ordered half-input and cubic branches

```text
xq = int64(q89)-37             # Op 59 counterpart, scale s89

# Op 60: COMPUTE product with saved centered half-factor, then REQUANTIZE.
p60 = xq * 255                # product scale s89*s91
q92 = clip(RQ(p60, 1077952640, 38)+37, -128, 127)

# Op 61: COMPUTE exact integer cube at product scale.
q93 = checked_int32(xq*xq*xq)  # scale s89^3
# Op 62: REQUANTIZE HERE before weighted-cube multiplication.
q94 = clip(RQ(q93, 1670962322, 45)+85, -128, 127)

# Op 63: COMPUTE product with saved centered cubic coefficient, then REQUANTIZE.
p63 = (int64(q94)-85)*183      # product scale s94*s95
q96 = clip(RQ(p63, 1500813243, 38)+85, -128, 127)

# Op 64: convert BOTH branches into a common output-unit fixed-point precision.
sum_fine = (int64(q89)-37)*1916390445 + (int64(q96)-85)*228467824
# REQUANTIZE combined sum ONCE before Op 65.
q97 = clip(((sum_fine + (1 << 30)) >> 31)+37, -128, 127)

# Op 65: COMPUTE product with saved centered argument coefficient, then REQUANTIZE.
p65 = (int64(q97)-37)*255      # product scale s97*s98
q99 = clip(RQ(p65, 1077952560, 38)+37, -128, 127)
```

Op 64 conversion ratios are $s_{89}/s_{97}$ and $s_{96}/s_{97}$. Both use multiplier encoding `2^-31`; the weighted sum is rounded once. Adding raw q89 and q96 codes directly would be incorrect because their scales and zero points differ.

#### D. Op 66–69: ordered integer tanh and factor conversion

For each code $c\in[-128,127]$, generate offline:

$$
L_{\tanh}[c+128]=\operatorname{round}_{\mathrm{even}}
\left(\tanh((c-37)s_{99})2^{24}\right).
$$

| Tanh-table property | Value |
|---|---|
| Entries / dtype / bytes | 256 / INT32 / 1024 |
| Full code-domain represented arguments | `-1.6600540140643716` to `0.9054840076714754` |
| Tanh output scale | `2^-24` |
| Maximum full-table approximation error | `2.979223762e-8` |
| Integer one | 16777216 |

```text
# Op 66–67 counterpart: direct lookup for each INT8 input code.
q101 = tanh_lut[int16(q99)+128] # scale 2^-24; no interpolation

# Op 68: exact ADD at the same scale.
q102 = checked_int32(int64(q101)+16777216)

# Op 69: REQUANTIZE HERE before final GELU multiplication.
q103 = clip(RQ(q102, 1269488229, 47)-128, -128, 127)
```

Tanh lookup computes a nonlinear output in its declared format; the separate Op 69 conversion changes that representation to the saved INT8 factor format. Table generation and diagnostic `np.tanh` use host floating point; candidate lookup and addition are integer operations.

#### E. Op 70: final product and confirmed two-stage rounding

$$
P=(q_{92}-37)(q_{103}+128),\qquad
s_P=s_{92}s_{103}=3.717778907509281\times10^{-5}.
$$

The output ratio is approximated by `1195729761 * 2^-37`. The selected computation **does not round once at bit 37**. It performs:

```text
product = (int64(q92)-37)*(int64(q103)+128)

# REQUANTIZE HERE with two explicit rounding stages.
# Stage 1: rounded Q31 high multiplication, ties toward +infinity here.
high = (product*1195729761 + (1 << 30)) >> 31

# Stage 2: nearest division by 64; halfway values away from zero.
remainder = high & 63
threshold = 31 + int(high < 0)
scaled = (high >> 6) + int(remainder > threshold)

# Only after BOTH rounding stages: add zero point and saturate.
q104 = clip(scaled-88, -128, 127)
```

This high-multiply expression is valid for the positive multiplier and bounded products used here; the exceptional signed-INT32-minimum × signed-INT32-minimum saturation case is unreachable. Output T104 retains scale `0.004273270256817341`, zero point -88. The intermediate high integer is not a T104 tensor with scale `2^-31`; that exponent describes the multiplier operation.

The single-rounding comparison is retained in `gelu_output_integer_trial`. It differed from the local reference at 7 benchmark and 27 calibration elements. Using the two-stage version removed all 34 differences. The selected result is **`gelu_output_two_round_trial`**.

#### F. Conversion points and width checks

| Op / location | Source scale → destination scale | Mq / shift | Next consumer |
|---|---|---|---|
| **58 FC output** | `s88*s29` → `s89` | 1274254450 / 39 | **Op 59–61** |
| **60 half-product output** | `s89*s91` → `s92` | 1077952640 / 38 | **Op 70 half branch** |
| 61 cube | `s89` → `s89^3` by COMPUTE | No separate conversion | Op 62 |
| **62 cube quantization** | `s89^3` → `s94` | 1670962322 / 45 | **Op 63** |
| **63 coefficient product** | `s94*s95` → `s96` | 1500813243 / 38 | **Op 64** |
| **64 branch sum** | `s89` and `s96` → `s97` | 1916390445 and 228467824 / common 31 | **Op 65** |
| **65 argument product** | `s97*s98` → `s99` | 1077952560 / 38 | **Op 66–67 lookup** |
| 68 +one | `2^-24` unchanged | Exact +16777216 | Op 69 |
| **69 factor quantization** | `2^-24` → `s103` | 1269488229 / 47 | **Op 70** |
| **70 final product** | `s92*s103` → `s104` | 1195729761 / **31, then 6** | **Op 71 projection** |

Cube full-domain bound is `165^3 = 4492125`, fitting INT32. Final product bound is `165*255 = 42075`, fitting INT32. FC accumulators were checked before storage. Conversion-product bounds cover signed INT32 sources for the helper conversions. Op 64's combined product bound covers both full INT8 domains. Tanh table monotonicity, zero lookup, and output bounds passed.

#### G. Measurements

Local Part 6 starts from reference T88, then carries the integer expansion and GELU through the whole part. Chained starts from Part 5's chained T88. Every tested local quantized output T89, T92, T94, T96, T97, T99, T103, and selected T104 matched the reference.

| Measurement | Benchmark | Calibration |
|---|---:|---:|
| Local T89 matches | 16384/16384 (100%) | 114688/114688 (100%) |
| Chained T89 match / maximum difference | 67.267% / 2 | 66.073% / 3 |
| Local cube maximum real error vs FP32 graph | 6.1642e-7 | 6.7604e-7 |
| Chained T94/T96 match / maximum difference | 97.607% / 3 | 97.675% / 8 |
| Chained T97/T99 match / maximum difference | 70.483% / 2 | 69.440% / 2 |
| Local tanh maximum real error vs graph | 5.9605e-8 | 5.9605e-8 |
| Chained T103 match / maximum difference | 70.557% / 3 | 69.503% / 4 |
| Local selected T104 matches | 16384/16384 (100%) | 114688/114688 (100%) |
| Chained selected T104 matches | 11519/16384 (70.306%) | 79110/114688 (68.978%) |
| Chained T104 maximum / mean code difference | 4 / 0.406433 | 6 / 0.423322 |
| Chained T104 maximum / mean represented error | 0.017093 / 0.001737 | 0.025640 / 0.001809 |
| Op 70 two-stage reference-input control mismatches | 0 | 0 |

| Saturation location | Benchmark local / chained | Calibration local / chained |
|---|---|---|
| T89 | 1 / 1 | 4 / 4 |
| T92 / T94 / T96 | 0 / 0 for each | 0 / 0 for each |
| T97 | 0 / 0 | 5 / 5 |
| T99 | 0 / 0 | 0 / 0 |
| T103 | 2 / 2 | 6 / 5 |
| T104 | 0 / 0 | 0 / 0 |

Reference-input controls for Op 63, 64, 65, tanh/factor conversion, and selected Op 70 had zero mismatches. Standard fixed conversions matched ideal single rounding on tested values; selected Op 70 intentionally uses the confirmed two-stage rounding instead. Higher cubic-branch match rates also reflect its coarser encoding; they are not model accuracy percentages.

### 4.7 Part 7 — MLP output projection and final residual, Op 71–73

#### A. Inputs, outputs, and parameters

| Tensor | Meaning | Shape | Candidate storage | Scale | Zero point |
|---|---|---|---|---|---:|
| T104 | Selected two-stage GELU output | `[1,32,512]` | INT8 | `0.004273270256817341` | -88 |
| T30 | Final projection weights | `[128,512]` | INT8; per-tensor | `0.0007148781442083418` | 0 |
| T22 | Bias | `[128]` | INT32 | `3.054867510894285e-06` | 0 |
| Accumulator | Dot products plus bias | `[32,128]` | INT32 | `3.054867510894285e-06` | 0 |
| T105 | Projection output | `[1,32,128]` | INT8 | `0.0026197750121355057` | 10 |
| T106 | Projection aligned for addition | `[1,32,128]` | INT32 | `2^-16` | 0 |
| T73 | First residual branch | `[1,32,128]` | INT32 | `2^-16` | 0 |
| T107 | Final block output | `[1,32,128]` | INT32 | `2^-16` | 0 |

T22 bias range is `-16701` to `16823`. Bias scale equals `s104*s30`. This projection has 512 input features; a helper hard-coded to reshape inputs into 128 columns cannot be used unchanged.

#### B. Equations and ordered execution

$$
A[t,o]=\sum_{k=0}^{511}(q_{104}[t,k]+88)q_{30}[o,k]+q_{22}[o].
$$

```text
# Op 71: COMPUTE projection at accumulator scale.
acc = (int64(q104)+88) @ int64(W30).T + int64(b22)
# REQUANTIZE HERE before conversion to residual format.
q105 = clip(RQ(checked_int32(acc), 641059315, 39)+10, -128, 127)

# Op 72 counterpart: REQUANTIZE HERE before final ADD.
q106 = checked_int32(RQ(int64(q105)-10, 720118272, 22))

# Op 73: COMPUTE ADD at an already shared scale; no further requantization.
q107 = checked_int32(int64(q106)+int64(q73))
```

| Location | Source → destination scale | Ratio | Mq / shift | Next consumer |
|---|---|---|---|---|
| **Op 71 FC output** | `s104*s30` → `s105` | `0.0011660801010557446` | 641059315 / 39 | **Op 72** |
| **Op 72 centered projection** | `s105` → `2^-16` | `s105 / 2^-16` | 720118272 / 22 | **Op 73** |
| Op 73 ADD | Both inputs `2^-16` | No conversion | Direct integer addition | Block output |

T107 remains INT32 at `2^-16`, zero point 0. Its represented value is `q107 * 2^-16`. No block-output INT8 format has been inserted here; the interface to the next block still needs to be designed and tested.

#### C. Width checks

The conservative full INT8 input/weight-domain accumulator bound, including saved biases, is 14,107,063 and fits INT32. The projection multiplier bound covers every signed INT32 accumulator. T105 centering/conversion is bounded over its full INT8 domain. Final residual sums were checked on tested inputs before INT32 storage. Reference T107 was checked against reference T73 + T106.

#### D. Measurements

Local uses reference T104 and reference T73 encoded directly at `2^-16`. Chained uses selected integer GELU T104 and the existing chained T73. **Local here isolates Part 7; it is not a complete block path starting from T0.** Chained T107 is the complete integer block candidate.

| Projection measurement | Benchmark | Calibration |
|---|---:|---:|
| Local T105 matches | 4096/4096 (100%) | 28672/28672 (100%) |
| Chained T105 matches | 2022/4096 (49.365%) | 13952/28672 (48.661%) |
| Chained maximum / mean code difference | 3 / 0.557617 | 4 / 0.568394 |
| Fixed versus ideal conversion mismatches | 0 | 0 |
| Projection saturation, local / chained | 2 / 2 | 2 / 3 |

| Residual measurement | Benchmark | Calibration |
|---|---:|---:|
| Local T73 encoding maximum error | 7.6294e-6 | 7.6294e-6 |
| Chained T73 maximum / mean error | 0.005364 / 0.001368 | 0.007440 / 0.001333 |
| Chained T106 maximum / mean error | 0.007864 / 0.001463 | 0.010486 / 0.001491 |
| Maximum final conversion/addition own error, either path | 7.4823e-6 | 7.5176e-6 |

Complete T107 results are in Section 6. Projection and residual errors combine and can partially cancel, so final errors are measured directly rather than estimated by summing intermediate maxima.

## 5. Consolidated conversion and format reference

### 5.1 All selected requantization points

`RQ` is the single-stage rule in Section 3. A `+z / clip` entry is applied after the listed rounding. Gamma/beta offline encoding, exact ALIGN steps, mean-through-scale, and nonlinear lookup/division are listed separately below.

| Op / location | Source → stored result | Mq | Right shift(s) | Output zero point / clamp | Next consumer |
|---|---|---:|---|---|---|
| After 2 | T33 fine `2^-30` → T33 `2^-15` | 1 | 15 | 0 / checked INT32 | 3 |
| After 11 | T42 fine `2^-31` → T42 `2^-16` | 1 | 15 | 0 / checked INT32 | 12 |
| 14 | T44 `2^-32` → T45 | 974600 | 47 | -17 / INT8 clip | 15–17 |
| 15 | Q accumulator → T46 | 1280944693 | 39 | -12 / INT8 clip | 18 |
| 16 | K accumulator → T47 | 1756819468 | 39 | -30 / INT8 clip | 20 |
| 17 | V accumulator → T48 | 1696087205 | 39 | 4 / INT8 clip | 21 |
| Softmax coordinate | Score magnitude → lookup coordinate at `2^-16` | 503030354 | 24 | 0 / bounded indexing | Exponential LUT |
| Exponential interpolation | Difference × 8-bit fraction → interpolated exponential | 1 | 8 | 0 / inactive values set zero | Denominator sum |
| 36 | T66 context → T67 | 1113218944 | 46 | 9 / INT8 clip | 37–39 |
| 40 | Output FC accumulator → T71 | 747156156 | 39 | -13 / INT8 clip | 41 |
| 41 counterpart | Centered T71 → T72 `2^-16` | 939062400 | 22 | 0 / checked INT32 | 42 |
| After 45 | T76 fine `2^-46` → T76 `2^-23` | 1 | 23 | 0 / checked INT32 | 46 |
| After 54 | T85 fine `2^-39` → T85 `2^-16` | 1 | 23 | 0 / checked INT32 | 55 |
| 57 | T87 `2^-32` → T88 | 1009162 | 47 | 2 / INT8 clip | 58 |
| 58 | Expansion accumulator → T89 | 1274254450 | 39 | 37 / INT8 clip | 59–61 |
| 60 | Half-factor product → T92 | 1077952640 | 38 | 37 / INT8 clip | 70 |
| 62 | Integer cube → T94 | 1670962322 | 45 | 85 / INT8 clip | 63 |
| 63 | Cubic-coefficient product → T96 | 1500813243 | 38 | 85 / INT8 clip | 64 |
| 64 | Linear and cubic weighted sum → T97 | 1916390445, 228467824 | Common 31, one combined rounding | 37 / INT8 clip | 65 |
| 65 | Argument-coefficient product → T99 | 1077952560 | 38 | 37 / INT8 clip | 66–67 |
| 69 | T102 `2^-24` → T103 | 1269488229 | 47 | -128 / INT8 clip | 70 |
| **70 selected** | Final GELU product → T104 | **1195729761** | **31, then 6; distinct rounding stages** | -88 / INT8 clip | 71 |
| 71 | Final FC accumulator → T105 | 641059315 | 39 | 10 / INT8 clip | 72 |
| 72 counterpart | Centered T105 → T106 `2^-16` | 720118272 | 22 | 0 / checked INT32 | 73 |

The interpolation row is a rounded fractional computation at exponential scale, not a change to the `2^-30` exponential output scale. Op 70's single-rounding `RQ(product,1195729761,37)` is the preserved comparison, not the selected output.

### 5.2 Exact alignments and computations without a separate requantization

| Location | Action | Integer factor / operation | Resulting scale |
|---|---|---|---|
| Before Op 2 subtraction | ALIGN T0 to mean | ×128 | `2^-15` |
| Op 0 mean | COMPUTE through scale | Store 128-feature sum | `2^-15` |
| Op 3 variance | COMPUTE through scale | Store 128-feature square sum | `2^-22` |
| Op 5 RSQRT | COMPUTE | `isqrt(2^54 // q35)` | `2^-16` |
| Before Op 11 ADD | ALIGN T40 | ×128 | `2^-31` |
| Before Op 13 ADD | ALIGN beta | ×65536 | `2^-32` |
| Op 28 QK | COMPUTE dot product | Centered INT8 products, sum over 8 | `sQ*sK` |
| Op 31 probabilities | COMPUTE rounded ratio | `(exp_q*65536 + sum//2)//sum` | `2^-16` |
| Op 35 context | COMPUTE dot product | Probabilities × centered V, sum over 32 | `2^-16*sV` |
| Op 37–39 head merging | KEEP | Reshape, transpose, reshape | `s67` unchanged |
| Before Op 42 ADD | ALIGN T0 residual | ×256 | `2^-16` |
| Op 43 mean | COMPUTE through scale | Store 128-feature sum | `2^-23` |
| Before Op 45 subtraction | ALIGN T73 | ×128 | `2^-23` |
| Op 46 variance | COMPUTE through scale | Store 128-feature square sum | `2^-30` |
| Op 48 RSQRT | COMPUTE | `isqrt(2^62 // q78)` | `2^-16` |
| Before Op 54 ADD | ALIGN T83 | ×128 | `2^-39` |
| Before Op 56 ADD | ALIGN beta | ×65536 | `2^-32` |
| Op 61 cube | COMPUTE | `(q89-37)^3` | `s89^3` |
| Op 67 tanh | COMPUTE by table | Lookup for T99 code | `2^-24` |
| Op 68 +one | COMPUTE exact addition | +16777216 | `2^-24` unchanged |
| Op 73 final ADD | COMPUTE | `q106 + q73` | `2^-16` unchanged |

### 5.3 Dense parameter ledger

| Parameter | Meaning | Shape | Saved/candidate dtype | Scale granularity | Scale | Zero point |
|---|---|---|---|---|---|---:|
| T27 | Q weights | `[128,128]` | INT8 / INT8 | Per-tensor | `0.0017367515247315168` | 0 |
| T25 | K weights | `[128,128]` | INT8 / INT8 | Per-tensor | `0.0017351132119074464` | 0 |
| T26 | V weights | `[128,128]` | INT8 / INT8 | Per-tensor | `0.0006367104360833764` | 0 |
| T28 | Attention output weights | `[128,128]` | INT8 / INT8 | Per-tensor | `0.0006937290891073644` | 0 |
| T29 | MLP expansion weights | `[512,128]` | INT8 / INT8 | Per-tensor | `0.000803247734438628` | 0 |
| T30 | MLP output weights | `[128,512]` | INT8 / INT8 | Per-tensor | `0.0007148781442083418` | 0 |
| T24 | Attention output bias | `[128]` | INT32 / INT32 | Corresponding accumulator scale | `4.6429730686642335e-6` | 0 |
| T23 | Expansion bias | `[512]` | INT32 / INT32 | Corresponding accumulator scale | `2.6081866111904916e-5` | 0 |
| T22 | MLP output bias | `[128]` | INT32 / INT32 | Corresponding accumulator scale | `3.054867510894285e-6` | 0 |
| T6/T7 | First gamma/beta | `[128]` each | FP32 / INT32 | Candidate uniform format | `2^-16` | 0 |
| T8/T9 | Second gamma/beta | `[128]` each | FP32 / INT32 | Candidate uniform format | `2^-16` | 0 |

Current dense weights have one scale each; do not substitute metadata from an earlier q-projection export. Q/K/V have no biases. For future per-channel exports, bias scales and conversion multipliers must be derived per output channel; this document records the current per-tensor export.

### 5.4 Boundary-format ledger

| Boundary | Meaning | Saved graph dtype | Selected candidate format |
|---|---|---|---|
| T0 | Block input | FP32 | INT8 `[1,32,128]`, `2^-8`, z=0 |
| T45 | LN1 output / QKV input | INT8 | INT8, `0.03362198546528816`, z=-17 |
| T46/T47/T48 | Q/K/V outputs | INT8 | Independent scales/zero points in Part 2 |
| T59 | Scores | FP32 | INT32 `[16,32,32]`, `0.0004575034413775367`, z=0 |
| T62/T63 | Probabilities | FP32 | INT32, `2^-16`, z=0 |
| T66 | Context | FP32 | INT32 `[16,32,8]`, `1.0587832122155305e-7`, z=0 |
| T70 | Merged attention | INT8 | INT8 `[1,32,128]`, `0.006692775525152683`, z=9 |
| T73 | First residual | FP32 | INT32 `[1,32,128]`, `2^-16`, z=0 |
| T88 | LN2 output / MLP input | INT8 | INT8 `[1,32,128]`, `0.03247051313519478`, z=2 |
| T89 | Expansion output | INT8 | INT8 `[1,32,512]`, `0.011252585798501968`, z=37 |
| T93 | Cube | FP32 | INT32 `[1,32,512]`, `s89^3`, z=0 |
| T101/T102 | Tanh / +one | FP32 | INT32 `[1,32,512]`, `2^-24`, z=0 |
| T104 | GELU output | INT8 | INT8 `[1,32,512]`, `0.004273270256817341`, z=-88 |
| T105 | Final projection | INT8 | INT8 `[1,32,128]`, `0.0026197750121355057`, z=10 |
| T107 | Block output | FP32 | INT32 `[1,32,128]`, `2^-16`, z=0 |

Detailed per-part tables also cover broadcast, intermediate, and affine tensors. The original graph table in the appendix states saved dtypes for every operator edge.

## 6. Validation and complete-block results

### 6.1 Local, chained, and controls

“Local” starts the selected part from its reference boundary and then carries candidate operations through that part. It is not a rule that every individual operation is reset to its own reference inputs. A reference-input control isolates a particular operation.

| Part | Local starting point | Chained starting point |
|---|---|---|
| 1 | One selected INT8 T0 encoding compared to original FP32 T0 graph | This encoding starts the complete candidate |
| 2 | Reference T45 | Integer LN1 T45 |
| 3 | Reference-matching local QKV outputs | Chained LN1/QKV outputs |
| 4 | Reference T70; existing INT8 T0 residual | Chained T70; same existing INT8 T0 residual |
| 5 | Reference T73 encoded at `2^-16` | Chained integer T73 |
| 6 | Reference T88 | Chained integer LN2 T88 |
| 7 | Reference T104 and reference T73 encoded at `2^-16` | Selected chained T104 and existing chained T73 |

The complete C candidate, when implemented, must carry actual integer outputs between parts, corresponding to the **chained** path. C functional validation should require exact agreement with Python's selected chained integers. Agreement with TFLite is a separate numerical-quality comparison.

### 6.2 Metric definitions

Let reference output $y_i$ and represented candidate output $\hat y_i=q_{107,i}2^{-16}$ have errors $e_i=\hat y_i-y_i$.

$$
\mathrm{MAE}=\frac1n\sum_i|e_i|,\qquad
\mathrm{RMSE}=\sqrt{\frac1n\sum_i e_i^2}.
$$

$$
\mathrm{Relative\ L2\ error}
=\frac{\sqrt{\sum_i e_i^2}}{\sqrt{\sum_i y_i^2}}\times100\%.
$$

Exact-match percentage counts identical quantized codes. Relative L2 measures error-vector magnitude relative to reference-vector magnitude. Neither is a language-model accuracy percentage. Relative L2 uses the full flattened test group: 4096 benchmark output values or 28,672 calibration output values. It is not the maximum per-sequence relative error.

### 6.3 Final-part local and complete chained T107

| Measurement | Benchmark | Calibration |
|---|---:|---:|
| Local maximum absolute error | `1.500546932e-5` | `1.498311758e-5` |
| Local mean absolute error | `5.079226639e-6` | `5.080643145e-6` |
| Local RMSE | `6.209905247e-6` | `6.219611354e-6` |
| Local relative L2 error | 0.005232% | 0.005238% |
| **Chained maximum absolute error** | **`1.019726694e-2`** | **`8.447572589e-3`** |
| **Chained mean absolute error** | **`1.841633877e-3`** | **`1.812179604e-3`** |
| **Chained RMSE** | **`2.294960230e-3`** | **`2.252417781e-3`** |
| **Chained relative L2 error** | **1.933702%** | **1.896908%** |

| Stored T107 range | Benchmark | Calibration |
|---|---|---|
| Local | -42813 to 34986 | -42813 to 39704 |
| Chained | -42556 to 34935 | -42556 to 39330 |

The chained block output is numerically close to the mixed-precision reference on these cases, establishing a working baseline. This does not establish complete-model logits, token agreement, generation quality, or performance on unseen inputs. Later blocks may amplify or reduce these differences. Input INT8 quantization is an established contributor to the current error; wider later storage cannot recover that lost information.

## 7. Workload and storage facts

### 7.1 Dense algorithmic MAC counts for one sequence

| Operation | Matrix dimensions | MACs |
|---|---|---:|
| Q projection | `[32,128] × [128,128]` | 524,288 |
| K projection | Same | 524,288 |
| V projection | Same | 524,288 |
| QK-transpose | 16 × `[32,8] × [8,32]` | 131,072 |
| Probability × V | 16 × `[32,32] × [32,8]` | 131,072 |
| Attention output projection | `[32,128] × [128,128]` | 524,288 |
| MLP expansion | `[32,128] × [128,512]` | 2,097,152 |
| MLP output projection | `[32,512] × [512,128]` | 2,097,152 |
| **Total dense matrix-product MACs** | | **6,553,600** |

The attention count includes the full dense matrices even where probabilities are masked zero. Nonlinear arithmetic, normalization, requantization, layout operations, loads, and stores require additional work. These are algorithmic counts, not Spike instruction measurements or hardware cycle measurements.

### 7.2 Known parameter/table payloads

| Item | Entries | Storage | Bytes |
|---|---:|---|---:|
| Four 128×128 projection weight matrices | 65,536 total | INT8 | 65,536 |
| Two 128×512 / 512×128 MLP weight matrices | 131,072 total | INT8 | 131,072 |
| Three FC biases: 128 + 512 + 128 | 768 total | INT32 | 3,072 |
| Two LayerNorm gamma/beta pairs | 512 total | INT32 candidate constants | 2,048 |
| Exponential LUT | 4,097 | INT32 | 16,388 |
| Tanh LUT | 256 | INT32 | 1,024 |
| **Subtotal of listed payloads** | | | **219,140** |

This subtotal is not the final RISC-V RAM/ROM footprint. Metadata, masks, other constants, code, activation buffers, alignment, and buffer reuse must be accounted for when designing static C allocation. Python trial dictionaries retain diagnostic arrays that a runtime need not retain.

## 8. Completion status and reproduction

### 8.1 Selected candidate and result variables

| Part / purpose | Analysis variable |
|---|---|
| Input cases | `op0_cases` |
| First LayerNorm final output | `ln1_affine_integer_trial` |
| Projection metadata / outputs | `qkv_specs` / `qkv_integer_trial` |
| Attention scores | `attention_score_trial` |
| Masked softmax inputs / output | `attention_softmax_inputs` / `softmax_integer_trial` |
| Attention context / merged output | `attention_context_trial` / `attention_merge_integer_trial` |
| Attention projection/residual | `out_residual_integer_trial` |
| Second LayerNorm final output | `ln2_affine_integer_trial` |
| Expansion | `mlp_expand_integer_trial` |
| GELU selected output | **`gelu_output_two_round_trial`** |
| GELU single-rounding comparison | `gelu_output_integer_trial` |
| Final projection preparation | `mlp_output_cases` |
| Complete selected block output | **`block0_integer_trial`** |

```python
# Selected complete chained block output for case i:
q107 = block0_integer_trial[i]["paths"]["chained"]["t107_q"]
# Diagnostic reconstruction only:
y_represented = q107.astype(np.float64) * (2.0 ** -16)
```

After a kernel restart, rerun analysis setup and subsequent analysis cells in order. Do not restart quantization or regenerate the fixed exports. The originally attached notebook ended at the softmax stage; later stages are the cells executed during this session and must be retained in the working analysis notebook.

### 8.2 Read-only full-precision GELU metadata check

Some scale displays in Part 6 were rounded when reported. This snippet reads the existing allocated interpreter and prints full-precision values; it does not alter the export or rerun quantization.

```python
details = {int(d["index"]): d for d in interpreter.get_tensor_details()}
for tid in [89, 91, 92, 94, 95, 96, 97, 98, 99, 103, 104]:
    p = details[tid]["quantization_parameters"]
    assert len(p["scales"]) == len(p["zero_points"]) == 1
    print(f"T{tid}: scale={float(p['scales'][0]):.17g}, "
          f"zero_point={int(p['zero_points'][0])}")
```

The verified integer constants in Section 5 should be retained when reproducing the selected candidate. If metadata is refreshed for documentation, replace only the explicitly approximate displays with the actual fixed-export values.

### 8.3 What is complete and what remains

All seven parts (74 original operations, Op 0–73) have Python integer counterparts and tested comparisons. The current candidate preserves the original INT8 T0 baseline. The corrected two-stage final GELU trial is selected; the prior single-rounding trial remains a comparison.

Outstanding work:

- Unseen-input validation and an isolated higher-precision T0 trial.
- A callable, reusable integer block function and testing beyond precomputed cases.
- Remaining seven blocks and a tested inter-block representation/interface.
- Final-model logits, next-token agreement, validation loss/perplexity, and generation quality.
- Static C kernels/allocation, bit-exact C-versus-Python validation, and new RV64IMC/Spike measurements for this candidate.
- Packed MAC/requantization semantics, custom Spike instructions, RTL integration, and hardware evaluation.

An earlier q-projection C/Spike baseline exists in the project. It does not constitute C/Spike validation of the new fixed export or this complete block candidate.

## 9. Appendix: original graph and layout constants

### 9.1 Saved tensor types versus candidate representations

The following operator table is transcribed from the saved output of the attached analysis notebook. `float32` entries are the original graph types; the candidate uses the formats above. Shape/axis constants are original integer graph metadata and remain offline/static metadata for the candidate.

| Op | Operation | Inputs | Outputs | Classification |
|---:|---|---|---|---|
| 0 | MEAN | T0 float32, T10 int32 | T31 float32 | Floating-point / mixed |
| 1 | RESHAPE | T31 float32, T11 int32 | T32 float32 | Data movement / shape |
| 2 | SQUARED_DIFFERENCE | T0 float32, T32 float32 | T33 float32 | Floating-point / mixed |
| 3 | MEAN | T33 float32, T10 int32 | T34 float32 | Floating-point / mixed |
| 4 | ADD | T34 float32, T2 float32 | T35 float32 | Floating-point / mixed |
| 5 | RSQRT | T35 float32 | T36 float32 | Floating-point / mixed |
| 6 | MUL | T36 float32, T31 float32 | T37 float32 | Floating-point / mixed |
| 7 | SUB | T5 float32, T37 float32 | T38 float32 | Floating-point / mixed |
| 8 | RESHAPE | T36 float32, T11 int32 | T39 float32 | Data movement / shape |
| 9 | MUL | T0 float32, T39 float32 | T40 float32 | Floating-point / mixed |
| 10 | RESHAPE | T38 float32, T11 int32 | T41 float32 | Data movement / shape |
| 11 | ADD | T40 float32, T41 float32 | T42 float32 | Floating-point / mixed |
| 12 | MUL | T42 float32, T6 float32 | T43 float32 | Floating-point / mixed |
| 13 | ADD | T43 float32, T7 float32 | T44 float32 | Floating-point / mixed |
| 14 | QUANTIZE | T44 float32 | T45 int8 | Quantize / requantize |
| 15 | FULLY_CONNECTED | T45 int8, T27 int8 | T46 int8 | INT8 computation |
| 16 | FULLY_CONNECTED | T45 int8, T25 int8 | T47 int8 | INT8 computation |
| 17 | FULLY_CONNECTED | T45 int8, T26 int8 | T48 int8 | INT8 computation |
| 18 | RESHAPE | T46 int8, T13 int32 | T49 int8 | Data movement / shape |
| 19 | TRANSPOSE | T49 int8, T14 int32 | T50 int8 | Data movement / shape |
| 20 | RESHAPE | T47 int8, T13 int32 | T51 int8 | Data movement / shape |
| 21 | RESHAPE | T48 int8, T13 int32 | T52 int8 | Data movement / shape |
| 22 | TRANSPOSE | T52 int8, T14 int32 | T53 int8 | Data movement / shape |
| 23 | TRANSPOSE | T51 int8, T1 int32 | T54 int8 | Data movement / shape |
| 24 | RESHAPE | T50 int8, T15 int32 | T55 int8 | Data movement / shape |
| 25 | DEQUANTIZE | T55 int8 | T56 float32 | Dequantize |
| 26 | RESHAPE | T54 int8, T16 int32 | T57 int8 | Data movement / shape |
| 27 | DEQUANTIZE | T57 int8 | T58 float32 | Dequantize |
| 28 | BATCH_MATMUL | T56 float32, T58 float32 | T59 float32 | Floating-point / mixed |
| 29 | RESHAPE | T59 float32, T17 int32 | T60 float32 | Data movement / shape |
| 30 | SELECT_V2 | T21 bool, T60 float32, T18 float32 | T61 float32 | Floating-point / mixed |
| 31 | SOFTMAX | T61 float32 | T62 float32 | Floating-point / mixed |
| 32 | RESHAPE | T62 float32, T19 int32 | T63 float32 | Data movement / shape |
| 33 | RESHAPE | T53 int8, T15 int32 | T64 int8 | Data movement / shape |
| 34 | DEQUANTIZE | T64 int8 | T65 float32 | Dequantize |
| 35 | BATCH_MATMUL | T63 float32, T65 float32 | T66 float32 | Floating-point / mixed |
| 36 | QUANTIZE | T66 float32 | T67 int8 | Quantize / requantize |
| 37 | RESHAPE | T67 int8, T20 int32 | T68 int8 | Data movement / shape |
| 38 | TRANSPOSE | T68 int8, T14 int32 | T69 int8 | Data movement / shape |
| 39 | RESHAPE | T69 int8, T12 int32 | T70 int8 | Data movement / shape |
| 40 | FULLY_CONNECTED | T70 int8, T28 int8, T24 int32 | T71 int8 | INT8 computation |
| 41 | DEQUANTIZE | T71 int8 | T72 float32 | Dequantize |
| 42 | ADD | T72 float32, T0 float32 | T73 float32 | Floating-point / mixed |
| 43 | MEAN | T73 float32, T10 int32 | T74 float32 | Floating-point / mixed |
| 44 | RESHAPE | T74 float32, T11 int32 | T75 float32 | Data movement / shape |
| 45 | SQUARED_DIFFERENCE | T73 float32, T75 float32 | T76 float32 | Floating-point / mixed |
| 46 | MEAN | T76 float32, T10 int32 | T77 float32 | Floating-point / mixed |
| 47 | ADD | T77 float32, T2 float32 | T78 float32 | Floating-point / mixed |
| 48 | RSQRT | T78 float32 | T79 float32 | Floating-point / mixed |
| 49 | MUL | T79 float32, T74 float32 | T80 float32 | Floating-point / mixed |
| 50 | SUB | T5 float32, T80 float32 | T81 float32 | Floating-point / mixed |
| 51 | RESHAPE | T79 float32, T11 int32 | T82 float32 | Data movement / shape |
| 52 | MUL | T73 float32, T82 float32 | T83 float32 | Floating-point / mixed |
| 53 | RESHAPE | T81 float32, T11 int32 | T84 float32 | Data movement / shape |
| 54 | ADD | T83 float32, T84 float32 | T85 float32 | Floating-point / mixed |
| 55 | MUL | T85 float32, T8 float32 | T86 float32 | Floating-point / mixed |
| 56 | ADD | T86 float32, T9 float32 | T87 float32 | Floating-point / mixed |
| 57 | QUANTIZE | T87 float32 | T88 int8 | Quantize / requantize |
| 58 | FULLY_CONNECTED | T88 int8, T29 int8, T23 int32 | T89 int8 | INT8 computation |
| 59 | DEQUANTIZE | T89 int8 | T90 float32 | Dequantize |
| 60 | MUL | T89 int8, T91 int8 | T92 int8 | INT8 computation |
| 61 | POW | T90 float32, T3 float32 | T93 float32 | Floating-point / mixed |
| 62 | QUANTIZE | T93 float32 | T94 int8 | Quantize / requantize |
| 63 | MUL | T94 int8, T95 int8 | T96 int8 | INT8 computation |
| 64 | ADD | T89 int8, T96 int8 | T97 int8 | INT8 computation |
| 65 | MUL | T97 int8, T98 int8 | T99 int8 | INT8 computation |
| 66 | DEQUANTIZE | T99 int8 | T100 float32 | Dequantize |
| 67 | TANH | T100 float32 | T101 float32 | Floating-point / mixed |
| 68 | ADD | T101 float32, T4 float32 | T102 float32 | Floating-point / mixed |
| 69 | QUANTIZE | T102 float32 | T103 int8 | Quantize / requantize |
| 70 | MUL | T92 int8, T103 int8 | T104 int8 | INT8 computation |
| 71 | FULLY_CONNECTED | T104 int8, T30 int8, T22 int32 | T105 int8 | INT8 computation |
| 72 | DEQUANTIZE | T105 int8 | T106 float32 | Dequantize |
| 73 | ADD | T73 float32, T106 float32 | T107 float32 | Floating-point / mixed |

### 9.2 Head layouts used by the candidate

| Op | Action | Input layout | Output layout / permutation |
|---|---|---|---|
| 18 | Q reshape | `[1,32,128]` | `[1,32,16,8]` |
| 19 | Q transpose | `[1,32,16,8]` | `[1,16,32,8]`, `(0,2,1,3)` |
| 20 | K reshape | `[1,32,128]` | `[1,32,16,8]` |
| 21 | V reshape | `[1,32,128]` | `[1,32,16,8]` |
| 22 | V transpose | `[1,32,16,8]` | `[1,16,32,8]`, `(0,2,1,3)` |
| 23 | K transpose | `[1,32,16,8]` | `[1,16,8,32]`, `(0,2,3,1)` |
| 24 | Q matrix reshape | `[1,16,32,8]` | `[16,32,8]` |
| 26 | K matrix reshape | `[1,16,8,32]` | `[16,8,32]` |
| 29 | Score reshape | `[16,32,32]` | `[1,16,32,32]` |
| 32 | Probability reshape | `[1,16,32,32]` | `[16,32,32]` |
| 33 | V matrix reshape | `[1,16,32,8]` | `[16,32,8]` |
| 37 | Context reshape | `[16,32,8]` | `[1,16,32,8]` |
| 38 | Context transpose | `[1,16,32,8]` | `[1,32,16,8]`, `(0,2,1,3)` |
| 39 | Merge heads | `[1,32,16,8]` | `[1,32,128]` |

Each layout operation keeps the source scale/zero point. Layout constants are read from the fixed graph during analysis; these shapes describe the tested batch-one, sequence-32 candidate.

### 9.3 Symbol-to-tensor map for block equations

| Symbol | Meaning | Graph/candidate boundary |
|---|---|---|
| X | Block input | T0 |
| H0 | First LayerNorm real output / projection encoding | T44 / T45 |
| Q, K, V | Projection outputs | T46, T47, T48 |
| S | Attention scores | T59 / T60 |
| P | Attention probabilities | T62 / T63 |
| C | Per-head context | T66 / T67 |
| A | Merged context | T70 |
| R | First residual output | T73 |
| H1 | Second LayerNorm real output / MLP encoding | T87 / T88 |
| U | MLP expansion output | T89 / original dequantized T90 |
| GELU(U) | Final nonlinear output | T104 |
| Y | Final block output | T107 |

