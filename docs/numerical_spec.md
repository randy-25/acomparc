# TinyStories blocks 0–7: integer numerical specification

Extracted from the uploaded, executed notebook. This defines the current candidate, not a claim of unseen-data model quality.

## Part 1 — Integer LN1, shared across blocks 0–7

### Input, output, and parameters

- Input: INT8 `[1,32,128]`, scale `2^-F`, zero point `0`.
- Output: INT8 `[1,32,128]`, using that block’s saved TFLite LN1 output scale and zero point.
- Gamma and beta: encoded offline as INT32 at scale `2^-16`.
- Intermediate computations: checked INT32/INT64 arithmetic.
- Feature count: `N = 128`.

| Blocks | F | Square/normalize shift | Variance scale | Integer epsilon | RSQRT numerator |
|---|---:|---:|---|---:|---|
| 0, 2–7 | 8 | 15 | `2^-22` | 42 | `2^54` |
| 1 | 7 | 14 | `2^-21` | 21 | `2^53` |

The algorithm is identical across blocks; parameters depend on the input scale and saved model metadata.

### Rounding rule

For centered integer `a`:

$$
RQ(a,M,r)=(aM+2^{r-1})\gg r
$$

The shift is arithmetic. Runtime rounding is nearest, with ties toward positive infinity, matching XREQUANT.

Offline parameter encoding uses nearest rounding with ties to even.

### 1. Mean and centered differences

For each token:

$$
m=\sum_{k=0}^{127}q_x[k]
$$

Store `m` as INT32 at scale `2^-(F+7)`. The feature sum represents the mean through this scale; no division is required.

Align the input before subtraction:

$$
d[k]=128q_x[k]-m
$$

Compute `d` using INT64 at scale `2^-(F+7)`.

### 2. Square and variance

Compute:

$$
u[k]=d[k]^2
$$

`u` uses INT64 at scale `2^-2(F+7)`.

**Requantization 1 — after squaring:**

$$
v[k]=RQ(u[k],1,F+7)
$$

Store `v` as INT32 at scale `2^-(F+7)`.

Sum the features:

$$
V=\sum_k v[k]
$$

Store `V` as INT32 at scale `2^-(F+14)`, representing the variance.

Encode epsilon at that same scale:

$$
\epsilon_q=\mathrm{round}_{even}(\epsilon\,2^{F+14})
$$

### 3. Inverse standard deviation

$$
I=\mathrm{isqrt} \left( \left\lfloor\frac{2^{F+46}}{V+\epsilon_q}\right\rfloor \right)
$$

`isqrt` returns the floor integer square root.

Store `I` as INT32 at scale `2^-16`. The denominator must be positive.

### 4. Normalize

Compute the INT64 product:

$$
n_{\mathrm{fine}}[k]=d[k]I
$$

Its scale is `2^-(F+23)`.

**Requantization 2 — after normalization:**

$$
n[k]=RQ(n_{\mathrm{fine}}[k],1,F+7)
$$

Store `n` as INT32 at scale `2^-16`.

### 5. Apply gamma and beta

Offline parameter encoding:

$$
\gamma_q=\mathrm{round}_{even}(\gamma\,2^{16}), \qquad \beta_q=\mathrm{round}_{even}(\beta\,2^{16})
$$

Runtime affine calculation:

$$
a[k]=n[k]\gamma_q[k]+\beta_q[k]\,2^{16}
$$

Store `a` as INT64 at scale `2^-32`.

Beta alignment is exact; it does not require rounded requantization.

### 6. Convert into the projection input format

Let `s_out` and `z_out` be this block’s saved TFLite LN1 output scale and zero point.

Prepare the multiplier offline:

$$
M=\mathrm{round}_{even} \left( \frac{2^{-32}}{s_{\mathrm{out}}}\,2^{47} \right)
$$

**Requantization 3 — after the affine calculation:**

$$
q_{\mathrm{out}}[k] = \mathrm{clip}_{[-128,127]} \left(RQ(a[k],M,47)+z_{\mathrm{out}}\right)
$$

This INT8 output feeds all three Q/K/V projections.

### Validation

- Check every intended INT32/INT64 result before storing it.
- Compare integer LN1 with TFLite LN1 using the same represented INT8 input.
- Current tests: eight cases per block; maximum output difference is one INT8 code.
- This is local LN1 validation, not full-chain inference validation.

## Part 2 — Integer Q/K/V projections, shared across blocks 0–7

### Input, output, and parameters

- Input: LN1 output, INT8 `[1,32,128]`.
- Weights: three INT8 matrices `[128,128]`, stored as `[output,input]`.
- Weight zero points: `0`.
- Bias: none for Q/K/V in the current exports.
- Accumulators: INT32 `[1,32,128]`.
- Outputs: three INT8 tensors `[1,32,128]`.
- Each projection uses its own weight scale, output scale, and output zero point.

Weights and quantization metadata come directly from each block’s saved TFLite.

### 1. Center the input

$$
x_c[t,k]=q_x[t,k]-z_x
$$

Promote before subtraction. Centered values may exceed INT8 range.

### 2. Accumulate the dot products

For each projection $p\in\{Q,K,V\}$:

$$
A_p[t,o]=\sum_{k=0}^{127}x_c[t,k]q_{W_p}[o,k]
$$

The accumulator scale is:

$$
s_{A_p}=s_xs_{W_p}
$$

- Store accumulators as INT32.
- Every product and partial sum must fit INT32.
- No requantization occurs inside the dot product.
- The C implementation uses XTILE for accumulation.

### 3. Prepare the conversion multiplier offline

Use the common right shift `r = 39`:

$$
M_p=\mathrm{round}_{even} \left( \frac{s_xs_{W_p}}{s_{\mathrm{out},p}}\,2^{39} \right)
$$

Store the multiplier as INT64. It may exceed positive INT32.

Check that accumulator multiplication and rounding-offset addition fit INT64.

### 4. Requantize immediately after accumulation

Runtime rounding:

$$
RQ(A,M,39)=(AM+2^{38})\gg39
$$

This is nearest rounding, with ties toward positive infinity.

**Requantization point — after each completed projection:**

$$
q_p[t,o]= \mathrm{clip}_{[-128,127]} \left( RQ(A_p[t,o],M_p,39)+z_{\mathrm{out},p} \right)
$$

The C implementation uses XREQUANT mode 2: INT32 → INT8.

Q/K/V are converted before attention head layout operations.
Reshape and transpose preserve their codes, scales, and zero points.

### Validation

Two paths are checked against TFLite:

1. Projection only: use TFLite LN1 output to isolate projection arithmetic.
2. LN1 → Q/K/V: use integer LN1 output to measure propagated differences.

Current results across eight blocks and eight cases per block:

- Projection only: 786,432 INT8 outputs, zero mismatches.
- LN1 → Q/K/V agreement: 98.083%–99.756%.
- Maximum chained difference: one INT8 code.
- Local and chained saturation counts are identical.
- Parameter and arithmetic-bound checks passed.

“Chained” here means LN1 → Q/K/V within each block, not the complete multi-block integer inference path.

## Part 3 — Integer attention, shared across blocks 0–7

### Inputs, output, and layout

- Inputs: Q/K/V, each INT8 `[1,32,128]`, with independent scales and zero points.
- Heads: 16; head dimension: 8; sequence length: 32.
- Output: merged context, INT8 `[1,32,128]`.
- The output scale and zero point come from each block’s saved TFLite.
- Head layout changes preserve integer codes and quantization.
- Retain the existing software attention algorithm.
- Use XREQUANT mode 2 for the final context conversion.

### 1. Compute QK scores

For each head:

$$
S[h,t,j]=\sum_{d=0}^{7} (q_Q[h,t,d]-z_Q)(q_K[h,j,d]-z_K)
$$

Store scores as INT32 `[16,32,32]` at scale:

$$
s_S=s_Qs_K
$$

Promote before zero-point subtraction.
Check product and partial-sum bounds.

No requantization occurs inside this dot product.
No additional attention scaling factor is inserted.

### 2. Apply causal masking and subtract the row maximum

Only keys `j <= t` participate:

$$
m[h,t]=\max_{j\le t}S[h,t,j]
$$

$$
D[h,t,j]=S[h,t,j]-m[h,t]
$$

Allowed shifted scores are nonpositive and retain scale `s_S`.
Masked entries must contribute zero probability.

### 3. Convert scores into exponential-table coordinates

Prepare the multiplier offline:

$$
M_{\mathrm{coord}} =\mathrm{round}_{even}(s_S\,2^{16}\,2^{24})
$$

**Rounded coordinate conversion:**

$$
c=RQ(-D,M_{\mathrm{coord}},24)
$$

Coordinates use scale `2^-16`.
The multiplier and product use checked INT64 arithmetic.

### 4. Approximate exponentials

Shared offline table:

$$
L[i]=\mathrm{round}_{even} \left(e^{-i/256}2^{30}\right), \qquad i=0,\ldots,4096
$$

For allowed entries:

- If `c >= 16 × 65536`, set the exponential to zero.
- Otherwise, calculate:

$$
i=\left\lfloor\frac{c}{256}\right\rfloor,\qquad f=c-256i
$$

$$
E=L[i]+RQ((L[i+1]-L[i])f,1,8)
$$

Store exponentials as INT32 at scale `2^-30`.
Interpolation rounding preserves that output scale.

Set masked exponentials to zero.

### 5. Normalize probabilities

Sum exponentials using INT64:

$$
Z[h,t]=\sum_{j\le t}E[h,t,j]
$$

The denominator must be positive.

Compute independently rounded probabilities:

$$
P_q[h,t,j] = \left\lfloor \frac{E[h,t,j]2^{16}+\lfloor Z[h,t]/2\rfloor} {Z[h,t]} \right\rfloor
$$

Store probabilities as INT32 at scale `2^-16`.

- Probability code `65536` represents `1`.
- Masked probabilities are zero.
- Do not force independently rounded row sums to equal `65536`.

### 6. Compute probability × V

$$
C[h,t,d] =\sum_{j=0}^{31} P_q[h,t,j](q_V[h,j,d]-z_V)
$$

Store context accumulators as INT32 at scale:

$$
s_C=2^{-16}s_V
$$

Check product and partial-sum bounds.
No requantization occurs inside the dot product.

Merge heads into token-major `[1,32,128]`.
This layout change preserves values and scale.

### 7. Requantize the context into INT8

Let `s_out` and `z_out` be the saved context-output quantization.

Prepare offline:

$$
M_C=\mathrm{round}_{even} \left(\frac{2^{-16}s_V}{s_{\mathrm{out}}}2^{46}\right)
$$

**Final context requantization:**

$$
q_{\mathrm{out}} = \mathrm{clip}_{[-128,127]} \left(RQ(C,M_C,46)+z_{\mathrm{out}}\right)
$$

Use an INT64 multiplier and checked INT64 product/rounding addition.
The output feeds Part 4’s attention output projection.

### Validation

Local path:
reference Q/K/V → integer attention.

Chained path:
integer LN1 → integer Q/K/V → integer attention.

Current eight-case results across all eight blocks:

- Local merged-output agreement: 99.927%–100%.
- Local maximum difference: one INT8 code.
- Local maximum probability error: approximately 1.31e-5.
- Maximum probability row-sum deviation: six integer steps at scale 2^-16.
- Chained context relative L2: approximately 0.169%–1.403%.
- Chained maximum merged-output difference: 14 INT8 codes.
- Local and chained saturation counts are identical.
- Arithmetic checks completed without overflow.

Retain the block-1 chained outlier for full-chain evaluation.
These results validate the current stage candidate on the tested cases;
they do not establish complete-model accuracy.

## Part 4 — Output projection and first residual, shared across blocks 0–7

### Input, output, and parameters

- Attention input: INT8 `[1,32,128]`.
- Original block input: INT8 `[1,32,128]`, scale `2^-F`, zero point `0`.
- Weights: INT8 `[128,128]`, stored as `[output,input]`, zero point `0`.
- Bias: saved INT32 `[128]`, at the projection accumulator scale.
- Final residual output: INT32 `[1,32,128]`, scale `2^-16`, zero point `0`.

Each block supplies its own weights, bias, quantization metadata, and multipliers.

### 1. Projection accumulation and bias

$$
A[t,o] = \sum_{k=0}^{127}(q_{\mathrm{attention}}[t,k]-z_{\mathrm{attention}}) q_W[o,k] +b_q[o]
$$

The accumulator scale is:

$$
s_A=s_{\mathrm{attention}}s_W
$$

- XTILE computes the dot products.
- Software adds the bias before requantization.
- Products, partial sums, and bias addition must fit INT32.
- No requantization occurs inside the dot product.

### 2. Preserve the INT8 projection output

Prepare the multiplier offline:

$$
M_P=\mathrm{round}_{even} \left(\frac{s_A}{s_P}2^{39}\right)
$$

**Requantization 1 — after accumulation and bias:**

$$
q_P= \mathrm{clip}_{[-128,127]} \left(RQ(A,M_P,39)+z_P\right)
$$

Here `s_P` and `z_P` are the saved TFLite projection-output quantization.

Use an INT64 multiplier and checked INT64 product/rounding arithmetic.
The C implementation uses XREQUANT mode 2.

### 3. Convert the projection branch to the residual scale

Center the INT8 projection output:

$$
c_P=q_P-z_P
$$

Prepare offline:

$$
M_R=\mathrm{round}_{even} \left(s_P\,2^{16}\,2^{22}\right)
$$

**Requantization 2 — before residual addition:**

$$
q_{\mathrm{branch}}=RQ(c_P,M_R,22)
$$

Store as INT32 at scale `2^-16`, zero point `0`.

This conversion does not saturate to INT8.
The result must fit INT32.
The C implementation uses XREQUANT mode 3.

### 4. Align the original block input exactly

$$
q_{\mathrm{aligned}}=q_{\mathrm{input}}2^{16-F}
$$

| Blocks | Input scale | Exact alignment |
|---|---|---:|
| 0, 2–7 | `2^-8` | 256 |
| 1 | `2^-7` | 512 |

This multiplication is exact; it introduces no rounding.

### 5. Add the residual branches

Both branches now use scale `2^-16`:

$$
q_R=q_{\mathrm{branch}}+q_{\mathrm{aligned}}
$$

Check the sum before INT32 storage.

There is no additional requantization or INT8 saturation here.
Retain `q_R` for:
- Part 5: second LayerNorm.
- Part 7: final residual connection.

### Rounding rule

$$
RQ(a,M,r)=(aM+2^{r-1})\gg r
$$

Runtime rounding is nearest, with ties toward positive infinity.
Offline multiplier preparation uses nearest, ties to even.

### Validation

Local path:
reference attention output → integer Part 4.

Chained path:
integer LN1 → Q/K/V → attention → Part 4.

Both paths use the same encoded original block input for the residual.

Current results across eight blocks and eight cases per block:

- Local projection: 262,144 outputs, zero mismatches.
- Local residual RMSE: approximately 4.4e-6.
- Chained projection maximum difference: five INT8 codes.
- Chained residual relative L2: approximately 0.372%–1.446%.
- Parameter and checked arithmetic tests completed without overflow.

These are within-block stage comparisons.
Complete multi-block integer inference validation remains pending.

## Part 5 — Integer LN2, shared across blocks 0–7

### Input, output, and parameters

- Input: first residual, INT32 `[1,32,128]`, scale `2^-16`, zero point `0`.
- Output: INT8 `[1,32,128]`, using the block’s saved LN2 output scale and zero point.
- Gamma/beta: independent LN2 parameters, encoded offline as INT32 at scale `2^-16`.
- Reuse the shared integer LayerNorm algorithm.

Common settings:

| Parameter | Value |
|---|---|
| Feature count | 128 |
| Square conversion shift | 23 |
| Normalization conversion shift | 23 |
| Variance scale | `2^-30` |
| Integer epsilon | 10737 |
| Reciprocal-square-root numerator | `2^62` |
| Inverse-standard-deviation scale | `2^-16` |
| Final conversion shift | 47 |

### 1. Mean and centered differences

For each token:

$$
m=\sum_{k=0}^{127}q_R[k]
$$

Store the sum as INT32 at scale `2^-23`, representing the mean.

Align and subtract:

$$
d[k]=128q_R[k]-m
$$

Compute using INT64 at scale `2^-23`.

### 2. Squared differences and variance

Compute `d²` using INT64 at scale `2^-46`.

**Requantization 1 — after squaring:**

$$
u[k]=RQ(d[k]^2,1,23)
$$

Store `u` as INT32 at scale `2^-23`.

Sum over features:

$$
V=\sum_k u[k]
$$

Store `V` as INT32 at scale `2^-30`, representing variance.

### 3. Inverse standard deviation

$$
I=\mathrm{isqrt} \left( \left\lfloor\frac{2^{62}}{V+10737}\right\rfloor \right)
$$

The denominator must be positive.
Store `I` as INT32 at scale `2^-16`.

### 4. Normalize

Compute `d × I` using INT64 at scale `2^-39`.

**Requantization 2 — after normalization:**

$$
n[k]=RQ(d[k]I,1,23)
$$

Store `n` as INT32 at scale `2^-16`.

### 5. Apply gamma and beta

Offline encoding:

$$
\gamma_q=\mathrm{round}_{even}(\gamma2^{16}), \qquad \beta_q=\mathrm{round}_{even}(\beta2^{16})
$$

Runtime affine calculation:

$$
a[k]=n[k]\gamma_q[k]+\beta_q[k]2^{16}
$$

The affine result uses INT64 at scale `2^-32`.
Beta alignment is exact.

### 6. Convert to the INT8 MLP input

Read `s_out` and `z_out` from the block’s saved LN2 output metadata.

Prepare offline:

$$
M=\mathrm{round}_{even} \left(\frac{2^{-32}}{s_{\mathrm{out}}}2^{47}\right)
$$

**Requantization 3 — after the affine calculation:**

$$
q_{\mathrm{out}}= \mathrm{clip}_{[-128,127]} \left(RQ(a,M,47)+z_{\mathrm{out}}\right)
$$

This output feeds Part 6’s MLP expansion.
Retain the original first residual separately for Part 7.

### Arithmetic rules

$$
RQ(a,M,r)=(aM+2^{r-1})\gg r
$$

- Runtime rounding: nearest, ties toward positive infinity.
- Offline parameter encoding: nearest, ties to even.
- Check all intended INT32/INT64 results before storing.
- No floating-point arithmetic is required at runtime.

### Validation

Local:
encode the reference FP32 residual at scale `2^-16`, then run integer LN2.
This includes residual-encoding error and LN2 approximation error.

Chained:
integer Parts 1–4 → integer LN2, without re-encoding the residual.

Current eight-case results across all blocks:

- Local agreement: 99.658%–99.927%.
- Local maximum difference: one INT8 code.
- Chained agreement: 83.698%–97.607%.
- Chained maximum difference: six INT8 codes.
- Saturation counts differ by one value in two calibration groups.
- Checked arithmetic completed without overflow.

Complete integer block and multi-block inference validation remain pending.

## Part 6 — MLP expansion and GELU, shared across blocks 0–7

### Input and output

- Input: LN2 output, INT8 `[1,32,128]`.
- Expansion: INT8 weights `[512,128]`, INT32 bias.
- Expansion and GELU outputs: INT8 `[1,32,512]`.
- Read each block’s constants, scales, and zero points from its saved TFLite.

### Expansion

$$
A[t,o]=\sum_k(q_x[t,k]-z_x)q_W[o,k]+b_q[o]
$$

XTILE computes the dot products; software adds bias.

**Requantize after accumulation and bias:**

$$
q_U=\mathrm{clip}_{INT8}(RQ(A,M_U,39)+z_U)
$$

$$
M_U=\mathrm{round}_{even} \left(\frac{s_xs_W}{s_U}2^{39}\right)
$$

### GELU sequence and conversion points

Use saved quantized coefficients, not replacements from the conceptual floating-point GELU formula.

| Stage | Computation | Conversion |
|---|---|---|
| Half branch | Centered expansion × centered half coefficient | INT8, shift 38 |
| Cube | Centered expansion cubed in INT32 | INT8, shift 45 |
| Weighted cube | Centered cube × centered cubic coefficient | INT8, shift 38 |
| Branch sum | Multiply centered linear/cubic branches by separate scale multipliers; add in INT64 | One combined rounding, shift 31, then INT8 |
| Tanh argument | Centered sum × centered argument coefficient | INT8, shift 38 |
| Tanh | Lookup indexed by argument code + 128 | INT32 at scale `2^-24` |
| One plus tanh | Add `2^24` exactly | Preserve scale `2^-24` |
| Factor | Convert one-plus-tanh | INT8, shift 47 |
| Final product | Centered half branch × centered factor | INT32, then two-round INT8 conversion |

Standard conversion multipliers approximate the source/destination scale ratio.
Products, rounding additions, and stored results must fit their intended types.

### Tanh table

For each INT8 argument code `q`:

$$
L[q+128]=\mathrm{round}_{even} \left(\tanh((q-z_a)s_a)2^{24}\right)
$$

Each block has a 256-entry INT32 table based on its own argument quantization.

### Final two-round conversion

Approximate the final product scale ratio as:

$$
\frac{s_{\mathrm{half}}s_{\mathrm{factor}}}{s_{\mathrm{GELU}}} \approx M_{31}2^{-31}2^{-r}
$$

Runtime:

1. Rounded Q31 multiplication, ties toward positive infinity.
2. Rounded division by `2^r`, ties away from zero.
3. Add the output zero point and saturate to INT8.

Use the existing XREQUANT mode 4.

### Validation

Across eight blocks and eight cases per block:

- Local expansion: 1,048,576 outputs, zero mismatches.
- All five local GELU front stages match TFLite exactly.
- Local factor and final GELU outputs match TFLite exactly.
- Chained final GELU agreement: 76.581%–95.123%.
- Chained maximum difference: 18 codes in block 1 calibration.
- Final saturation counts are identical between local and chained paths.

Chained results include preceding integer stages within each block.
Complete multi-block integer inference validation remains pending.

## Part 7 — MLP output, final residual, and INT8 block output

### Inputs and output

- GELU input to this part: INT8 `[1,32,512]`.
- Retained first residual: INT32 `[1,32,128]`, scale `2^-16`, zero point `0`.
- Projection weights: INT8 `[128,512]`, zero point `0`.
- Projection bias: saved INT32 `[128]`, at accumulator scale.
- Internal final residual: INT32 `[1,32,128]`, scale `2^-16`.
- Block output: INT8 `[1,32,128]`, with the selected boundary scale and zero point `0`.

### 1. Project 512 → 128

$$
A[t,o] = \sum_{k=0}^{511}(q_G[t,k]-z_G)q_W[o,k]+b_q[o]
$$

The accumulator scale is:

$$
s_A=s_Gs_W
$$

XTILE computes the dot products; software adds bias.
Products, partial sums, and bias addition must fit INT32.

### 2. Preserve the saved INT8 projection output

Prepare offline:

$$
M_P=\mathrm{round}_{even} \left(\frac{s_A}{s_P}2^{39}\right)
$$

**Requantization 1 — after accumulation and bias:**

$$
q_P=\mathrm{clip}_{[-128,127]} \left(RQ(A,M_P,39)+z_P\right)
$$

Use the block’s saved projection-output scale and zero point.
The C implementation uses XREQUANT mode 2.

### 3. Convert the projection branch to scale 2^-16

Prepare offline:

$$
M_R=\mathrm{round}_{even}(s_P2^{16}2^{22})
$$

**Requantization 2 — before residual addition:**

$$
q_B=RQ(q_P-z_P,M_R,22)
$$

Store `q_B` as INT32 at scale `2^-16`, zero point `0`.
No INT8 saturation is applied here.
The C implementation uses XREQUANT mode 3.

### 4. Add the retained first residual

$$
q_Y=q_R+q_B
$$

Both branches use scale `2^-16`.
Check the sum before INT32 storage.

No rounding is required for this addition.

### 5. Produce the INT8 block output

For selected boundary scale `2^-F_out`:

**Requantization 3 — after the final residual addition:**

$$
q_{\mathrm{block,out}} = \mathrm{clip}_{[-128,127]} \left(RQ(q_Y,1,16-F_{\mathrm{out}})\right)
$$

| Blocks | Output scale | Rounded right shift |
|---|---|---:|
| 0, 7 | `2^-7` | 9 |
| 1–6 | `2^-8` | 8 |

The C implementation can use XREQUANT mode 2 with multiplier `1`.

The next block consumes these same INT8 codes using this output scale.
No second conversion is required at the connection.

### Arithmetic rules

$$
RQ(a,M,r)=(aM+2^{r-1})\gg r
$$

- Runtime rounding: nearest, ties toward positive infinity.
- Offline multiplier preparation: nearest, ties to even.
- Multiplier products and rounding additions use checked INT64.
- Stored accumulators, converted branches, and residual sums must fit INT32.

### Validation

Local:
reference GELU output and independently encoded reference first residual → Part 7.

Chained:
integer Parts 1–6 and the retained integer Part 4 residual → Part 7.

Current eight-case results across all blocks:

- Local projection: 262,144 outputs, zero mismatches.
- Local pre-boundary RMSE: approximately 6.2e-6.
- Local boundary agreement: 99.683%–99.937%.
- Within-block chained pre-boundary relative L2: 0.765%–2.381%.
- Chained boundary agreement: 77.808%–95.776%.
- Chained maximum boundary difference: five INT8 codes.
- No final-boundary clipping occurred.
- Checked arithmetic completed without overflow.

Complete multi-block integer inference validation remains pending.

