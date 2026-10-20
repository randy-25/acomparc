# Part 6: MLP Expansion and GELU — Disassembly Analysis

## 1. Function Overview

This note analyzes `block0_mlp_gelu_forward()` and every helper it actually executes in the **supplied compiled ELF**. Case 0 provides the detailed instruction mix; Section 5.4 reconciles all eight cases with the supplied Spike `minstret` results. Counts describe dynamic executions, not the number of lines in the disassembly. See [counting method](analysis_method.md).

| Parameter | Current block-0 value |
|---|---|
| projection T / K / O | 32 / 128 / 512 |
| bias / input zero point | 512 INT32 elements / 2 |
| expansion output elements | 16,384 |
| T89 multiplier / shift / zero point | 1,274,254,450 / 39 / 37 |
| GELU add chunks | 4 × 4,096 elements |
| tanh LUT | 256 INT32 entries, scale 2^-24 |
| final conversion | Q31 multiplier 1,195,729,761; second shift 6; output zero point -88 |


## 2. Block Wrapper: `block0_mlp_gelu_forward()`

### 2.1 Source and operation order

```c
void block0_mlp_gelu_forward(
    const int8_t *input,
    int8_t *output,
    Block0MlpGeluWorkspace *workspace
)
{
    const size_t count =
        BLOCK0_TOKEN_COUNT * BLOCK0_MLP_EXPAND_OUTPUT_FEATURES;

    /* Op 58: expansion, including its INT8 T89 boundary. */
    block0_mlp_expand_forward(
        input,
        workspace->t89,
        workspace->product_i32
    );

    /* Op 59-60: half-input branch -> T92. */
    gelu_multiply_scalar_i8(
        workspace->t89, workspace->product_i32, count,
        BLOCK0_GELU_HALF_INPUT_ZERO_POINT,
        BLOCK0_GELU_HALF_CENTERED_CONSTANT
    );

    requantize_i32_to_i8(
        workspace->product_i32, workspace->half, count,
        BLOCK0_GELU_HALF_MULTIPLIER,
        BLOCK0_GELU_HALF_SHIFT,
        BLOCK0_GELU_HALF_OUTPUT_ZERO_POINT
    );

    /* Op 61-62: signed cube -> T94. */
    gelu_cube_i8(
        workspace->t89, workspace->product_i32, count,
        BLOCK0_GELU_CUBE_INPUT_ZERO_POINT
    );

    requantize_i32_to_i8(
        workspace->product_i32, workspace->branch, count,
        BLOCK0_GELU_CUBE_MULTIPLIER,
        BLOCK0_GELU_CUBE_SHIFT,
        BLOCK0_GELU_CUBE_OUTPUT_ZERO_POINT
    );

    /* Op 63: weighted cubic term -> T96. */
    gelu_multiply_scalar_i8(
        workspace->branch, workspace->product_i32, count,
        BLOCK0_GELU_WEIGHTED_CUBE_INPUT_ZERO_POINT,
        BLOCK0_GELU_WEIGHTED_CUBE_CENTERED_CONSTANT
    );

    requantize_i32_to_i8(
        workspace->product_i32, workspace->branch, count,
        BLOCK0_GELU_WEIGHTED_CUBE_MULTIPLIER,
        BLOCK0_GELU_WEIGHTED_CUBE_SHIFT,
        BLOCK0_GELU_WEIGHTED_CUBE_OUTPUT_ZERO_POINT
    );

    /* Op 64: combine branches, rounding once, in chunks -> T97. */
    for (size_t offset = 0; offset < count;
         offset += BLOCK0_GELU_ADD_CHUNK_ELEMENTS) {
        size_t chunk = count - offset;

        if (chunk > BLOCK0_GELU_ADD_CHUNK_ELEMENTS) {
            chunk = BLOCK0_GELU_ADD_CHUNK_ELEMENTS;
        }

        gelu_add_scaled_i8(
            workspace->t89 + offset,
            workspace->branch + offset,
            workspace->wide_i64,
            chunk,
            BLOCK0_GELU_ADD_LINEAR_ZERO_POINT,
            BLOCK0_GELU_ADD_CUBIC_ZERO_POINT,
            BLOCK0_GELU_ADD_LINEAR_MULTIPLIER,
            BLOCK0_GELU_ADD_CUBIC_MULTIPLIER
        );

        requantize_i64_to_i8(
            workspace->wide_i64,
            workspace->branch + offset,
            chunk,
            INT64_C(1),
            BLOCK0_GELU_ADD_SHIFT,
            BLOCK0_GELU_ADD_OUTPUT_ZERO_POINT
        );
    }

    /* Op 65: quantized tanh argument -> T99. */
    gelu_multiply_scalar_i8(
        workspace->branch, workspace->product_i32, count,
        BLOCK0_GELU_ARGUMENT_INPUT_ZERO_POINT,
        BLOCK0_GELU_ARGUMENT_CENTERED_CONSTANT
    );

    requantize_i32_to_i8(
        workspace->product_i32, workspace->branch, count,
        BLOCK0_GELU_ARGUMENT_MULTIPLIER,
        BLOCK0_GELU_ARGUMENT_SHIFT,
        BLOCK0_GELU_ARGUMENT_OUTPUT_ZERO_POINT
    );

    /* Op 66-67: tanh lookup -> T101. */
    gelu_tanh_lut_i8(
        workspace->branch,
        workspace->product_i32,
        count,
        block0_gelu_tanh_lut
    );

    /* Op 68: add integer one -> T102. */
    gelu_add_scalar_i32(
        workspace->product_i32,
        workspace->product_i32,
        count,
        BLOCK0_GELU_TANH_ONE
    );

    /* Op 69: quantized one-plus-tanh factor -> T103. */
    requantize_i32_to_i8(
        workspace->product_i32, workspace->branch, count,
        BLOCK0_GELU_FACTOR_MULTIPLIER,
        BLOCK0_GELU_FACTOR_SHIFT,
        BLOCK0_GELU_FACTOR_OUTPUT_ZERO_POINT
    );

    /* Op 70: centered T92 * centered T103. */
    gelu_multiply_i8(
        workspace->half,
        workspace->branch,
        workspace->product_i32,
        count,
        BLOCK0_GELU_OUTPUT_HALF_ZERO_POINT,
        BLOCK0_GELU_OUTPUT_FACTOR_ZERO_POINT
    );

    /* Selected two-stage rounding -> T104. */
    requantize_two_round_i32_to_i8(
        workspace->product_i32,
        output,
        count,
        BLOCK0_GELU_OUTPUT_Q31_MULTIPLIER,
        BLOCK0_GELU_OUTPUT_SECOND_SHIFT,
        BLOCK0_GELU_OUTPUT_ZERO_POINT
    );
}
```


### 2.2 Compiled wrapper

The assembly below includes the wrapper only. Each `jal` or tail `j` transfers execution to a helper whose work is counted separately. `executed` is the case-0 execution frequency of that instruction. A helper reused several times has aggregated frequencies in its own listing. Unexecuted alternate paths are shown with zero.

```asm
0000000080000f82 <block0_mlp_gelu_forward>:
80000f82:	7179                	addi	sp,sp,-48                                                        # executed: 1
80000f84:	ec26                	sd	s1,24(sp)                                                          # executed: 1
80000f86:	6e04                	ld	s1,24(a2)                                                          # executed: 1
80000f88:	f022                	sd	s0,32(sp)                                                          # executed: 1
80000f8a:	8432                	mv	s0,a2                                                              # executed: 1
80000f8c:	4889                	li	a7,2                                                               # executed: 1
80000f8e:	20000813          	li	a6,512                                                               # executed: 1
80000f92:	86a6                	mv	a3,s1                                                              # executed: 1
80000f94:	e84a                	sd	s2,16(sp)                                                          # executed: 1
80000f96:	e44e                	sd	s3,8(sp)                                                           # executed: 1
80000f98:	00043903          	ld	s2,0(s0)                                                             # executed: 1
80000f9c:	08000793          	li	a5,128                                                               # executed: 1
80000fa0:	02000713          	li	a4,32                                                                # executed: 1
80000fa4:	00011617          	auipc	a2,0x11                                                           # executed: 1
80000fa8:	5ac60613          	addi	a2,a2,1452 # 80012550 <block0_mlp_expand_bias>                     # executed: 1
80000fac:	89ae                	mv	s3,a1                                                              # executed: 1
80000fae:	00012597          	auipc	a1,0x12                                                           # executed: 1
80000fb2:	da258593          	addi	a1,a1,-606 # 80012d50 <block0_mlp_expand_weights>                  # executed: 1
80000fb6:	f406                	sd	ra,40(sp)                                                          # executed: 1
80000fb8:	b25ff0ef          	jal	80000adc <projection_accumulate_i8>                                 # executed: 1
80000fbc:	4bf396b7          	lui	a3,0x4bf39                                                          # executed: 1
80000fc0:	85ca                	mv	a1,s2                                                              # executed: 1
80000fc2:	8526                	mv	a0,s1                                                              # executed: 1
80000fc4:	02500793          	li	a5,37                                                                # executed: 1
80000fc8:	47268693          	addi	a3,a3,1138 # 4bf39472 <_start-0x340c6b8e>                          # executed: 1
80000fcc:	02700713          	li	a4,39                                                                # executed: 1
80000fd0:	6611                	lui	a2,0x4                                                            # executed: 1
80000fd2:	a35ff0ef          	jal	80000a06 <requantize_i32_to_i8>                                     # executed: 1
80000fd6:	6c0c                	ld	a1,24(s0)                                                          # executed: 1
80000fd8:	6008                	ld	a0,0(s0)                                                           # executed: 1
80000fda:	0ff00713          	li	a4,255                                                               # executed: 1
80000fde:	02500693          	li	a3,37                                                                # executed: 1
80000fe2:	6611                	lui	a2,0x4                                                            # executed: 1
80000fe4:	dd0ff0ef          	jal	800005b4 <gelu_multiply_scalar_i8>                                  # executed: 1
80000fe8:	640c                	ld	a1,8(s0)                                                           # executed: 1
80000fea:	6c08                	ld	a0,24(s0)                                                          # executed: 1
80000fec:	404046b7          	lui	a3,0x40404                                                          # executed: 1
80000ff0:	02500793          	li	a5,37                                                                # executed: 1
80000ff4:	02600713          	li	a4,38                                                                # executed: 1
80000ff8:	08068693          	addi	a3,a3,128 # 40404080 <_start-0x3fbfbf80>                           # executed: 1
80000ffc:	6611                	lui	a2,0x4                                                            # executed: 1
80000ffe:	a09ff0ef          	jal	80000a06 <requantize_i32_to_i8>                                     # executed: 1
80001002:	6c0c                	ld	a1,24(s0)                                                          # executed: 1
80001004:	6008                	ld	a0,0(s0)                                                           # executed: 1
80001006:	02500693          	li	a3,37                                                                # executed: 1
8000100a:	6611                	lui	a2,0x4                                                            # executed: 1
8000100c:	dc6ff0ef          	jal	800005d2 <gelu_cube_i8>                                             # executed: 1
80001010:	680c                	ld	a1,16(s0)                                                          # executed: 1
80001012:	6c08                	ld	a0,24(s0)                                                          # executed: 1
80001014:	6398e6b7          	lui	a3,0x6398e                                                          # executed: 1
80001018:	05500793          	li	a5,85                                                                # executed: 1
8000101c:	c9268693          	addi	a3,a3,-878 # 6398dc92 <_start-0x1c67236e>                          # executed: 1
80001020:	02d00713          	li	a4,45                                                                # executed: 1
80001024:	6611                	lui	a2,0x4                                                            # executed: 1
80001026:	9e1ff0ef          	jal	80000a06 <requantize_i32_to_i8>                                     # executed: 1
8000102a:	6c0c                	ld	a1,24(s0)                                                          # executed: 1
8000102c:	6808                	ld	a0,16(s0)                                                          # executed: 1
8000102e:	0b700713          	li	a4,183                                                               # executed: 1
80001032:	05500693          	li	a3,85                                                                # executed: 1
80001036:	6611                	lui	a2,0x4                                                            # executed: 1
80001038:	d7cff0ef          	jal	800005b4 <gelu_multiply_scalar_i8>                                  # executed: 1
8000103c:	680c                	ld	a1,16(s0)                                                          # executed: 1
8000103e:	6c08                	ld	a0,24(s0)                                                          # executed: 1
80001040:	597496b7          	lui	a3,0x59749                                                          # executed: 1
80001044:	7bb68693          	addi	a3,a3,1979 # 597497bb <_start-0x268b6845>                          # executed: 1
80001048:	05500793          	li	a5,85                                                                # executed: 1
8000104c:	02600713          	li	a4,38                                                                # executed: 1
80001050:	6611                	lui	a2,0x4                                                            # executed: 1
80001052:	9b5ff0ef          	jal	80000a06 <requantize_i32_to_i8>                                     # executed: 1
80001056:	4481                	li	s1,0                                                               # executed: 1
80001058:	6905                	lui	s2,0x1                                                            # executed: 1
8000105a:	680c                	ld	a1,16(s0)                                                          # executed: 4
8000105c:	6008                	ld	a0,0(s0)                                                           # executed: 4
8000105e:	7010                	ld	a2,32(s0)                                                          # executed: 4
80001060:	0d9e28b7          	lui	a7,0xd9e2                                                           # executed: 4
80001064:	7239d837          	lui	a6,0x7239d                                                          # executed: 4
80001068:	95a6                	add	a1,a1,s1                                                          # executed: 4
8000106a:	9526                	add	a0,a0,s1                                                          # executed: 4
8000106c:	47088893          	addi	a7,a7,1136 # d9e2470 <_start-0x7261db90>                           # executed: 4
80001070:	c2d80813          	addi	a6,a6,-979 # 7239cc2d <_start-0xdc633d3>                           # executed: 4
80001074:	05500793          	li	a5,85                                                                # executed: 4
80001078:	02500713          	li	a4,37                                                                # executed: 4
8000107c:	6685                	lui	a3,0x1                                                            # executed: 4
8000107e:	d76ff0ef          	jal	800005f4 <gelu_add_scaled_i8>                                       # executed: 4
80001082:	680c                	ld	a1,16(s0)                                                          # executed: 4
80001084:	7008                	ld	a0,32(s0)                                                          # executed: 4
80001086:	02500793          	li	a5,37                                                                # executed: 4
8000108a:	95a6                	add	a1,a1,s1                                                          # executed: 4
8000108c:	477d                	li	a4,31                                                              # executed: 4
8000108e:	4685                	li	a3,1                                                               # executed: 4
80001090:	6605                	lui	a2,0x1                                                            # executed: 4
80001092:	92fff0ef          	jal	800009c0 <requantize_i64_to_i8>                                     # executed: 4
80001096:	94ca                	add	s1,s1,s2                                                          # executed: 4
80001098:	6791                	lui	a5,0x4                                                            # executed: 4
8000109a:	fcf490e3          	bne	s1,a5,8000105a <block0_mlp_gelu_forward+0xd8>                       # executed: 4
8000109e:	6c0c                	ld	a1,24(s0)                                                          # executed: 1
800010a0:	6808                	ld	a0,16(s0)                                                          # executed: 1
800010a2:	8626                	mv	a2,s1                                                              # executed: 1
800010a4:	0ff00713          	li	a4,255                                                               # executed: 1
800010a8:	02500693          	li	a3,37                                                                # executed: 1
800010ac:	d08ff0ef          	jal	800005b4 <gelu_multiply_scalar_i8>                                  # executed: 1
800010b0:	680c                	ld	a1,16(s0)                                                          # executed: 1
800010b2:	6c08                	ld	a0,24(s0)                                                          # executed: 1
800010b4:	404046b7          	lui	a3,0x40404                                                          # executed: 1
800010b8:	02500793          	li	a5,37                                                                # executed: 1
800010bc:	02600713          	li	a4,38                                                                # executed: 1
800010c0:	8626                	mv	a2,s1                                                              # executed: 1
800010c2:	03068693          	addi	a3,a3,48 # 40404030 <_start-0x3fbfbfd0>                            # executed: 1
800010c6:	941ff0ef          	jal	80000a06 <requantize_i32_to_i8>                                     # executed: 1
800010ca:	6c0c                	ld	a1,24(s0)                                                          # executed: 1
800010cc:	6808                	ld	a0,16(s0)                                                          # executed: 1
800010ce:	8626                	mv	a2,s1                                                              # executed: 1
800010d0:	00011697          	auipc	a3,0x11                                                           # executed: 1
800010d4:	08068693          	addi	a3,a3,128 # 80012150 <block0_gelu_tanh_lut>                        # executed: 1
800010d8:	d4aff0ef          	jal	80000622 <gelu_tanh_lut_i8>                                         # executed: 1
800010dc:	6c0c                	ld	a1,24(s0)                                                          # executed: 1
800010de:	8626                	mv	a2,s1                                                              # executed: 1
800010e0:	010006b7          	lui	a3,0x1000                                                           # executed: 1
800010e4:	852e                	mv	a0,a1                                                              # executed: 1
800010e6:	d5eff0ef          	jal	80000644 <gelu_add_scalar_i32>                                      # executed: 1
800010ea:	680c                	ld	a1,16(s0)                                                          # executed: 1
800010ec:	6c08                	ld	a0,24(s0)                                                          # executed: 1
800010ee:	4baae6b7          	lui	a3,0x4baae                                                          # executed: 1
800010f2:	8626                	mv	a2,s1                                                              # executed: 1
800010f4:	a6568693          	addi	a3,a3,-1435 # 4baada65 <_start-0x3455259b>                         # executed: 1
800010f8:	f8000793          	li	a5,-128                                                              # executed: 1
800010fc:	02f00713          	li	a4,47                                                                # executed: 1
80001100:	907ff0ef          	jal	80000a06 <requantize_i32_to_i8>                                     # executed: 1
80001104:	6c10                	ld	a2,24(s0)                                                          # executed: 1
80001106:	680c                	ld	a1,16(s0)                                                          # executed: 1
80001108:	6408                	ld	a0,8(s0)                                                           # executed: 1
8000110a:	86a6                	mv	a3,s1                                                              # executed: 1
8000110c:	f8000793          	li	a5,-128                                                              # executed: 1
80001110:	02500713          	li	a4,37                                                                # executed: 1
80001114:	d48ff0ef          	jal	8000065c <gelu_multiply_i8>                                         # executed: 1
80001118:	6c08                	ld	a0,24(s0)                                                          # executed: 1
8000111a:	7402                	ld	s0,32(sp)                                                          # executed: 1
8000111c:	70a2                	ld	ra,40(sp)                                                          # executed: 1
8000111e:	6942                	ld	s2,16(sp)                                                          # executed: 1
80001120:	8626                	mv	a2,s1                                                              # executed: 1
80001122:	85ce                	mv	a1,s3                                                              # executed: 1
80001124:	64e2                	ld	s1,24(sp)                                                          # executed: 1
80001126:	69a2                	ld	s3,8(sp)                                                           # executed: 1
80001128:	474566b7          	lui	a3,0x47456                                                          # executed: 1
8000112c:	36168693          	addi	a3,a3,865 # 47456361 <_start-0x38ba9c9f>                           # executed: 1
80001130:	fa800793          	li	a5,-88                                                               # executed: 1
80001134:	4719                	li	a4,6                                                               # executed: 1
80001136:	6145                	addi	sp,sp,48                                                         # executed: 1
80001138:	b291                	j	80000a7c <requantize_two_round_i32_to_i8>                           # executed: 1
```


The wrapper itself executes **221 instructions**, excluding all callee bodies. Stack saves retain registers and pointers across calls; the final helper may be tail-called after restoration. A tail call is counted as one wrapper jump followed by the helper's body and return, not a second wrapper epilogue.

`block0_mlp_expand_forward()` is present as a separate symbol, but it is **inlined in this wrapper**. The compiled wrapper calls projection accumulation and T89 conversion directly. The four-iteration GELU chunk loop makes some wrapper instructions execute four times; counting assembly lines once would undercount it.

#### Incoming registers

| Register on entry | C argument |
|---|---|
| `a0` | const int8_t *input |
| `a1` | int8_t *output |
| `a2` | Block0MlpGeluWorkspace *workspace |

#### Retained values and stack frame

| Register | Role in this wrapper |
|---|---|
| `s0` | Block0MlpGeluWorkspace pointer throughout the function |
| `s1` | Initially product_i32 pointer; later GELU chunk offset (0, 4096, 8192, 12288) |
| `s2` | Initially t89 pointer; later chunk increment 4096 |
| `s3` | Final INT8 T104 output pointer |

All six wrappers analyzed in the new notes allocate 48 bytes. They save `ra` at `40(sp)`, `s0` at `32(sp)`, `s1` at `24(sp)`, `s2` at `16(sp)`, and `s3` at `8(sp)`: 40 bytes of saved values plus eight bytes unused in this frame, keeping the 16-byte stack alignment. These saved slots hold register values, not activation tensors. The callee-saved registers retain values across helper calls, and the wrapper restores the caller's original values before its final tail transfer.

MLP workspace pointer fields are at offsets 0 (`t89`), 8 (`half`), 16 (`branch`), 24 (`product_i32`), and 32 (`wide_i64`). The compiler reuses `s1` and `s2` after expansion, so their meanings must be followed in program order. The chunk loop at `0x8000105a–0x8000109a` executes four times.

### 2.3 Helper arguments and data boundaries

Pointer arguments select the source tensors and caller-owned scratch shown in the C wrapper. Non-pointer arguments decoded at each actual call are:

| Call in source order | Actual non-pointer arguments |
|---|---|
| expansion accumulation | T=32, K=128, O=512, input z=2; bias present |
| T89 conversion | N=16384, M=1,274,254,450, shift=39, output z=37 |
| half-input multiply | N=16384, input z=37, centered scalar=255 |
| T92 conversion | N=16384, M=1,077,952,640, shift=38, output z=37 |
| cube | N=16384, input z=37 |
| T94 conversion | N=16384, M=1,670,962,322, shift=45, output z=85 |
| weighted-cube multiply | N=16384, input z=85, centered scalar=183 |
| T96 conversion | N=16384, M=1,500,813,243, shift=38, output z=85 |
| chunk 0: scaled add | n=4096, linear z=37, cubic z=85, multipliers=1,916,390,445 and 228,467,824 |
| chunk 0: T97 conversion | N=4096, M=1, shift=31, output z=37 |
| chunk 1: scaled add | n=4096, linear z=37, cubic z=85, multipliers=1,916,390,445 and 228,467,824 |
| chunk 1: T97 conversion | N=4096, M=1, shift=31, output z=37 |
| chunk 2: scaled add | n=4096, linear z=37, cubic z=85, multipliers=1,916,390,445 and 228,467,824 |
| chunk 2: T97 conversion | N=4096, M=1, shift=31, output z=37 |
| chunk 3: scaled add | n=4096, linear z=37, cubic z=85, multipliers=1,916,390,445 and 228,467,824 |
| chunk 3: T97 conversion | N=4096, M=1, shift=31, output z=37 |
| tanh-argument multiply | N=16384, input z=37, centered scalar=255 |
| T99 conversion | N=16384, M=1,077,952,560, shift=38, output z=37 |
| tanh LUT | N=16384, direct signed-code index +128 |
| one-plus-tanh add | N=16384, scalar=16,777,216 |
| T103 conversion | N=16384, M=1,269,488,229, shift=47, output z=-128 |
| half × factor multiply | N=16384, left z=37, right z=-128 |
| T104 two-round conversion | N=16384, Q31 M=1,195,729,761, second shift=6, output z=-88 |

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
80000b10:	c7a5                	beqz	a5,80000b78 <projection_accumulate_i8+0x9c>                      # executed: 16,384
80000b12:	00758e33          	add	t3,a1,t2                                                            # executed: 16,384
80000b16:	853a                	mv	a0,a4                                                              # executed: 16,384
80000b18:	4e81                	li	t4,0                                                               # executed: 16,384
80000b1a:	00050303          	lb	t1,0(a0)                                                             # executed: 2,097,152
80000b1e:	000e0803          	lb	a6,0(t3)                                                             # executed: 2,097,152
80000b22:	0505                	addi	a0,a0,1                                                          # executed: 2,097,152
80000b24:	4113033b          	subw	t1,t1,a7                                                           # executed: 2,097,152
80000b28:	0268083b          	mulw	a6,a6,t1                                                           # executed: 2,097,152
80000b2c:	0e05                	addi	t3,t3,1                                                          # executed: 2,097,152
80000b2e:	01d80ebb          	addw	t4,a6,t4                                                           # executed: 2,097,152
80000b32:	ffe514e3          	bne	a0,t5,80000b1a <projection_accumulate_i8+0x3e>                      # executed: 2,097,152
80000b36:	c609                	beqz	a2,80000b40 <projection_accumulate_i8+0x64>                      # executed: 16,384
80000b38:	000fa503          	lw	a0,0(t6) # 40000000 <_start-0x40000000>                              # executed: 16,384
80000b3c:	01d50ebb          	addw	t4,a0,t4                                                           # executed: 16,384
80000b40:	01d2a023          	sw	t4,0(t0)                                                             # executed: 16,384
80000b44:	0f91                	addi	t6,t6,4                                                          # executed: 16,384
80000b46:	0291                	addi	t0,t0,4                                                          # executed: 16,384
80000b48:	93be                	add	t2,t2,a5                                                          # executed: 16,384
80000b4a:	fcdf93e3          	bne	t6,a3,80000b10 <projection_accumulate_i8+0x34>                      # executed: 16,384
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


### 3.2 `gelu_multiply_scalar_i8()` — Centered scalar multiplication

Compute $(x-z)c$ into INT32 storage using a 64-bit `mul`. The seven-instruction element loop is `0x800005ba–0x800005cc`: one load, two pointer updates, one subtraction, one multiplication, one store, one branch. The complete count is $7N+4=114{,}692$ for N = 16,384. This helper runs for the half-input, weighted-cube, and tanh-argument branches with different constants. Requantization is separate.

#### Corresponding C

```c
void gelu_multiply_scalar_i8(
    const int8_t *input,
    int32_t *output,
    size_t count,
    int32_t input_zero_point,
    int32_t centered_scalar
)
{
    for (size_t i = 0; i < count; ++i) {
        const int64_t centered =
            (int64_t)input[i] - input_zero_point;

        const int64_t product = centered * centered_scalar;

        output[i] = (int32_t)product;
    }
}
```


#### Compiled assembly

```asm
00000000800005b4 <gelu_multiply_scalar_i8>:
800005b4:	ce11                	beqz	a2,800005d0 <gelu_multiply_scalar_i8+0x1c>                       # executed: 3
800005b6:	060a                	slli	a2,a2,0x2                                                        # executed: 3
800005b8:	962e                	add	a2,a2,a1                                                          # executed: 3
800005ba:	00050783          	lb	a5,0(a0)                                                             # executed: 49,152
800005be:	0591                	addi	a1,a1,4                                                          # executed: 49,152
800005c0:	0505                	addi	a0,a0,1                                                          # executed: 49,152
800005c2:	8f95                	sub	a5,a5,a3                                                          # executed: 49,152
800005c4:	02e787b3          	mul	a5,a5,a4                                                            # executed: 49,152
800005c8:	fef5ae23          	sw	a5,-4(a1)                                                            # executed: 49,152
800005cc:	fec597e3          	bne	a1,a2,800005ba <gelu_multiply_scalar_i8+0x6>                        # executed: 49,152
800005d0:	8082                	ret                                                                   # executed: 3
```


### 3.3 `gelu_cube_i8()` — Signed cube

Compute $(x-37)^3$, using `subw` and two `mulw` per element, followed by a separate conversion. The eight-instruction loop at `0x800005d8–0x800005ee` gives $8N+4=131{,}076$. These 32,768 multiplications are elementwise cube arithmetic, not matrix MACs.

#### Corresponding C

```c
void gelu_cube_i8(
    const int8_t *input,
    int32_t *output,
    size_t count,
    int32_t input_zero_point
)
{
    for (size_t i = 0; i < count; ++i) {
        const int32_t centered =
            (int32_t)input[i] - input_zero_point;

        const int32_t square = centered * centered;

        output[i] = square * centered;
    }
}
```


#### Compiled assembly

```asm
00000000800005d2 <gelu_cube_i8>:
800005d2:	c205                	beqz	a2,800005f2 <gelu_cube_i8+0x20>                                  # executed: 1
800005d4:	060a                	slli	a2,a2,0x2                                                        # executed: 1
800005d6:	962e                	add	a2,a2,a1                                                          # executed: 1
800005d8:	00050783          	lb	a5,0(a0)                                                             # executed: 16,384
800005dc:	0591                	addi	a1,a1,4                                                          # executed: 16,384
800005de:	0505                	addi	a0,a0,1                                                          # executed: 16,384
800005e0:	9f95                	subw	a5,a5,a3                                                         # executed: 16,384
800005e2:	02f7873b          	mulw	a4,a5,a5                                                           # executed: 16,384
800005e6:	02f707bb          	mulw	a5,a4,a5                                                           # executed: 16,384
800005ea:	fef5ae23          	sw	a5,-4(a1)                                                            # executed: 16,384
800005ee:	fec595e3          	bne	a1,a2,800005d8 <gelu_cube_i8+0x6>                                   # executed: 16,384
800005f2:	8082                	ret                                                                   # executed: 1
```


### 3.4 `gelu_add_scaled_i8()` — Combine linear and cubic branches

Compute $(x-37)\cdot1{,}916{,}390{,}445+(c-85)\cdot228{,}467{,}824$ in INT64. The twelve-instruction element loop at `0x800005f8–0x8000061c` has two loads, two centering subtracts, two `mul`, one numerical add, three pointer updates, one INT64 store, and one branch. The count is $12n+3=49{,}155$ for each 4,096-element chunk. Four chunks cover all 16,384 elements. Each chunk is converted once after combining both products; there is no independent rounding of the two products.

#### Corresponding C

```c
void gelu_add_scaled_i8(
    const int8_t *linear,
    const int8_t *cubic,
    int64_t *output,
    size_t count,
    int32_t linear_zero_point,
    int32_t cubic_zero_point,
    int64_t linear_multiplier,
    int64_t cubic_multiplier
)
{
    for (size_t i = 0; i < count; ++i) {
        const int64_t centered_linear =
            (int64_t)linear[i] - linear_zero_point;

        const int64_t centered_cubic =
            (int64_t)cubic[i] - cubic_zero_point;

        const int64_t linear_product =
            centered_linear * linear_multiplier;

        const int64_t cubic_product =
            centered_cubic * cubic_multiplier;

        output[i] = linear_product + cubic_product;
    }
}
```


#### Compiled assembly

```asm
00000000800005f4 <gelu_add_scaled_i8>:
800005f4:	c695                	beqz	a3,80000620 <gelu_add_scaled_i8+0x2c>                            # executed: 4
800005f6:	96aa                	add	a3,a3,a0                                                          # executed: 4
800005f8:	00050303          	lb	t1,0(a0)                                                             # executed: 16,384
800005fc:	00058e03          	lb	t3,0(a1)                                                             # executed: 16,384
80000600:	0505                	addi	a0,a0,1                                                          # executed: 16,384
80000602:	40e30333          	sub	t1,t1,a4                                                            # executed: 16,384
80000606:	40fe0e33          	sub	t3,t3,a5                                                            # executed: 16,384
8000060a:	03030333          	mul	t1,t1,a6                                                            # executed: 16,384
8000060e:	0585                	addi	a1,a1,1                                                          # executed: 16,384
80000610:	0621                	addi	a2,a2,8                                                          # executed: 16,384
80000612:	031e0e33          	mul	t3,t3,a7                                                            # executed: 16,384
80000616:	9372                	add	t1,t1,t3                                                          # executed: 16,384
80000618:	fe663c23          	sd	t1,-8(a2)                                                            # executed: 16,384
8000061c:	fca69ee3          	bne	a3,a0,800005f8 <gelu_add_scaled_i8+0x4>                             # executed: 16,384
80000620:	8082                	ret                                                                   # executed: 4
```


### 3.5 `gelu_tanh_lut_i8()` — Direct tanh lookup

Map signed INT8 T99 to the 256-entry table using `index = input + 128`. The `addiw` at `0x80000630`, `slli`, address `add`, and `lw` implement the lookup. Table values are INT32 at scale $2^{-24}$. The nine-instruction loop gives $9N+4=147{,}460$. No runtime tanh calculation or interpolation is executed. The LUT is block-specific in the current implementation.

#### Corresponding C

```c
void gelu_tanh_lut_i8(
    const int8_t *input,
    int32_t *output,
    size_t count,
    const int32_t *table
)
{
    for (size_t i = 0; i < count; ++i) {
        const size_t index =
            (size_t)((int32_t)input[i] + 128);

        output[i] = table[index];
    }
}
```


#### Compiled assembly

```asm
0000000080000622 <gelu_tanh_lut_i8>:
80000622:	c205                	beqz	a2,80000642 <gelu_tanh_lut_i8+0x20>                              # executed: 1
80000624:	060a                	slli	a2,a2,0x2                                                        # executed: 1
80000626:	962e                	add	a2,a2,a1                                                          # executed: 1
80000628:	00050783          	lb	a5,0(a0)                                                             # executed: 16,384
8000062c:	0591                	addi	a1,a1,4                                                          # executed: 16,384
8000062e:	0505                	addi	a0,a0,1                                                          # executed: 16,384
80000630:	0807879b          	addiw	a5,a5,128                                                         # executed: 16,384
80000634:	078a                	slli	a5,a5,0x2                                                        # executed: 16,384
80000636:	97b6                	add	a5,a5,a3                                                          # executed: 16,384
80000638:	439c                	lw	a5,0(a5)                                                           # executed: 16,384
8000063a:	fef5ae23          	sw	a5,-4(a1)                                                            # executed: 16,384
8000063e:	feb615e3          	bne	a2,a1,80000628 <gelu_tanh_lut_i8+0x6>                               # executed: 16,384
80000642:	8082                	ret                                                                   # executed: 1
```


### 3.6 `gelu_add_scalar_i32()` — Add one at the tanh scale

Add $2^{24}=16{,}777{,}216$ to every INT32 tanh value. The six-instruction loop gives $6N+4=98{,}308$. Its `addw` is an elementwise numerical add, not an accumulator MAC.

#### Corresponding C

```c
void gelu_add_scalar_i32(
    const int32_t *input,
    int32_t *output,
    size_t count,
    int32_t scalar
)
{
    for (size_t i = 0; i < count; ++i) {
        output[i] = (int32_t)((int64_t)input[i] + scalar);
    }
}
```


#### Compiled assembly

```asm
0000000080000644 <gelu_add_scalar_i32>:
80000644:	ca19                	beqz	a2,8000065a <gelu_add_scalar_i32+0x16>                           # executed: 1
80000646:	060a                	slli	a2,a2,0x2                                                        # executed: 1
80000648:	962a                	add	a2,a2,a0                                                          # executed: 1
8000064a:	411c                	lw	a5,0(a0)                                                           # executed: 16,384
8000064c:	0511                	addi	a0,a0,4                                                          # executed: 16,384
8000064e:	0591                	addi	a1,a1,4                                                          # executed: 16,384
80000650:	9fb5                	addw	a5,a5,a3                                                         # executed: 16,384
80000652:	fef5ae23          	sw	a5,-4(a1)                                                            # executed: 16,384
80000656:	fec51ae3          	bne	a0,a2,8000064a <gelu_add_scalar_i32+0x6>                            # executed: 16,384
8000065a:	8082                	ret                                                                   # executed: 1
```


### 3.7 `gelu_multiply_i8()` — Multiply the half-input and one-plus-tanh factors

Center the INT8 half-input with zero point 37 and factor with zero point -128, multiply, and store INT32 products. The ten-instruction loop at `0x80000660–0x8000067e` has two `lb`, two `subw`, one `mulw`, three pointer updates, one `sw`, and one `bne`. The total is $10N+3=163{,}843$. Op 70's two-stage conversion follows in a separate function.

#### Corresponding C

```c
void gelu_multiply_i8(
    const int8_t *left,
    const int8_t *right,
    int32_t *output,
    size_t count,
    int32_t left_zero_point,
    int32_t right_zero_point
)
{
    for (size_t i = 0; i < count; ++i) {
        const int32_t centered_left =
            (int32_t)left[i] - left_zero_point;

        const int32_t centered_right =
            (int32_t)right[i] - right_zero_point;

        output[i] = centered_left * centered_right;
    }
}
```


#### Compiled assembly

```asm
000000008000065c <gelu_multiply_i8>:
8000065c:	c29d                	beqz	a3,80000682 <gelu_multiply_i8+0x26>                              # executed: 1
8000065e:	96aa                	add	a3,a3,a0                                                          # executed: 1
80000660:	00050883          	lb	a7,0(a0)                                                             # executed: 16,384
80000664:	00058803          	lb	a6,0(a1)                                                             # executed: 16,384
80000668:	0505                	addi	a0,a0,1                                                          # executed: 16,384
8000066a:	40e888bb          	subw	a7,a7,a4                                                           # executed: 16,384
8000066e:	40f8083b          	subw	a6,a6,a5                                                           # executed: 16,384
80000672:	0318083b          	mulw	a6,a6,a7                                                           # executed: 16,384
80000676:	0585                	addi	a1,a1,1                                                          # executed: 16,384
80000678:	0611                	addi	a2,a2,4                                                          # executed: 16,384
8000067a:	ff062e23          	sw	a6,-4(a2)                                                            # executed: 16,384
8000067e:	fea691e3          	bne	a3,a0,80000660 <gelu_multiply_i8+0x4>                               # executed: 16,384
80000682:	8082                	ret                                                                   # executed: 1
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
80000a06:	c231                	beqz	a2,80000a4a <requantize_i32_to_i8+0x44>                          # executed: 6
80000a08:	060a                	slli	a2,a2,0x2                                                        # executed: 6
80000a0a:	fff7081b          	addiw	a6,a4,-1                                                          # executed: 6
80000a0e:	4885                	li	a7,1                                                               # executed: 6
80000a10:	010898b3          	sll	a7,a7,a6                                                            # executed: 6
80000a14:	962a                	add	a2,a2,a0                                                          # executed: 6
80000a16:	07f00e13          	li	t3,127                                                               # executed: 6
80000a1a:	f8000313          	li	t1,-128                                                              # executed: 6
80000a1e:	00052803          	lw	a6,0(a0)                                                             # executed: 98,304
80000a22:	0511                	addi	a0,a0,4                                                          # executed: 98,304
80000a24:	02d80833          	mul	a6,a6,a3                                                            # executed: 98,304
80000a28:	9846                	add	a6,a6,a7                                                          # executed: 98,304
80000a2a:	40e85833          	sra	a6,a6,a4                                                            # executed: 98,304
80000a2e:	983e                	add	a6,a6,a5                                                          # executed: 98,304
80000a30:	010e5463          	bge	t3,a6,80000a38 <requantize_i32_to_i8+0x32>                          # executed: 98,304
80000a34:	07f00813          	li	a6,127                                                               # executed: 3
80000a38:	00685463          	bge	a6,t1,80000a40 <requantize_i32_to_i8+0x3a>                          # executed: 98,304
80000a3c:	f8000813          	li	a6,-128                                                              # executed: 0
80000a40:	01058023          	sb	a6,0(a1)                                                             # executed: 98,304
80000a44:	0585                	addi	a1,a1,1                                                          # executed: 98,304
80000a46:	fca61ce3          	bne	a2,a0,80000a1e <requantize_i32_to_i8+0x18>                          # executed: 98,304
80000a4a:	8082                	ret                                                                   # executed: 6
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
800009c0:	c231                	beqz	a2,80000a04 <requantize_i64_to_i8+0x44>                          # executed: 4
800009c2:	060e                	slli	a2,a2,0x3                                                        # executed: 4
800009c4:	fff7081b          	addiw	a6,a4,-1                                                          # executed: 4
800009c8:	4885                	li	a7,1                                                               # executed: 4
800009ca:	010898b3          	sll	a7,a7,a6                                                            # executed: 4
800009ce:	962a                	add	a2,a2,a0                                                          # executed: 4
800009d0:	07f00e13          	li	t3,127                                                               # executed: 4
800009d4:	f8000313          	li	t1,-128                                                              # executed: 4
800009d8:	00053803          	ld	a6,0(a0)                                                             # executed: 16,384
800009dc:	0521                	addi	a0,a0,8                                                          # executed: 16,384
800009de:	03068833          	mul	a6,a3,a6                                                            # executed: 16,384
800009e2:	9846                	add	a6,a6,a7                                                          # executed: 16,384
800009e4:	40e85833          	sra	a6,a6,a4                                                            # executed: 16,384
800009e8:	983e                	add	a6,a6,a5                                                          # executed: 16,384
800009ea:	010e5463          	bge	t3,a6,800009f2 <requantize_i64_to_i8+0x32>                          # executed: 16,384
800009ee:	07f00813          	li	a6,127                                                               # executed: 0
800009f2:	00685463          	bge	a6,t1,800009fa <requantize_i64_to_i8+0x3a>                          # executed: 16,384
800009f6:	f8000813          	li	a6,-128                                                              # executed: 0
800009fa:	01058023          	sb	a6,0(a1)                                                             # executed: 16,384
800009fe:	0585                	addi	a1,a1,1                                                          # executed: 16,384
80000a00:	fca61ce3          	bne	a2,a0,800009d8 <requantize_i64_to_i8+0x18>                          # executed: 16,384
80000a04:	8082                	ret                                                                   # executed: 4
```


### 4.3 `requantize_two_round_i32_to_i8()` — Selected two-stage rounding for Op 70

First compute $h=(v\cdot1{,}195{,}729{,}761+2^{30})\gg31$. Then divide h by $2^6$ with the source's signed, ties-away-from-zero rule:
$$
m=63,\quad r=h\mathbin{\&}m,\quad \theta=(m\gg1)+[h<0],\quad u=(h\gg6)+[r>\theta].
$$
Finally add -88 and clamp to INT8. `srli` extracts the sign predicate, `and` extracts the remainder, and `slt` generates the correction. The seventeen-instruction ordinary element loop plus ten setup instructions and one return gives $17N+11+C=278{,}539+C$. Both optional saturation assignments execute zero times in all eight cases.

**Do not combine this into the ordinary single-round conversion.** The selected integer candidate requires both rounding stages, and a single-round replacement changes some outputs.

#### Corresponding C

```c
void requantize_two_round_i32_to_i8(
    const int32_t *input,
    int8_t *output,
    size_t count,
    int32_t q31_multiplier,
    unsigned int second_right_shift,
    int32_t output_zero_point
)
{
    const int64_t mask =
        (INT64_C(1) << second_right_shift) - 1;

    for (size_t i = 0; i < count; ++i) {
        /* Stage 1: rounded Q31 multiplication. */
        const int64_t product =
            (int64_t)input[i] * q31_multiplier;

        const int64_t high =
            (product + (INT64_C(1) << 30)) >> 31;

        /* Stage 2: rounded division, ties away from zero. */
        const int64_t remainder = high & mask;

        const int64_t threshold =
            (mask >> 1) + (high < 0);

        const int64_t rounded =
            (high >> second_right_shift)
            + (remainder > threshold);

        int64_t result = rounded + output_zero_point;

        if (result < -128) {
            result = -128;
        } else if (result > 127) {
            result = 127;
        }

        output[i] = (int8_t)result;
    }
}
```


#### Compiled assembly

```asm
0000000080000a7c <requantize_two_round_i32_to_i8>:
80000a7c:	ce39                	beqz	a2,80000ada <requantize_two_round_i32_to_i8+0x5e>                # executed: 1
80000a7e:	4e05                	li	t3,1                                                               # executed: 1
80000a80:	00ee1e33          	sll	t3,t3,a4                                                            # executed: 1
80000a84:	060a                	slli	a2,a2,0x2                                                        # executed: 1
80000a86:	1e7d                	addi	t3,t3,-1                                                         # executed: 1
80000a88:	962a                	add	a2,a2,a0                                                          # executed: 1
80000a8a:	401e5293          	srai	t0,t3,0x1                                                          # executed: 1
80000a8e:	40000fb7          	lui	t6,0x40000                                                          # executed: 1
80000a92:	07f00f13          	li	t5,127                                                               # executed: 1
80000a96:	f8000e93          	li	t4,-128                                                              # executed: 1
80000a9a:	00052803          	lw	a6,0(a0)                                                             # executed: 16,384
80000a9e:	0511                	addi	a0,a0,4                                                          # executed: 16,384
80000aa0:	02d80833          	mul	a6,a6,a3                                                            # executed: 16,384
80000aa4:	987e                	add	a6,a6,t6                                                          # executed: 16,384
80000aa6:	41f85813          	srai	a6,a6,0x1f                                                         # executed: 16,384
80000aaa:	03f85313          	srli	t1,a6,0x3f                                                         # executed: 16,384
80000aae:	010e78b3          	and	a7,t3,a6                                                            # executed: 16,384
80000ab2:	9316                	add	t1,t1,t0                                                          # executed: 16,384
80000ab4:	011328b3          	slt	a7,t1,a7                                                            # executed: 16,384
80000ab8:	40e85833          	sra	a6,a6,a4                                                            # executed: 16,384
80000abc:	9846                	add	a6,a6,a7                                                          # executed: 16,384
80000abe:	983e                	add	a6,a6,a5                                                          # executed: 16,384
80000ac0:	010f5463          	bge	t5,a6,80000ac8 <requantize_two_round_i32_to_i8+0x4c>                # executed: 16,384
80000ac4:	07f00813          	li	a6,127                                                               # executed: 0
80000ac8:	01d85463          	bge	a6,t4,80000ad0 <requantize_two_round_i32_to_i8+0x54>                # executed: 16,384
80000acc:	f8000813          	li	a6,-128                                                              # executed: 0
80000ad0:	01058023          	sb	a6,0(a1)                                                             # executed: 16,384
80000ad4:	0585                	addi	a1,a1,1                                                          # executed: 16,384
80000ad6:	fca612e3          	bne	a2,a0,80000a9a <requantize_two_round_i32_to_i8+0x1e>                # executed: 16,384
80000ada:	8082                	ret                                                                   # executed: 1
```


## 5. Complete Instruction Accounting

### 5.1 Loop and computation counts

The projection performs **2,097,152 mathematical MACs** and produces 16,384 outputs. Its biased formula is

$$
24+32\{9+512[12+8(128)]\}=16,974,136.
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


Outside the inner loop: 24 entry/exit instructions, $32\times9=288$ token instructions, and $16,384\times12=196,608$ output instructions, totaling **196,920**. Of those output instructions, exactly **32,768** are the bias load and bias addition. Thus the accumulation kernel's full count is **16,974,136**.

One mathematical MAC remains one multiplication plus one accumulator addition. Eight describes the complete scalar loop iteration. Its 75% supporting-instruction share does not by itself establish a memory-bandwidth bottleneck.

### 5.2 Requantization counts

| Conversion | Elements | Formula | C, case 0 | Executed instructions |
|---|---|---|---|---|
| T89 conversion | 16,384 | 11N + 9 + C | 1 | 180,234 |
| T92 conversion | 16,384 | 11N + 9 + C | 0 | 180,233 |
| T94 conversion | 16,384 | 11N + 9 + C | 0 | 180,233 |
| T96 conversion | 16,384 | 11N + 9 + C | 0 | 180,233 |
| chunk 0: T97 conversion | 4,096 | 11N + 9 + C | 0 | 45,065 |
| chunk 1: T97 conversion | 4,096 | 11N + 9 + C | 0 | 45,065 |
| chunk 2: T97 conversion | 4,096 | 11N + 9 + C | 0 | 45,065 |
| chunk 3: T97 conversion | 4,096 | 11N + 9 + C | 0 | 45,065 |
| T99 conversion | 16,384 | 11N + 9 + C | 0 | 180,233 |
| T103 conversion | 16,384 | 11N + 9 + C | 2 | 180,235 |
| T104 two-round conversion | 16,384 | 17N + 11 + C | 0 | 278,539 |
| **Total** |  |  |  | **1,540,200** |

Conversion kernels account for **7.86%** of this stage body in case 0. Computation kernels, conversion kernels, and wrapper instructions are counted separately.

### 5.3 Function totals and instruction breakdown

| Order | Work | Function | Case-0 instructions | Share of stage body |
|---|---|---|---|---|
| 0 | wrapper | `block0_mlp_gelu_forward` | 221 | 0.0011% |
| 1 | expansion accumulation | `projection_accumulate_i8` | 16,974,136 | 86.6207% |
| 2 | T89 conversion | `requantize_i32_to_i8` | 180,234 | 0.9198% |
| 3 | half-input multiply | `gelu_multiply_scalar_i8` | 114,692 | 0.5853% |
| 4 | T92 conversion | `requantize_i32_to_i8` | 180,233 | 0.9197% |
| 5 | cube | `gelu_cube_i8` | 131,076 | 0.6689% |
| 6 | T94 conversion | `requantize_i32_to_i8` | 180,233 | 0.9197% |
| 7 | weighted-cube multiply | `gelu_multiply_scalar_i8` | 114,692 | 0.5853% |
| 8 | T96 conversion | `requantize_i32_to_i8` | 180,233 | 0.9197% |
| 9 | chunk 0: scaled add | `gelu_add_scaled_i8` | 49,155 | 0.2508% |
| 10 | chunk 0: T97 conversion | `requantize_i64_to_i8` | 45,065 | 0.2300% |
| 11 | chunk 1: scaled add | `gelu_add_scaled_i8` | 49,155 | 0.2508% |
| 12 | chunk 1: T97 conversion | `requantize_i64_to_i8` | 45,065 | 0.2300% |
| 13 | chunk 2: scaled add | `gelu_add_scaled_i8` | 49,155 | 0.2508% |
| 14 | chunk 2: T97 conversion | `requantize_i64_to_i8` | 45,065 | 0.2300% |
| 15 | chunk 3: scaled add | `gelu_add_scaled_i8` | 49,155 | 0.2508% |
| 16 | chunk 3: T97 conversion | `requantize_i64_to_i8` | 45,065 | 0.2300% |
| 17 | tanh-argument multiply | `gelu_multiply_scalar_i8` | 114,692 | 0.5853% |
| 18 | T99 conversion | `requantize_i32_to_i8` | 180,233 | 0.9197% |
| 19 | tanh LUT | `gelu_tanh_lut_i8` | 147,460 | 0.7525% |
| 20 | one-plus-tanh add | `gelu_add_scalar_i32` | 98,308 | 0.5017% |
| 21 | T103 conversion | `requantize_i32_to_i8` | 180,235 | 0.9198% |
| 22 | half × factor multiply | `gelu_multiply_i8` | 163,843 | 0.8361% |
| 23 | T104 two-round conversion | `requantize_two_round_i32_to_i8` | 278,539 | 1.4214% |
|  | **Total, including callees** |  | **19,595,940** | **100%** |

Across this stage, multiplication instructions (`mul` + `mulw`) total **2,359,296**. Register additions (`add` + `addw`) total **2,490,536**. These opcode totals include address/setup arithmetic where the compiler uses those mnemonics; they are not counts of mathematical MACs. Immediate additions (`addi` + `addiw`) are listed separately below.

| Disassembly mnemonic | Executed, case 0 | Function in this stage |
|---|---|---|
| `beqz` | 32,792 | Zero-size checks or zero-bit tests. |
| `addi` | 4,784,183 | Add an immediate: pointer/loop/stack updates, plus immediate numerical constants. |
| `sd` | 16,394 | Store INT64 intermediates or save registers. |
| `mv` | 16,434 | Copy a pointer/value or prepare call arguments; also select attention maxima. |
| `mul` | 212,992 | 64-bit low-product arithmetic: requantization, scaling, normalization, and occasional dimension setup. |
| `add` | 360,616 | 64-bit addition: numerical arithmetic and address/index calculations. |
| `slli` | 16,434 | Immediate left shift: byte offsets, alignment, or integer constants. |
| `li` | 16,502 | Load a constant, initialize a loop/accumulator, or assign a saturation endpoint. |
| `lb` | 4,341,760 | Load a signed INT8 activation or weight. |
| `subw` | 2,146,304 | 32-bit subtraction: input centering or attention score shifting. |
| `mulw` | 2,146,304 | 32-bit low-product arithmetic: projection/attention MAC or GELU products. |
| `addw` | 2,129,920 | 32-bit addition: accumulation, residual/elementwise arithmetic, or rounding-independent sums. |
| `bne` | 2,375,716 | Continue a counted pointer/index loop. |
| `ld` | 16,441 | Load INT64 scratch data, pointers, or saved registers. |
| `sw` | 131,072 | Store an INT32 result, intermediate, or probability. |
| `ret` | 23 | Return to the caller; charged to the returning helper. |
| `j` | 1 | Unconditional jump, including a tail call or loop transfer. |
| `lw` | 163,840 | Load a signed INT32 accumulator, probability, parameter, or LUT value. |
| `bge` | 262,144 | Signed comparisons for clipping, maximum selection, or the exponential cutoff. |
| `addiw` | 16,394 | 32-bit immediate addition: rounding-shift setup or LUT index calculation. |
| `sll` | 11 | Register-controlled left shift: rounding constants or probability scaling. |
| `lui` | 37 | Construct a constant or high part of an address. |
| `sub` | 81,920 | 64-bit subtraction: centering, normalization, interpolation, square-root work, or row addressing. |
| `sra` | 131,072 | Arithmetic right shift: rounded fixed-point conversion. |
| `srai` | 16,385 | Immediate arithmetic right shift: fixed-point conversion, interpolation, or signed division setup. |
| `srli` | 16,384 | Logical right shift: integer square-root iteration, sign extraction, or division setup. |
| `sb` | 131,072 | Store an INT8 activation. |
| `and` | 16,384 | Extract the remainder for GELU second-round conversion. |
| `slt` | 16,384 | Compute the second-round correction predicate. |
| `jal` | 22 | Call a helper; its instruction is charged to the caller. |
| `auipc` | 3 | Construct a PC-relative parameter/table address. |
| **Total** | **19,595,940** |  |

### 5.4 Reconciliation with `minstret`

The supplied Spike counter interval contains **5 instructions outside the stage body**: the first counter read, argument setup, and the call. The ending read has not yet retired when its value is sampled. Instructions that subtract, save, or print the delta are outside this interval. This is specific to the supplied compiled call site.

```asm
80001ac2:	b0202af3          	csrr	s5,minstret
80001ac6:	658e                	ld	a1,192(sp)
80001ac8:	654a                	ld	a0,144(sp)
80001aca:	10b0                	addi	a2,sp,104
80001acc:	cb6ff0ef          	jal	80000f82 <block0_mlp_gelu_forward>
80001ad0:	b02027f3          	csrr	a5,minstret
```


| Case | Supplied raw minstret | Caller/measurement overhead | Counted stage body + callees | Unexplained difference |
|---|---|---|---|---|
| 0 | 19,595,945 | 5 | 19,595,940 | 0 |
| 1 | 19,595,945 | 5 | 19,595,940 | 0 |
| 2 | 19,595,943 | 5 | 19,595,938 | 0 |
| 3 | 19,595,945 | 5 | 19,595,940 | 0 |
| 4 | 19,595,943 | 5 | 19,595,938 | 0 |
| 5 | 19,595,948 | 5 | 19,595,943 | 0 |
| 6 | 19,595,942 | 5 | 19,595,937 | 0 |
| 7 | 19,595,942 | 5 | 19,595,937 | 0 |
| **Mean** | 19,595,944.125 | 5 | 19,595,939.125 | **0** |


Every case reconciles exactly. The body count includes all helper calls and their returns, and each wrapper instruction once per actual execution. The separately run staged and full paths both passed the supplied T107 comparisons.

| Conversion call | C0 clips | C1 clips | C2 clips | C3 clips | C4 clips | C5 clips | C6 clips | C7 clips |
|---|---|---|---|---|---|---|---|---|
| T89 conversion | 1 | 1 | 0 | 1 | 0 | 2 | 0 | 0 |
| T92 conversion | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| T94 conversion | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| T96 conversion | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| chunk 0: T97 conversion | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| chunk 1: T97 conversion | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| chunk 2: T97 conversion | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| chunk 3: T97 conversion | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 |
| T99 conversion | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| T103 conversion | 2 | 2 | 0 | 1 | 0 | 2 | 0 | 0 |
| T104 two-round conversion | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

These are clipping assignments executed during the semantic replay of each helper call, not clipping counts guessed from the stage residual. Each clipped element adds one instruction in this binary.

## 6. Findings

| Region | Case-0 instructions | Share of stage body |
|---|---|---|
| Computation | 18,055,519 | 92.14% |
| Requantization | 1,540,200 | 7.86% |
| Wrapper | 221 | 0.00% |

| More specific region | Case-0 instructions | Share of stage body |
|---|---|---|
| Expansion accumulation | 16,974,136 | 86.62% |
| Expansion T89 conversion | 180,234 | 0.92% |
| GELU helpers, including GELU conversions | 2,441,349 | 12.46% |
| Wrapper, including chunk control | 221 | 0.0011% |

The stage name combines two costs: the expansion projection is the main instruction-count hotspot; GELU is a smaller, multi-pass integer pipeline. GELU still executes 16,384-element passes and separate conversions, including four INT64 chunks and the required two-round final conversion. The entire stage is about **34.27%** of full-block raw instructions.

Instruction shares identify where instructions are executed. They do not establish load latency, multiplier latency, cache behavior, or CV-Wally cycle speedup. No accelerator or custom-instruction speedup is measured here. The eight fixture passes validate these supplied integer cases; they do not establish equality with the original mixed TFLite graph or unseen-data quality.

[Back to results](Result.md) · [Counting method and validation](analysis_method.md)
