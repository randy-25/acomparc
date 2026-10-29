# Scalar and custom instruction comparison

Counts below describe instruction retirement and static assembly. They do not estimate accelerator cycle latency or RTL speedup. No counter-overhead correction is applied.

## Assembly checks

Scalar: no XTILE/XREQUANT opcodes found. Custom: both RUN opcodes found. Detection uses encoded opcode/funct3 fields, including when objdump displays `.4byte` instead of a mnemonic.

| Custom operation | Static instruction sites |
|---|---:|
| xrequant_cfg | 5 |
| xrequant_run | 5 |
| xrequant_set_io | 5 |
| xrequant_set_mul | 5 |
| xrequant_set_zp | 5 |
| xtile_cfg | 1 |
| xtile_run | 1 |
| xtile_set_ab | 1 |
| xtile_set_c | 1 |
| xtile_set_zp | 1 |

Static sites are not execution counts. See `static_functions.csv` and the two extracted kernel disassemblies.

## Raw retired-instruction comparison

Totals aggregate 1 matched case(s); stage rows also aggregate all eight blocks. Whole-chain and stage intervals are separately measured.

| Region | Scalar raw | Custom raw | Reduction | Scalar/custom instruction ratio | Share of custom stage intervals |
|---|---:|---:|---:|---:|---:|
| whole_chain | 461087565 | 36302539 | 92.1268% | 12.701 | — |
| ln1 | 2117309 | 1232756 | 41.7772% | 1.718 | 3.3958% |
| qkv | 102736208 | 1636 | 99.9984% | 62797.193 | 0.0045% |
| attention | 23437008 | 23011073 | 1.8174% | 1.019 | 63.3868% |
| out_residual | 35132301 | 592363 | 98.3139% | 59.309 | 1.6317% |
| ln2 | 2117329 | 1232771 | 41.7771% | 1.718 | 3.3958% |
| mlp_gelu | 159391281 | 9704984 | 93.9112% | 16.424 | 26.7336% |
| mlp_out_boundary | 136156197 | 527024 | 99.6129% | 258.349 | 1.4518% |

The ratio is a retired-instruction ratio, not a cycle speedup. Per-block results and raw aggregate rows are in `dynamic_regions.csv`. The sum of stage intervals can differ from a complete-block or chain interval because their call paths and measurement boundaries differ.
