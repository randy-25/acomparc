# Counting Method and Validation

## Scope and evidence

These notes describe the files from the user's `latest.zip`: current C kernels, block-0 orchestration and parameters, profile driver, RISC-V ELF, matching disassembly, and fixtures. They do not describe an edited or rebuilt laptop repository.

The dynamic measurements in `Result.md` are the user's supplied Spike `minstret` results for `rv64imc_zicsr`. The more detailed per-helper and per-PC breakdown was obtained locally by a **scoped RV64 instruction-semantic replay** of the supplied disassembly with memory initialized from the supplied ELF. This is an analysis harness, not Spike and not a cycle simulator.

The analysis decodes 1,591 instruction sites across 39 function symbols and 38 printed mnemonics in the inference region. Instruction encoding bytes in the disassembly were checked against the ELF load image. The replay executes each actual branch and arithmetic operation, with RV64 register widths and `*w` sign extension, from `block0_forward()` through its helpers. It starts directly at that function, using the supplied workspace pointer table and fixture inputs; it does not execute CRT startup, counter CSR reads, printing, or test comparisons as inference instructions.

Compressed and uncompressed instructions each count as one retired instruction. Printed aliases such as `li`, `mv`, and `ret` each refer to one encoded instruction at the listed address. Multi-instruction constant construction is counted at every address actually executed. The notes count dynamic execution frequency, not static opcode occurrences.

The scoped replay was written for this particular supplied ELF. Its fixed fixture and workspace addresses must be updated if the ELF is rebuilt. It is not a general RISC-V simulator or an architectural certification test.

## Validation performed locally

1. Checked the parsed instruction bytes against the supplied ELF.
2. Replayed all eight fixture inputs through the supplied compiled `block0_forward()`.
3. Compared each final INT32 T107 element against the corresponding golden embedded in the supplied ELF: **32,768 outputs, zero mismatches**.
4. Summed per-PC counts, per-helper counts, and per-stage counts independently within the replay; the totals agree.
5. Compared replay stage bodies plus exact compiled call-site overhead against all **56 supplied stage measurements**: zero differences.
6. Compared replay full-body counts plus the full call-site overhead against all **eight supplied full measurements**: zero differences.
7. Separately compiled the uploaded C snapshot with host GCC using `-std=c11 -O2 -Wall -Wextra -Werror`. The original profile driver passed both full and staged output checks for all eight fixtures, 32,768 outputs per path.

No new Spike run or CV-Wally hardware/cycle measurement was performed locally. Matching fixture outputs and instruction counts strongly supports this breakdown for the supplied binary; it does not establish correctness for untested inputs or equality with the original mixed TFLite graph.

## Counting boundaries

The first read retires after sampling `minstret`; the second read has not retired when its value is sampled. The user's adjacent-read sanity result of 1 and eight-NOP result of 9 are consistent with that boundary. The raw interval includes the first read, argument preparation, and the call into the measured function. It excludes later subtraction, storage, and printing.

| Region | First read PC | Ending read PC | Outside-body instructions in raw delta |
|---|---|---|---|
| full block | 0x800019ae | 0x800019c2 | 6 |
| Part 1 | 0x80001a3c | 0x80001a4a | 5 |
| Part 2 | 0x80001a54 | 0x80001a66 | 7 |
| Part 3 | 0x80001a70 | 0x80001a84 | 8 |
| Part 4 | 0x80001a8e | 0x80001aa0 | 7 |
| Part 5 | 0x80001aaa | 0x80001ab8 | 5 |
| Part 6 | 0x80001ac2 | 0x80001ad0 | 5 |
| Part 7 | 0x80001ada | 0x80001af2 | 8 |

### Full-block call site

```asm
800019ae:	b0202af3          	csrr	s5,minstret
800019b2:	001bc597          	auipc	a1,0x1bc
800019b6:	69658593          	addi	a1,a1,1686 # 801be048 <product_workspace>
800019ba:	0090                	addi	a2,sp,64
800019bc:	854a                	mv	a0,s2
800019be:	ffaff0ef          	jal	800011b8 <block0_forward>
800019c2:	b0202673          	csrr	a2,minstret
```


`block0_forward()` executes 48 wrapper instructions in addition to its seven stage bodies. The standalone stage measurement overheads sum to 45, while the full measurement adds six instructions outside the full function. Therefore:

$$
I_{\rm full,raw}=\sum_s I_{s,\rm body}+48+6,
$$
$$
\sum_s I_{s,\rm raw}=\sum_s I_{s,\rm body}+45,
$$
$$
I_{\rm full,raw}-\sum_s I_{s,\rm raw}=48+6-45=9.
$$

This explains the constant nine-instruction difference without assigning it to a tensor operation. These offsets are specific to this compiled binary and call site.

## Ownership and avoiding double counting

- A function's **inclusive** count contains its own instructions and all executed callees. A wrapper's **exclusive** count contains only instructions belonging to that wrapper. Detailed helper tables add wrapper-exclusive and helper counts; they do not add the inclusive wrapper a second time.
- The `jal` is charged to its caller; the `ret` to the returning callee. A tail `j` is charged once to the caller and the final helper returns directly to the original caller.
- Source wrappers `block0_attention_scores_forward()` and `block0_mlp_expand_forward()` are inlined inside the corresponding measured stages. Their standalone symbols are not separately charged there.
- Shared helper assembly listings show the **aggregate case-0 frequency within that stage**. For example, a GELU conversion helper appears in several calls, and its annotation includes all those calls. Per-invocation counts remain separate in the helper tables.
- Tensor MAC counts are derived from loop shapes; opcode histograms also include requantization products, alignment products, elementwise GELU products, and dimension/address calculations. `add` and `addw` totals therefore are not synonymous with MAC counts.
- Immediate additions (`addi`, `addiw`) include pointers, loop counters, stack adjustment, and some numerical constants. They are not all tensor additions. Instruction-purpose descriptions preserve that distinction.

## Interpreting bottlenecks

The evidence locates **instruction-count hotspots**. It cannot determine which instruction category dominates cycles. A load instruction may hit or miss a cache; a multiplication, division, and branch may have different latencies and pipeline behavior. Reducing retired instructions does not imply an equal cycle speedup. No accelerator setup, transfer, synchronization, or hardware overlap is modeled.

Means are descriptive arithmetic averages across one benchmark and seven calibration inputs. They are not statistical estimates of unseen workloads. All-case tables retain the observed variation.

## Reproducibility and source identity

The `evidence/` folder contains per-PC frequencies, helper counts, stage opcode mixes, and exact reconciliation data. It also contains the scoped replay scripts and a short reproduction guide. SHA-256 hashes for the supplied sources and binaries are recorded in `evidence/counts.json`.

| Supplied file | SHA-256 |
|---|---|
| `build/block0/profile/test_block0_profile.elf` | `9817604c28aa5ca6e39bae36fac60ccbb8888ab57befde5b481620aa1442aecb` |
| `build/block0/profile/test_block0_profile.dis` | `d039b27d350df42e7cab3aad78d04f28be5111cf24522445fd9e06b1d6d0011e` |
| `src/blocks/block0.c` | `10d92eb8b95bb68cae39496053a0de8cf4a5763602d8b1a11f080ab195fe388a` |
| `params/block0_params.h` | `ebfae626d9242550128e5b33901530648dac456e22979a3d8ef6290faaa8058c` |

[Back to results](Result.md)

## Appendix: Full-block wrapper

The wrapper below executes 48 instructions, each once per inference in these fixtures, in addition to its seven stage call trees. It is outside the seven stage bodies.

```asm
00000000800011b8 <block0_forward>:
800011b8:	1101                	addi	sp,sp,-32
800011ba:	e04a                	sd	s2,0(sp)
800011bc:	892e                	mv	s2,a1
800011be:	6a2c                	ld	a1,80(a2)
800011c0:	ec06                	sd	ra,24(sp)
800011c2:	e822                	sd	s0,16(sp)
800011c4:	e426                	sd	s1,8(sp)
800011c6:	8432                	mv	s0,a2
800011c8:	84aa                	mv	s1,a0
800011ca:	9f9ff0ef          	jal	80000bc2 <block0_ln1_forward>
800011ce:	6c18                	ld	a4,24(s0)
800011d0:	7434                	ld	a3,104(s0)
800011d2:	7030                	ld	a2,96(s0)
800011d4:	6c2c                	ld	a1,88(s0)
800011d6:	6828                	ld	a0,80(s0)
800011d8:	aafff0ef          	jal	80000c86 <block0_qkv_forward>
800011dc:	6c1c                	ld	a5,24(s0)
800011de:	6458                	ld	a4,136(s0)
800011e0:	7834                	ld	a3,112(s0)
800011e2:	7430                	ld	a2,104(s0)
800011e4:	702c                	ld	a1,96(s0)
800011e6:	6c28                	ld	a0,88(s0)
800011e8:	b71ff0ef          	jal	80000d58 <block0_attention_forward>
800011ec:	6c18                	ld	a4,24(s0)
800011ee:	7c34                	ld	a3,120(s0)
800011f0:	6450                	ld	a2,136(s0)
800011f2:	7828                	ld	a0,112(s0)
800011f4:	85a6                	mv	a1,s1
800011f6:	bfdff0ef          	jal	80000df2 <block0_out_residual_forward>
800011fa:	682c                	ld	a1,80(s0)
800011fc:	6448                	ld	a0,136(s0)
800011fe:	8622                	mv	a2,s0
80001200:	c73ff0ef          	jal	80000e72 <block0_ln2_forward>
80001204:	604c                	ld	a1,128(s0)
80001206:	6828                	ld	a0,80(s0)
80001208:	02840613          	addi	a2,s0,40
8000120c:	d77ff0ef          	jal	80000f82 <block0_mlp_gelu_forward>
80001210:	6c18                	ld	a4,24(s0)
80001212:	7c34                	ld	a3,120(s0)
80001214:	644c                	ld	a1,136(s0)
80001216:	6048                	ld	a0,128(s0)
80001218:	6442                	ld	s0,16(sp)
8000121a:	60e2                	ld	ra,24(sp)
8000121c:	64a2                	ld	s1,8(sp)
8000121e:	864a                	mv	a2,s2
80001220:	6902                	ld	s2,0(sp)
80001222:	6105                	addi	sp,sp,32
80001224:	bf19                	j	8000113a <block0_mlp_out_residual_forward>
```
