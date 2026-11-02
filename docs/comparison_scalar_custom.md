# Comparison with scalar C accumulation

This comparison provides a rough estimate of the potential cycle reduction from replacing scalar MAC computations with a systolic-array engine. The custom-engine estimate currently covers computation only, excluding memory transfers and control overhead.

## Scalar and Custom Engine Cycle Comparison

- [Scalar cycle measurements](scalar_cycle_analysis.md)
- [Systolic array estimation](systolic_array_estimation_cycle.md)


| Kernel / scope | Scalar cycles | Estimated array compute cycles | Scalar / array-compute ratio |
|---|---:|---:|---:|
| Q projection | 4,254,689 | 9,216 | 461.66× |
| MLP expansion | 17,272,290 | 36,864 | 468.54× |
| MLP output | 17,147,067 | 33,792 | 507.43× |
| One block — extrapolated | 51,438,113 | 107,520 | 478.41× |
| Eight blocks — extrapolated | 411,504,904 | 860,160 | 478.41× |

The scalar block estimate uses the measured Q kernel as a proxy for all four 128→128 projections (Q, K, V, and attention output):

$$
T_{scalar,block}=4(4,254,689)+17,272,290+17,147,067=51,438,113
$$

$$
T_{scalar,8\ blocks}=8(51,438,113)=411,504,904\text{ cycles}
$$

The eight-block cycle-count ratio is therefore:

$$
\frac{411,504,904}{860,160}\approx478.41
$$


## Scope

The **860,160-cycle estimate covers dense projection computation only**, with no stalls during array execution. It excludes memory transfers, buffer preloading, zero-point preparation, tile setup and transitions, result readout, bias, requantization, attention matrix products, normalization, softmax, GELU, residual operations, embedding, and the final head. Accumulator clearing is assumed to add no counted cycles; an RTL implementation must account for any extra work.
