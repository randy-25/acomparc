/*
 * Install as experiments/integer_inference/tests/block0/profile/test_block0_profile.c.
 * Link the same kernels, parameters, runtime and final-stage fixtures as full/.
 *
 * Pass A measures one block0_forward invocation per case.
 * Pass B separately measures the seven stage calls in an equivalent C chain.
 * Both outputs are validated against the same chained Python T107 golden.
 * Reports RAW minstret deltas: do not reuse a correction from another build.
 * Validation and printing are outside all intervals. No CV-Wally cycle claims.
 */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "../../../src/blocks/block0.h"
#include "../../../params/block0_params.h"
#include "../../../fixtures/block0/mlp_out_residual_golden.h"

#define ELEMENTS (BLOCK0_TOKEN_COUNT * BLOCK0_HIDDEN_SIZE)
#define MLP_ELEMENTS \
    (BLOCK0_TOKEN_COUNT * BLOCK0_MLP_EXPAND_OUTPUT_FEATURES)
#define PROFILE_STAGE_COUNT 7u

_Static_assert(ELEMENTS == BLOCK0_MLP_OUT_RESIDUAL_ELEMENTS,
               "Golden output shape mismatch");
_Static_assert(MLP_ELEMENTS == BLOCK0_MLP_OUT_RESIDUAL_INPUT_ELEMENTS,
               "MLP workspace shape mismatch");
_Static_assert(BLOCK0_MLP_OUT_INPUT_FEATURES ==
               BLOCK0_MLP_EXPAND_OUTPUT_FEATURES,
               "MLP feature mismatch");
_Static_assert(BLOCK0_MLP_OUT_OUTPUT_FEATURES == BLOCK0_HIDDEN_SIZE,
               "Projection output feature mismatch");
_Static_assert(BLOCK0_GELU_ADD_CHUNK_ELEMENTS >= ELEMENTS,
               "Shared fine workspace must hold LN intermediates");
_Static_assert(BLOCK0_RESIDUAL2_FRAC_BITS == 16u,
               "Golden output uses scale 2^-16");

#if defined(__riscv) && (__riscv_xlen == 64)
#define BLOCK0_HAS_INSTRET 1

static inline __attribute__((always_inline))
uint64_t read_minstret(void)
{
    uint64_t value;
    __asm__ volatile ("csrr %0, minstret"
                      : "=r"(value) : : "memory");
    return value;
}

/* All three reads and the eight nops are in one assembly block. */
static int check_instruction_counter(void)
{
    uint64_t first, second, third;
    __asm__ volatile (
        "csrr %0, minstret\n\t"
        "csrr %1, minstret\n\t"
        ".rept 8\n\t"
        "nop\n\t"
        ".endr\n\t"
        "csrr %2, minstret"
        : "=&r"(first), "=&r"(second), "=&r"(third)
        : : "memory"
    );

    const uint64_t adjacent = second - first;
    const uint64_t eight_nops = third - second;
    printf("Counter sanity: adjacent=%" PRIu64 ", eight_nops=%" PRIu64 "\n",
           adjacent, eight_nops);

    /* For this uninterrupted sequence, expected deltas are 1 and 9. */
    if (adjacent != UINT64_C(1) || eight_nops != UINT64_C(9)) {
        printf("FAIL: unexpected minstret behavior; inspect the counter probe\n");
        return 1;
    }
    return 0;
}
#else
#define BLOCK0_HAS_INSTRET 0
#endif

#if BLOCK0_HAS_INSTRET
static const char *const stage_names[PROFILE_STAGE_COUNT] = {
    "LN1", "QKV", "Attention", "Out projection + residual",
    "LN2", "MLP expansion + GELU", "MLP output + residual"
};

/* Store each result after its ending read; do not print between stages. */
#define PROFILE_STAGE(index, ...)                                      \
    do {                                                               \
        const uint64_t stage_before = read_minstret();                  \
        __VA_ARGS__;                                                   \
        const uint64_t stage_after = read_minstret();                   \
        stage_counts[(index)] = stage_after - stage_before;             \
    } while (0)
#else
#define PROFILE_STAGE(index, ...) do { __VA_ARGS__; } while (0)
#endif

static int check_output(unsigned int c, const char *path,
                        const int32_t *actual, const int32_t *expected)
{
    for (size_t i = 0; i < ELEMENTS; ++i) {
        if (actual[i] != expected[i]) {
            printf("FAIL case %u %s T107 index %u: got %" PRId32
                   ", expected %" PRId32 "\n",
                   c, path, (unsigned int)i, actual[i], expected[i]);
            return 1;
        }
    }
    return 0;
}

/* Same shared working-buffer arrangement as the supplied full-block test. */
static int32_t mean_sum[BLOCK0_TOKEN_COUNT];
static int32_t variance[BLOCK0_TOKEN_COUNT];
static int32_t inv_std[BLOCK0_TOKEN_COUNT];
static int32_t product_workspace[MLP_ELEMENTS];
static int64_t fine_workspace[BLOCK0_GELU_ADD_CHUNK_ELEMENTS];
static int32_t score_workspace[BLOCK0_SCORE_ELEMENTS];

static int8_t activation[ELEMENTS];
static int8_t query[ELEMENTS];
static int8_t key[ELEMENTS];
static int8_t value[ELEMENTS];
static int8_t attention_output[ELEMENTS];
static int8_t projection_output[ELEMENTS];
static int8_t t89[MLP_ELEMENTS];
static int8_t half[MLP_ELEMENTS];
static int8_t branch[MLP_ELEMENTS];
static int8_t mlp_output[MLP_ELEMENTS];

int main(void)
{
    Block0Workspace workspace = {
        .ln = {
            .mean_sum = mean_sum,
            .variance = variance,
            .inv_std = inv_std,
            .feature_i32 = product_workspace,
            .fine_i64 = fine_workspace
        },
        .mlp = {
            .t89 = t89,
            .half = half,
            .branch = branch,
            .product_i32 = product_workspace,
            .wide_i64 = fine_workspace
        },
        .activation = activation,
        .query = query,
        .key = key,
        .value = value,
        .attention_output = attention_output,
        .projection_output = projection_output,
        .mlp_output = mlp_output,
        .score_workspace = score_workspace
    };

#if BLOCK0_HAS_INSTRET
    if (check_instruction_counter()) {
        return 1;
    }
#else
    printf("HOST: validating outputs; minstret measurement skipped\n");
#endif

    for (unsigned int c = 0;
         c < BLOCK0_MLP_OUT_RESIDUAL_CASE_COUNT; ++c) {
        const int8_t *input = block0_mlp_out_residual_t0[c];
        const int32_t *expected = block0_mlp_out_residual_chained[c].t107;

#if BLOCK0_HAS_INSTRET
        const uint64_t before = read_minstret();
#endif

        block0_forward(input, product_workspace, &workspace);

#if BLOCK0_HAS_INSTRET
        const uint64_t after = read_minstret();
#endif

        if (check_output(c, "full", product_workspace, expected)) {
            return 1;
        }

#if BLOCK0_HAS_INSTRET
        printf("Case %u: block0_forward raw minstret delta = %" PRIu64 "\n",
               c, after - before);
#endif
        printf("Case %u: full T107 PASS\n", c);

        /*
         * A second, separately measured pass from the same T0.
         * This follows block0_forward's buffer reuse and stage order.
         * It is not part of the full-forward measurement above.
         */
#if BLOCK0_HAS_INSTRET
        uint64_t stage_counts[PROFILE_STAGE_COUNT];
#endif

        PROFILE_STAGE(0,
            block0_ln1_forward(input, workspace.activation, &workspace.ln)
        );
        PROFILE_STAGE(1,
            block0_qkv_forward(
                workspace.activation, workspace.query, workspace.key,
                workspace.value, workspace.ln.feature_i32
            )
        );
        PROFILE_STAGE(2,
            block0_attention_forward(
                workspace.query, workspace.key, workspace.value,
                workspace.attention_output, workspace.score_workspace,
                workspace.ln.feature_i32
            )
        );
        /* Scores are dead; their storage now retains T73. */
        PROFILE_STAGE(3,
            block0_out_residual_forward(
                workspace.attention_output, input, workspace.score_workspace,
                workspace.projection_output, workspace.ln.feature_i32
            )
        );
        PROFILE_STAGE(4,
            block0_ln2_forward(
                workspace.score_workspace, workspace.activation, &workspace.ln
            )
        );
        PROFILE_STAGE(5,
            block0_mlp_gelu_forward(
                workspace.activation, workspace.mlp_output, &workspace.mlp
            )
        );
        /* Retained T73 is still live; final output may reuse feature_i32. */
        PROFILE_STAGE(6,
            block0_mlp_out_residual_forward(
                workspace.mlp_output, workspace.score_workspace,
                product_workspace, workspace.projection_output,
                workspace.ln.feature_i32
            )
        );

        if (check_output(c, "staged", product_workspace, expected)) {
            return 1;
        }

#if BLOCK0_HAS_INSTRET
        uint64_t stage_sum = 0;
        for (unsigned int stage = 0; stage < PROFILE_STAGE_COUNT; ++stage) {
            printf("Case %u: Part %u %s raw minstret delta = %" PRIu64 "\n",
                   c, stage + 1u, stage_names[stage], stage_counts[stage]);
            stage_sum += stage_counts[stage];
        }
        printf("Case %u: sum of raw stage deltas = %" PRIu64 "\n", c, stage_sum);
#endif
        printf("Case %u: staged T107 PASS\n", c);
    }

    printf("PASS: %u cases, %u outputs checked per path, "
           "full and staged paths have 0 mismatches\n",
           (unsigned int)BLOCK0_MLP_OUT_RESIDUAL_CASE_COUNT,
           (unsigned int)(BLOCK0_MLP_OUT_RESIDUAL_CASE_COUNT * ELEMENTS));
    return 0;
}
