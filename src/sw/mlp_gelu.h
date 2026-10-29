#ifndef TINYSTORIES_ALLBLOCKS_MLP_GELU_H
#define TINYSTORIES_ALLBLOCKS_MLP_GELU_H
#include <stdint.h>

#define MLP_GELU_ADD_CHUNK_ELEMENTS 4096u
typedef struct {
    int8_t *t89;          /* 16384: expansion output, retained through GELU */
    int8_t *half;         /* 16384: half-input branch */
    int8_t *branch;       /* 16384: cube -> weighted -> sum -> argument -> factor */
    int32_t *product_i32; /* 16384: expansion accumulators and GELU products */
    int64_t *wide_i64;    /* 4096: shared wide branch-sum chunk */
} MlpGeluWorkspace;
#endif
