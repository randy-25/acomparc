#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "model.h"
#include "block0.h"
#include "block1.h"
#include "block2.h"
#include "block3.h"
#include "block4.h"
#include "block5.h"
#include "block6.h"
#include "block7.h"
#include "full_golden.h"

#ifndef PROFILE_ONLY
#define PROFILE_ONLY 0
#endif
#ifndef PROFILE_CASE_COUNT
#define PROFILE_CASE_COUNT 1
#endif
#if BUILD_CUSTOM_BACKEND
#define BACKEND_NAME "custom"
#else
#define BACKEND_NAME "scalar"
#endif
_Static_assert(PROFILE_CASE_COUNT >= 1 && PROFILE_CASE_COUNT <= FULL_CASE_COUNT, "Invalid profile case count");
_Static_assert(MLP_GELU_ADD_CHUNK_ELEMENTS >= FULL_ELEMENTS, "Shared wide allocation is too small");

#define GUARDED_TYPE(name,type,count) \
    typedef struct { uint64_t before; type data[count]; uint64_t after; } name
GUARDED_TYPE(SmallI8,int8_t,4096);
GUARDED_TYPE(BigI8,int8_t,16384);
GUARDED_TYPE(BigI32,int32_t,16384);
GUARDED_TYPE(RowI32,int32_t,32);
GUARDED_TYPE(WideI64,int64_t,MLP_GELU_ADD_CHUNK_ELEMENTS);
static SmallI8 input,output,scratch,activation,query,key,value,attention,projection;
static BigI8 expansion,half,branch,mlp_output;
static BigI32 product,scores;
static RowI32 mean_sum,variance,inv_std;
static WideI64 wide;
static TransformerBlockWorkspace workspace = {
    {mean_sum.data,variance.data,inv_std.data,product.data,wide.data},
    {expansion.data,half.data,branch.data,product.data,wide.data},
    activation.data,query.data,key.data,value.data,attention.data,projection.data,
    mlp_output.data,scores.data
};
#define CANARY UINT64_C(0x6ea193b8c07245df)
#define SET(buffer) do { (buffer).before=CANARY; (buffer).after=~CANARY; } while(0)
#define GUARD(buffer) do { if ((buffer).before!=CANARY || (buffer).after!=~CANARY) { \
    printf("FAIL case %u: %s guard changed\n",c,#buffer); return 1; } } while(0)

static void reset_buffers(unsigned int c)
{
    SET(input); SET(output); SET(scratch); SET(activation); SET(query); SET(key); SET(value);
    SET(attention); SET(projection); SET(expansion); SET(half); SET(branch); SET(mlp_output);
    SET(product); SET(scores); SET(mean_sum); SET(variance); SET(inv_std); SET(wide);
    for (size_t i=0;i<4096;++i) {
        input.data[i]=full_cases[c].input[i]; output.data[i]=93; scratch.data[i]=93;
        activation.data[i]=93; query.data[i]=93; key.data[i]=93; value.data[i]=93;
        attention.data[i]=93; projection.data[i]=93;
    }
    for (size_t i=0;i<16384;++i) {
        expansion.data[i]=93; half.data[i]=93; branch.data[i]=93; mlp_output.data[i]=93;
        product.data[i]=1234567; scores.data[i]=-12345;
    }
    for (size_t i=0;i<32;++i) mean_sum.data[i]=variance.data[i]=inv_std.data[i]=-12345;
    for (size_t i=0;i<MLP_GELU_ADD_CHUNK_ELEMENTS;++i) wide.data[i]=INT64_C(1234567890123);
}

static int check_guards(unsigned int c)
{
    GUARD(input); GUARD(output); GUARD(scratch); GUARD(activation); GUARD(query); GUARD(key); GUARD(value);
    GUARD(attention); GUARD(projection); GUARD(expansion); GUARD(half); GUARD(branch); GUARD(mlp_output);
    GUARD(product); GUARD(scores); GUARD(mean_sum); GUARD(variance); GUARD(inv_std); GUARD(wide);
    for (size_t i=0;i<FULL_ELEMENTS;++i) if(input.data[i]!=full_cases[c].input[i]) {
        printf("FAIL case %u: input changed at %u\n",c,(unsigned int)i); return 1;
    }
    return 0;
}

static int check_i8(unsigned int c,unsigned int b,const char *path,const int8_t *actual)
{
    for (size_t i=0;i<FULL_ELEMENTS;++i) if(actual[i]!=full_cases[c].output[b][i]) {
        printf("FAIL case %u block %u %s INT8[%u]: got %d expected %d\n",c,b,path,
            (unsigned int)i,(int)actual[i],(int)full_cases[c].output[b][i]); return 1;
    }
    return 0;
}

static int check_i32(unsigned int c,unsigned int b,const char *path)
{
    for (size_t i=0;i<FULL_ELEMENTS;++i) if(scores.data[i]!=full_cases[c].pre_boundary[b][i]) {
        printf("FAIL case %u block %u %s INT32[%u]: got %" PRId32 " expected %" PRId32 "\n",
            c,b,path,(unsigned int)i,scores.data[i],full_cases[c].pre_boundary[b][i]); return 1;
    }
    return 0;
}

static int check_final(unsigned int c)
{
    return check_i8(c,7,"actual chain",output.data) || check_i32(c,7,"actual chain") ||
           check_i8(c,6,"chain scratch",scratch.data) || check_guards(c);
}

#if PROFILE_ONLY
#if !defined(__riscv) || !defined(__riscv_xlen) || __riscv_xlen != 64
#error "Instruction profiling requires RV64; native validation uses PROFILE_ONLY=0"
#endif
static inline __attribute__((always_inline)) uint64_t read_minstret(void)
{
    uint64_t value;
    __asm__ volatile("csrr %0, minstret" : "=r"(value) : : "memory");
    return value;
}

static int check_counter(void)
{
    uint64_t first,second,third;
    __asm__ volatile("csrr %0, minstret\n\tcsrr %1, minstret\n\t"
        ".rept 8\n\tnop\n\t.endr\n\tcsrr %2, minstret"
        : "=&r"(first),"=&r"(second),"=&r"(third) : : "memory");
    printf("Counter sanity: adjacent=%" PRIu64 ", eight_nops=%" PRIu64 "\n",second-first,third-second);
    return second-first!=UINT64_C(1) || third-second!=UINT64_C(9);
}
#endif

static int check_block_chain(unsigned int c,uint64_t *counts)
{
    const int8_t *current=input.data;
#if !PROFILE_ONLY
    (void)counts;
#endif
    for (unsigned int b=0;b<TINYSTORIES_BLOCK_COUNT;++b) {
        int8_t *next=(b&1u)?output.data:scratch.data;
#if PROFILE_ONLY
        const uint64_t before=read_minstret();
#endif
        tinystories_block_forwards[b](current,next,&workspace);
#if PROFILE_ONLY
        const uint64_t after=read_minstret(); counts[b]=after-before;
#endif
        if(check_i8(c,b,"block chain",next) || check_i32(c,b,"block chain") || check_guards(c)) return 1;
        current=next;
    }
    return 0;
}

#if PROFILE_ONLY
typedef void (*Ln1)(const int8_t *,int8_t *,LayerNormI8Workspace *);
typedef void (*Qkv)(const int8_t *,int8_t *,int8_t *,int8_t *,int32_t *);
typedef void (*Attention)(const int8_t *,const int8_t *,const int8_t *,int8_t *,int32_t *,int32_t *);
typedef void (*OutResidual)(const int8_t *,const int8_t *,int32_t *,int8_t *,int32_t *);
typedef void (*Ln2)(const int32_t *,int8_t *,LayerNormI8Workspace *);
typedef void (*Mlp)(const int8_t *,int8_t *,MlpGeluWorkspace *);
typedef void (*MlpOut)(const int8_t *,const int32_t *,int32_t *,int8_t *,int32_t *);
typedef void (*Boundary)(const int32_t *,int8_t *);
typedef struct {Ln1 ln1;Qkv qkv;Attention attention;OutResidual out_residual;Ln2 ln2;Mlp mlp;MlpOut mlp_out;Boundary boundary;} Stages;
#define STAGES(n) {block##n##_ln1_forward,block##n##_qkv_forward,block##n##_attention_forward, \
    block##n##_out_residual_forward,block##n##_ln2_forward,block##n##_mlp_gelu_forward, \
    block##n##_mlp_out_residual_forward,block##n##_output_boundary_forward}
static const Stages stages[] = {STAGES(0),STAGES(1),STAGES(2),STAGES(3),STAGES(4),STAGES(5),STAGES(6),STAGES(7)};
static const char *const stage_names[] = {"ln1","qkv","attention","out_residual","ln2","mlp_gelu","mlp_out_boundary"};
#define MEASURE(index,...) do { const uint64_t before=read_minstret(); \
    __VA_ARGS__; const uint64_t after=read_minstret(); counts[b][index]=after-before; } while(0)

static int check_stage_chain(unsigned int c,uint64_t counts[8][7])
{
    const int8_t *current=input.data;
    for (unsigned int b=0;b<TINYSTORIES_BLOCK_COUNT;++b) {
        const Stages *p=&stages[b];
        int8_t *next=(b&1u)?output.data:scratch.data;
        MEASURE(0,p->ln1(current,workspace.activation,&workspace.ln));
        MEASURE(1,p->qkv(workspace.activation,workspace.query,workspace.key,workspace.value,workspace.ln.feature_i32));
        MEASURE(2,p->attention(workspace.query,workspace.key,workspace.value,workspace.attention_output,
                              workspace.score_workspace,workspace.ln.feature_i32));
        MEASURE(3,p->out_residual(workspace.attention_output,current,workspace.score_workspace,
                                 workspace.projection_output,workspace.ln.feature_i32));
        MEASURE(4,p->ln2(workspace.score_workspace,workspace.activation,&workspace.ln));
        MEASURE(5,p->mlp(workspace.activation,workspace.mlp_output,&workspace.mlp));
        MEASURE(6,p->mlp_out(workspace.mlp_output,workspace.score_workspace,workspace.score_workspace,
                             workspace.projection_output,workspace.ln.feature_i32);
                  p->boundary(workspace.score_workspace,next));
        if(check_i8(c,b,"stage chain",next) || check_i32(c,b,"stage chain") || check_guards(c)) return 1;
        current=next;
    }
    return 0;
}
#endif

int main(void)
{
#if PROFILE_ONLY
    if(check_counter()) {printf("FAIL: counter sanity\n");return 1;}
    printf("PROFILE_HEADER,backend,case,block,region,raw_minstret\n");
    for (unsigned int c=0;c<PROFILE_CASE_COUNT;++c) {
        uint64_t block_counts[8],stage_counts[8][7];
        reset_buffers(c);
        const uint64_t before=read_minstret();
        tinystories_blocks_forward(input.data,output.data,scratch.data,&workspace);
        const uint64_t after=read_minstret();
        const uint64_t chain_count=after-before;
        if(check_final(c)) return 1;
        reset_buffers(c);if(check_block_chain(c,block_counts)) return 1;
        reset_buffers(c);if(check_stage_chain(c,stage_counts)) return 1;
        printf("PROFILE,%s,%u,-1,chain,%" PRIu64 "\n",BACKEND_NAME,c,chain_count);
        uint64_t block_sum=0;
        for (unsigned int b=0;b<8;++b) {
            uint64_t stage_sum=0;block_sum+=block_counts[b];
            printf("PROFILE,%s,%u,%u,forward,%" PRIu64 "\n",BACKEND_NAME,c,b,block_counts[b]);
            for (unsigned int s=0;s<7;++s) {
                stage_sum+=stage_counts[b][s];
                printf("PROFILE,%s,%u,%u,%s,%" PRIu64 "\n",BACKEND_NAME,c,b,stage_names[s],stage_counts[b][s]);
            }
            printf("PROFILE,%s,%u,%u,sum_stage_raw,%" PRIu64 "\n",BACKEND_NAME,c,b,stage_sum);
        }
        printf("PROFILE,%s,%u,-1,sum_block_raw,%" PRIu64 "\n",BACKEND_NAME,c,block_sum);
    }
    printf("PASS: backend=%s mode=profile cases=%u; all chain, block and stage outputs match\n",BACKEND_NAME,(unsigned int)PROFILE_CASE_COUNT);
#else
    for (unsigned int c=0;c<FULL_CASE_COUNT;++c) {
        reset_buffers(c);tinystories_blocks_forward(input.data,output.data,scratch.data,&workspace);
        if(check_final(c)) return 1;
        reset_buffers(c);if(check_block_chain(c,NULL)) return 1;
        printf("Case %u (%s): actual eight-block chain and every INT8/INT32 boundary PASS\n",c,full_case_names[c]);
    }
    printf("PASS: backend=%s; 8 complete C chains, 64 block checks, 262144 INT8 and 262144 INT32 boundary values; 0 mismatches; guards unchanged\n",BACKEND_NAME);
#endif
    return 0;
}
