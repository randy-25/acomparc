"""Export the prepared constants as integer-only C parameter data."""
import math
from pathlib import Path
import numpy as np

LN_FIELDS = ('gamma','beta','square_shift','epsilon_q','rsqrt_numerator',
             'normalize_shift','beta_alignment','output_multiplier','output_shift','output_zero_point')


def block_inventory(bundle):
    index = bundle['block_index']
    prefix, symbol = f'BLOCK{index}', f'block{index}'
    macros, arrays = {}, {}

    def macro(name, value, kind='int'):
        macros[prefix + '_' + name] = (int(value), kind)

    def array(name, values, ctype):
        arrays[symbol + '_' + name] = (ctype, np.asarray(values).reshape(-1))

    for tag in ('ln1', 'ln2'):
        p = bundle[tag]
        array(tag + '_gamma', p['gamma'], 'int32_t')
        array(tag + '_beta', p['beta'], 'int32_t')
        for field, kind in [('square_shift','u'),('epsilon_q','i32'),('rsqrt_numerator','u64'),
                            ('normalize_shift','u'),('output_multiplier','i64'),('output_shift','u'),
                            ('output_zero_point','int')]:
            macro(tag.upper() + '_' + field.upper(), p[field], kind)
        macro(tag.upper() + '_BETA_ALIGNMENT', 1 << 16, 'i64')

    projections = [(name.lower(), name, bundle['qkv'][name]) for name in ('Q','K','V')]
    projections += [('out','OUT_PROJ',bundle['out_residual']),
                    ('mlp_expand','MLP_EXPAND',bundle['mlp_expand']),
                    ('mlp_out','MLP_OUT',bundle['mlp_out'])]
    for name, tag, p in projections:
        array(name + '_weights', p['weights'], 'int8_t')
        if p.get('bias') is not None:
            array(name + '_bias', p['bias'], 'int32_t')
        macro(tag + '_MULTIPLIER', p['multiplier'], 'i64')
        macro(tag + '_OUTPUT_ZERO_POINT', p['output_zero_point'])
        if name in ('q','k','v'):
            macro('QKV_INPUT_ZERO_POINT', p['input_zero_point'])
            macro('QKV_REQUANT_SHIFT', p['shift'], 'u')
        else:
            macro(tag + '_INPUT_ZERO_POINT', p['input_zero_point'])
            macro(tag + '_SHIFT', p['shift'], 'u')
        if name.startswith('mlp_'):
            macro(tag + '_INPUT_FEATURES', p['weights'].shape[1], 'u')
            macro(tag + '_OUTPUT_FEATURES', p['weights'].shape[0], 'u')
            macro(tag + '_WEIGHT_COUNT', p['weights'].size, 'u')

    p = bundle['attention']
    array('softmax_exp_lut', p['exp_lut'], 'int32_t')
    macro('SOFTMAX_EXP_LUT_SIZE', p['exp_lut'].size, 'u')
    macro('SOFTMAX_COORD_MULTIPLIER', p['coordinate_multiplier'], 'i64')
    macro('SOFTMAX_COORD_SHIFT', p['coordinate_shift'], 'u')
    macro('SOFTMAX_PROB_FRAC_BITS', 16, 'u')
    macro('ATTN_CONTEXT_MULTIPLIER', p['context_multiplier'], 'i64')
    macro('ATTN_CONTEXT_SHIFT', p['context_shift'], 'u')
    macro('ATTN_CONTEXT_OUTPUT_ZERO_POINT', p['context_output_zero_point'])
    for tag, key in [('RESIDUAL1','out_residual'),('RESIDUAL2','mlp_out')]:
        p = bundle[key]
        macro(tag + '_BRANCH_MULTIPLIER', p['branch_multiplier'], 'i64')
        macro(tag + '_BRANCH_SHIFT', p['branch_shift'], 'u')
        macro(tag + '_FRAC_BITS', 16, 'u')
    macro('RESIDUAL1_INPUT_ALIGNMENT', bundle['out_residual']['input_alignment'], 'i64')
    macro('RESIDUAL1_INPUT_ZERO_POINT', 0)

    g = bundle['gelu']
    for name in ('half','cube','weighted_cube','argument'):
        p = g[name]
        tag = 'GELU_' + name.upper()
        macro(tag + '_INPUT_ZERO_POINT', g['input_zero_point'] if name=='cube' else p['input_zero_point'])
        if name != 'cube':
            macro(tag + '_CENTERED_CONSTANT', p['centered_constant'], 'i32')
        macro(tag + '_MULTIPLIER', p['multiplier'], 'i64')
        macro(tag + '_SHIFT', p['shift'], 'u')
        macro(tag + '_OUTPUT_ZERO_POINT', p['output_zero_point'])
    p = g['sum']
    for suffix, key, kind in [('LINEAR_ZERO_POINT','linear_zero_point','int'),
                             ('CUBIC_ZERO_POINT','cubic_zero_point','int'),
                             ('LINEAR_MULTIPLIER','linear_multiplier','i64'),
                             ('CUBIC_MULTIPLIER','cubic_multiplier','i64'),
                             ('SHIFT','shift','u'),('OUTPUT_ZERO_POINT','output_zero_point','int')]:
        macro('GELU_ADD_' + suffix, p[key], kind)
    array('gelu_tanh_lut', g['tanh_lut'], 'int32_t')
    macro('GELU_TANH_LUT_SIZE', 256, 'u')
    macro('GELU_TANH_INPUT_ZERO_POINT', g['argument']['output_zero_point'])
    macro('GELU_TANH_FRAC_BITS', 24, 'u')
    macro('GELU_TANH_ONE', 1 << 24, 'i32')
    for field, kind in [('multiplier','i64'),('shift','u'),('output_zero_point','int')]:
        macro('GELU_FACTOR_' + field.upper(), g['factor'][field], kind)
    for suffix,key,kind in [('HALF_ZERO_POINT','half_zero_point','int'),
                           ('FACTOR_ZERO_POINT','factor_zero_point','int'),
                           ('Q31_MULTIPLIER','q31_multiplier','i32'),
                           ('SECOND_SHIFT','second_shift','u'),('ZERO_POINT','output_zero_point','int')]:
        macro('GELU_OUTPUT_' + suffix, g['final'][key], kind)

    macro('INPUT_FRAC_BITS', -int(round(math.log2(bundle['input_scale']))), 'u')
    macro('OUTPUT_FRAC_BITS', -int(round(math.log2(bundle['output_scale']))), 'u')
    for suffix,key,kind in [('MULTIPLIER','multiplier','i64'),('SHIFT','shift','u'),('ZERO_POINT','output_zero_point','int')]:
        macro('OUTPUT_REQUANT_' + suffix, bundle['mlp_out']['boundary'][key], kind)
    return macros, arrays


def write_c_parameters(directory, bundles, profile, overwrite=False):
    directory = Path(directory)
    names = ['integer_parameter_types.h'] + [f'block{i}_params.{ext}' for i in range(8) for ext in ('h','c')]
    if not overwrite and any((directory/name).exists() for name in names):
        raise FileExistsError('C parameters already exist; use --overwrite explicitly')
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'integer_parameter_types.h').write_text('''#ifndef TINYSTORIES_INTEGER_PARAMETER_TYPES_H
#define TINYSTORIES_INTEGER_PARAMETER_TYPES_H
#include <stdint.h>
typedef struct {
    const int32_t *gamma;
    const int32_t *beta;
    unsigned int square_shift;
    int32_t epsilon_q;
    uint64_t rsqrt_numerator;
    unsigned int normalize_shift;
    int64_t beta_alignment;
    int64_t output_multiplier;
    unsigned int output_shift;
    int32_t output_zero_point;
} LayerNormI8Params;
#endif
''')
    for bundle in bundles:
        index = bundle['block_index']
        prefix, symbol = f'BLOCK{index}', f'block{index}'
        macros, arrays = block_inventory(bundle)
        banner = (f'/* Generated integer parameter data; {profile["profile"]}.\n'
                  f' * Model SHA-256: {profile["model_sha256"][str(index)]}\n'
                  ' * C kernel/orchestration implementation is a separate next step. */\n')
        header = [banner, f'#ifndef {prefix}_PARAMS_H\n#define {prefix}_PARAMS_H\n',
                  '#include <stdint.h>\n#include "integer_parameter_types.h"\n']
        for name,(value,kind) in macros.items():
            if kind=='u':
                rendered=f'{value}u'
            elif kind in ('i32','i64','u64'):
                rendered={'i32':'INT32_C','i64':'INT64_C','u64':'UINT64_C'}[kind]+f'({value})'
            else:
                rendered=f'({value})' if value<0 else str(value)
            header.append(f'#define {name} {rendered}\n')
        source=[banner, f'#include "{symbol}_params.h"\n']
        for name,(ctype,values) in arrays.items():
            header.append(f'extern const {ctype} {name}[{values.size}];\n')
            lines=['    '+', '.join(str(int(v)) for v in values[pos:pos+16])+',' for pos in range(0,values.size,16)]
            source.append(f'const {ctype} {name}[{values.size}] = {{\n'+'\n'.join(lines)+'\n};\n')
        for layer in ('ln1','ln2'):
            header.append(f'extern const LayerNormI8Params {symbol}_{layer}_params;\n')
            fields=[]
            for field in LN_FIELDS:
                value=f'{symbol}_{layer}_{field}' if field in ('gamma','beta') else f'{prefix}_{layer.upper()}_{field.upper()}'
                fields.append(f'    .{field} = {value},')
            source.append(f'const LayerNormI8Params {symbol}_{layer}_params = {{\n'+'\n'.join(fields)+'\n};\n')
        header.append('#endif\n')
        (directory/f'{symbol}_params.h').write_text(''.join(header))
        (directory/f'{symbol}_params.c').write_text('\n'.join(source))
