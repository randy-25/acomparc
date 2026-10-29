"""Frozen-profile parameter preparation extracted from the notebook."""
import json
import math
from pathlib import Path
import numpy as np
from .integer_reference import checked_integer
from .tflite_reader import load_model, validate_layout, model_view, require

def tensor_int8_quant(detail):
    if np.dtype(detail['dtype']) != np.dtype(np.int8):
        raise ValueError('Expected an INT8 tensor')
    quant = detail['quantization_parameters']
    if len(quant['scales']) != 1 or len(quant['zero_points']) != 1:
        raise ValueError('Expected per-tensor activation quantization')
    scale = float(quant['scales'][0])
    zero_point = int(quant['zero_points'][0])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid scale')
    if not -128 <= zero_point <= 127:
        raise ValueError('Invalid INT8 zero point')
    return (scale, zero_point)

def read_projection_params(model, op_index, expected_weight_shape):
    interpreter = model['interpreter']
    tensors = model['tensors']
    op = model['ops'][op_index]
    assert op['op_name'] == 'FULLY_CONNECTED'
    inputs = [int(v) for v in op['inputs'] if int(v) >= 0]
    assert len(inputs) in (2, 3)
    assert len(op['outputs']) == 1
    input_id, weight_id = inputs[:2]
    output_id = int(op['outputs'][0])
    input_scale, input_zp = tensor_int8_quant(tensors[input_id])
    output_scale, output_zp = tensor_int8_quant(tensors[output_id])
    weights = interpreter.get_tensor(weight_id).copy()
    assert weights.dtype == np.int8
    assert weights.shape == expected_weight_shape
    output_features, input_features = weights.shape
    assert tuple(tensors[input_id]['shape']) == (1, 32, input_features)
    assert tuple(tensors[output_id]['shape']) == (1, 32, output_features)
    wq = tensors[weight_id]['quantization_parameters']
    weight_scales = np.asarray(wq['scales'], dtype=np.float64)
    weight_zps = np.asarray(wq['zero_points'], dtype=np.int64)
    assert weight_scales.size in (1, output_features)
    assert weight_zps.size in (1, output_features)
    assert np.all(np.isfinite(weight_scales))
    assert np.all(weight_scales > 0)
    assert np.all(weight_scales == weight_scales[0])
    assert np.all(weight_zps == 0)
    if weight_scales.size > 1 or weight_zps.size > 1:
        assert int(wq['quantized_dimension']) == 0
    weight_scale = float(weight_scales[0])
    bias = None
    if len(inputs) == 3:
        bias_id = inputs[2]
        bias = interpreter.get_tensor(bias_id).copy()
        assert bias.dtype == np.int32
        assert bias.shape == (output_features,)
        bq = tensors[bias_id]['quantization_parameters']
        assert len(bq['scales']) in (1, output_features)
        assert len(bq['zero_points']) in (1, output_features)
        assert np.all(np.asarray(bq['zero_points']) == 0)
        assert np.allclose(np.asarray(bq['scales'], dtype=np.float64), input_scale * weight_scale, rtol=1e-05, atol=0)
        if len(bq['scales']) > 1 or len(bq['zero_points']) > 1:
            assert int(bq['quantized_dimension']) == 0
    input_bound = max(abs(-128 - input_zp), abs(127 - input_zp))
    accumulator_bounds = input_bound * np.abs(weights.astype(np.int64)).sum(axis=1)
    if bias is not None:
        accumulator_bounds += np.abs(bias.astype(np.int64))
    assert np.all(accumulator_bounds <= np.iinfo(np.int32).max)
    multiplier = int(np.rint(input_scale * weight_scale / output_scale * (1 << 39)))
    assert 0 < multiplier <= np.iinfo(np.int64).max
    assert (1 << 31) * multiplier + (1 << 38) <= np.iinfo(np.int64).max, 'Projection product/rounding exceeds INT64'
    return {'input_tensor': input_id, 'output_tensor': output_id, 'input_scale': input_scale, 'input_zero_point': input_zp, 'weights': weights, 'weight_scale': weight_scale, 'bias': bias, 'output_scale': output_scale, 'output_zero_point': output_zp, 'multiplier': multiplier, 'shift': 39}

def make_gelu_conversion(model, output_id, source_scale, source_bound, shift):
    output_scale, output_zp = tensor_int8_quant(model['tensors'][output_id])
    assert math.isfinite(source_scale) and source_scale > 0
    assert 0 <= source_bound <= np.iinfo(np.int32).max
    multiplier = int(np.rint(source_scale / output_scale * (1 << shift)))
    assert 0 < multiplier <= np.iinfo(np.int64).max
    assert source_bound * multiplier + (1 << shift - 1) <= np.iinfo(np.int64).max, 'GELU product/rounding exceeds INT64'
    return {'output_tensor': output_id, 'output_scale': output_scale, 'output_zero_point': output_zp, 'multiplier': multiplier, 'shift': shift}

def make_gelu_scalar_mul(model, op_index):
    op = model['ops'][op_index]
    assert op['op_name'] == 'MUL'
    input_id, constant_id = [int(v) for v in op['inputs']]
    output_id = int(op['outputs'][0])
    input_scale, input_zp = tensor_int8_quant(model['tensors'][input_id])
    constant_scale, constant_zp = tensor_int8_quant(model['tensors'][constant_id])
    constant = model['interpreter'].get_tensor(constant_id)
    assert constant.dtype == np.int8 and constant.size == 1
    centered_constant = int(constant.item()) - constant_zp
    input_bound = max(abs(-128 - input_zp), abs(127 - input_zp))
    source_bound = input_bound * abs(centered_constant)
    params = make_gelu_conversion(model, output_id, input_scale * constant_scale, source_bound, shift=38)
    params.update({'input_tensor': input_id, 'input_zero_point': input_zp, 'centered_constant': centered_constant})
    return params

def boundary_specs(profile):
    require(profile.get("profile") == "tinystories-allblocks-int8-v1", "Unsupported numerical profile")
    exponents=profile["boundary_scale_exponents"]
    require(exponents == [-8,-7,-8,-8,-8,-8,-8,-8,-7], "Profile v1 has fixed boundary scales; a different policy needs a new profile")
    require(profile.get('shape') == [1,32,128] and profile.get('blocks') == 8
            and profile.get('heads') == 16 and profile.get('head_dim') == 8
            and profile.get('mlp_features') == 512
            and profile.get('boundary_zero_point') == 0
            and profile.get('residual_scale_exponent') == -16,
            'Unsupported dimensions or fixed-point profile')
    edges=[{"edge":i,"dtype":"int8","scale":2.0**e,"scale_exponent":e,"zero_point":0} for i,e in enumerate(exponents)]
    return [{"input":edges[i],"output":edges[i+1]} for i in range(8)]


def build_from_views(block_models, profile):
    require(len(block_models)==8, "Expected eight blocks")
    block_boundary_specs=boundary_specs(profile)
    ln1_params = {}
    for block_index in range(8):
        model = block_models[block_index]
        interpreter = model['interpreter']
        ops = model['ops']
        boundary = block_boundary_specs[block_index]['input']
        assert boundary['zero_point'] == 0
        input_frac_bits = -int(boundary['scale_exponent'])
        assert input_frac_bits in (7, 8)
        gamma_id = int(ops[12]['inputs'][1])
        beta_id = int(ops[13]['inputs'][1])
        epsilon_id = int(ops[4]['inputs'][1])
        output_id = int(ops[14]['outputs'][0])
        gamma = interpreter.get_tensor(gamma_id).astype(np.float64).reshape(128)
        beta = interpreter.get_tensor(beta_id).astype(np.float64).reshape(128)
        epsilon = float(interpreter.get_tensor(epsilon_id).item())
        if not (np.all(np.isfinite(gamma)) and np.all(np.isfinite(beta)) and math.isfinite(epsilon) and (epsilon > 0)):
            raise ValueError('Invalid LN1 constants')
        quant = model['tensors'][output_id]['quantization_parameters']
        assert len(quant['scales']) == len(quant['zero_points']) == 1
        output_scale = float(quant['scales'][0])
        output_zp = int(quant['zero_points'][0])
        assert math.isfinite(output_scale) and output_scale > 0
        mean_frac_bits = input_frac_bits + 7
        variance_frac_bits = input_frac_bits + 14
        numerator_exponent = variance_frac_bits + 32
        params = {'input_frac_bits': input_frac_bits, 'square_shift': mean_frac_bits, 'normalize_shift': mean_frac_bits, 'epsilon_q': int(np.rint(epsilon * (1 << variance_frac_bits))), 'rsqrt_numerator': 1 << numerator_exponent, 'gamma': checked_integer(np.rint(gamma * (1 << 16)), np.int32, 'gamma'), 'beta': checked_integer(np.rint(beta * (1 << 16)), np.int32, 'beta'), 'output_scale': output_scale, 'output_zero_point': output_zp, 'output_multiplier': int(np.rint(2.0 ** (-32) / output_scale * (1 << 47))), 'output_shift': 47, 'output_tensor': output_id}
        assert 0 < params['epsilon_q'] <= np.iinfo(np.int32).max
        assert 0 < params['output_multiplier'] <= np.iinfo(np.int64).max
        ln1_params[block_index] = params
    qkv_params = {}
    for block_index in range(8):
        model = block_models[block_index]
        interpreter = model['interpreter']
        tensors = model['tensors']
        qkv_params[block_index] = {}
        for name, op_index in (('Q', 15), ('K', 16), ('V', 17)):
            op = model['ops'][op_index]
            assert op['op_name'] == 'FULLY_CONNECTED'
            inputs = [int(v) for v in op['inputs'] if int(v) >= 0]
            assert len(inputs) == 2, 'This Q/K/V path expects no bias'
            input_id, weight_id = inputs
            output_id = int(op['outputs'][0])
            assert input_id == ln1_params[block_index]['output_tensor']
            input_scale, input_zp = tensor_int8_quant(tensors[input_id])
            output_scale, output_zp = tensor_int8_quant(tensors[output_id])
            weights = interpreter.get_tensor(weight_id).copy()
            assert weights.dtype == np.int8
            assert weights.shape == (128, 128)
            weight_quant = tensors[weight_id]['quantization_parameters']
            weight_scales = np.asarray(weight_quant['scales'], dtype=np.float64)
            weight_zps = np.asarray(weight_quant['zero_points'], dtype=np.int64)
            assert weight_scales.size in (1, 128)
            assert weight_zps.size in (1, 128)
            assert np.all(np.isfinite(weight_scales))
            assert np.all(weight_scales > 0)
            assert np.all(weight_zps == 0)
            assert np.all(weight_scales == weight_scales[0]), 'Nonuniform per-channel scales need a different conversion path'
            if weight_scales.size > 1 or weight_zps.size > 1:
                assert int(weight_quant['quantized_dimension']) == 0
            weight_scale = float(weight_scales[0])
            shift = 39
            multiplier = int(np.rint(input_scale * weight_scale / output_scale * (1 << shift)))
            input_bound = max(abs(-128 - input_zp), abs(127 - input_zp))
            accumulator_bounds = input_bound * np.abs(weights.astype(np.int64)).sum(axis=1)
            assert np.all(accumulator_bounds <= np.iinfo(np.int32).max)
            assert 0 < multiplier <= np.iinfo(np.int64).max
            assert (1 << 31) * multiplier + (1 << shift - 1) <= np.iinfo(np.int64).max, 'INT64 product/rounding bound exceeded'
            qkv_params[block_index][name] = {'input_tensor': input_id, 'output_tensor': output_id, 'input_scale': input_scale, 'input_zero_point': input_zp, 'weights': weights, 'weight_scale': weight_scale, 'output_scale': output_scale, 'output_zero_point': output_zp, 'multiplier': multiplier, 'shift': shift}
    attention_params = {}
    for block_index in range(8):
        model = block_models[block_index]
        interpreter = model['interpreter']
        ops = model['ops']
        assert ops[28]['op_name'] == 'BATCH_MATMUL'
        assert ops[31]['op_name'] == 'SOFTMAX'
        assert ops[35]['op_name'] == 'BATCH_MATMUL'
        mask_id = int(ops[30]['inputs'][0])
        mask = interpreter.get_tensor(mask_id)
        np.testing.assert_array_equal(mask, np.tri(32, dtype=bool).reshape(1, 1, 32, 32))
        query = qkv_params[block_index]['Q']
        key = qkv_params[block_index]['K']
        query_zp = query['output_zero_point']
        key_zp = key['output_zero_point']
        query_bound = max(abs(-128 - query_zp), abs(127 - query_zp))
        key_bound = max(abs(-128 - key_zp), abs(127 - key_zp))
        score_bound = 8 * query_bound * key_bound
        assert score_bound <= np.iinfo(np.int32).max
        assert 2 * score_bound <= np.iinfo(np.int32).max
        attention_params[block_index] = {'query_zero_point': query_zp, 'key_zero_point': key_zp, 'score_scale': query['output_scale'] * key['output_scale'], 'score_bound': score_bound, 'score_tensor': int(ops[28]['outputs'][0]), 'probability_tensor': int(ops[31]['outputs'][0]), 'context_tensor': int(ops[35]['outputs'][0]), 'output_tensor': int(ops[39]['outputs'][0])}
    softmax_exp_lut = checked_integer(np.rint(np.exp(-np.arange(4097, dtype=np.float64) / 256.0) * (1 << 30)), np.int32, 'exponential table')
    assert softmax_exp_lut[0] == 1 << 30
    assert np.all(np.diff(softmax_exp_lut.astype(np.int64)) <= 0)
    for block_index in range(8):
        params = attention_params[block_index]
        multiplier = int(np.rint(params['score_scale'] * (1 << 16) * (1 << 24)))
        assert 0 < multiplier <= np.iinfo(np.int64).max
        assert 2 * params['score_bound'] * multiplier + (1 << 23) <= np.iinfo(np.int64).max, 'Softmax coordinate product/rounding exceeds INT64'
        params.update({'coordinate_multiplier': multiplier, 'coordinate_shift': 24, 'exp_lut': softmax_exp_lut})
    for block_index in range(8):
        model = block_models[block_index]
        params = attention_params[block_index]
        value = qkv_params[block_index]['V']
        output_scale, output_zp = tensor_int8_quant(model['tensors'][params['output_tensor']])
        quantized_context_id = int(model['ops'][36]['outputs'][0])
        assert tensor_int8_quant(model['tensors'][quantized_context_id]) == (output_scale, output_zp)
        context_scale = 2.0 ** (-16) * value['output_scale']
        multiplier = int(np.rint(context_scale / output_scale * (1 << 46)))
        value_zp = value['output_zero_point']
        value_bound = max(abs(-128 - value_zp), abs(127 - value_zp))
        context_bound = 32 * (1 << 16) * value_bound
        assert context_bound <= np.iinfo(np.int32).max
        assert 0 < multiplier <= np.iinfo(np.int64).max
        assert context_bound * multiplier + (1 << 45) <= np.iinfo(np.int64).max, 'Context product/rounding exceeds INT64'
        params.update({'value_zero_point': value_zp, 'context_scale': context_scale, 'context_multiplier': multiplier, 'context_shift': 46, 'context_output_scale': output_scale, 'context_output_zero_point': output_zp})
    out_residual_params = {}
    for block_index in range(8):
        model = block_models[block_index]
        interpreter = model['interpreter']
        tensors = model['tensors']
        op = model['ops'][40]
        assert op['op_name'] == 'FULLY_CONNECTED'
        inputs = [int(v) for v in op['inputs'] if int(v) >= 0]
        assert len(inputs) == 3
        input_id, weight_id, bias_id = inputs
        output_id = int(op['outputs'][0])
        input_scale, input_zp = tensor_int8_quant(tensors[input_id])
        output_scale, output_zp = tensor_int8_quant(tensors[output_id])
        weights = interpreter.get_tensor(weight_id).copy()
        bias = interpreter.get_tensor(bias_id).copy()
        assert weights.dtype == np.int8 and weights.shape == (128, 128)
        assert bias.dtype == np.int32 and bias.shape == (128,)
        wq = tensors[weight_id]['quantization_parameters']
        weight_scales = np.asarray(wq['scales'], dtype=np.float64)
        weight_zps = np.asarray(wq['zero_points'], dtype=np.int64)
        assert weight_scales.size in (1, 128)
        assert weight_zps.size in (1, 128)
        assert np.all(np.isfinite(weight_scales))
        assert np.all(weight_scales > 0)
        assert np.all(weight_scales == weight_scales[0])
        assert np.all(weight_zps == 0)
        if weight_scales.size > 1 or weight_zps.size > 1:
            assert int(wq['quantized_dimension']) == 0
        weight_scale = float(weight_scales[0])
        bq = tensors[bias_id]['quantization_parameters']
        assert len(bq['scales']) in (1, 128)
        assert len(bq['zero_points']) in (1, 128)
        assert np.all(np.asarray(bq['zero_points']) == 0)
        assert np.allclose(np.asarray(bq['scales'], dtype=np.float64), input_scale * weight_scale, rtol=1e-05, atol=0)
        if len(bq['scales']) > 1 or len(bq['zero_points']) > 1:
            assert int(bq['quantized_dimension']) == 0
        multiplier = int(np.rint(input_scale * weight_scale / output_scale * (1 << 39)))
        branch_multiplier = int(np.rint(output_scale * (1 << 16) * (1 << 22)))
        boundary = block_boundary_specs[block_index]['input']
        assert boundary['zero_point'] == 0
        input_frac_bits = -int(boundary['scale_exponent'])
        assert input_frac_bits in (7, 8)
        input_alignment = 1 << 16 - input_frac_bits
        input_bound = max(abs(-128 - input_zp), abs(127 - input_zp))
        accumulator_bounds = input_bound * np.abs(weights.astype(np.int64)).sum(axis=1) + np.abs(bias.astype(np.int64))
        assert np.all(accumulator_bounds <= np.iinfo(np.int32).max)
        assert 0 < multiplier <= np.iinfo(np.int64).max
        assert (1 << 31) * multiplier + (1 << 38) <= np.iinfo(np.int64).max
        centered_output_bound = max(abs(-128 - output_zp), abs(127 - output_zp))
        assert 0 < branch_multiplier <= np.iinfo(np.int64).max
        assert centered_output_bound * branch_multiplier + (1 << 21) <= np.iinfo(np.int64).max
        branch_bound = centered_output_bound * branch_multiplier + (1 << 21) >> 22
        assert branch_bound + 128 * input_alignment <= np.iinfo(np.int32).max
        out_residual_params[block_index] = {'input_tensor': input_id, 'output_tensor': output_id, 'residual_tensor': int(model['ops'][42]['outputs'][0]), 'input_zero_point': input_zp, 'weights': weights, 'bias': bias, 'multiplier': multiplier, 'shift': 39, 'output_scale': output_scale, 'output_zero_point': output_zp, 'branch_multiplier': branch_multiplier, 'branch_shift': 22, 'input_alignment': input_alignment}
    ln2_params = {}
    for block_index in range(8):
        model = block_models[block_index]
        interpreter = model['interpreter']
        ops = model['ops']
        assert ops[43]['op_name'] == 'MEAN'
        assert ops[55]['op_name'] == 'MUL'
        assert ops[56]['op_name'] == 'ADD'
        assert ops[57]['op_name'] == 'QUANTIZE'
        gamma_id = int(ops[55]['inputs'][1])
        beta_id = int(ops[56]['inputs'][1])
        epsilon_id = int(ops[47]['inputs'][1])
        input_id = int(ops[43]['inputs'][0])
        output_id = int(ops[57]['outputs'][0])
        assert input_id == out_residual_params[block_index]['residual_tensor']
        gamma = interpreter.get_tensor(gamma_id).astype(np.float64).reshape(128)
        beta = interpreter.get_tensor(beta_id).astype(np.float64).reshape(128)
        epsilon = float(interpreter.get_tensor(epsilon_id).item())
        assert np.all(np.isfinite(gamma))
        assert np.all(np.isfinite(beta))
        assert math.isfinite(epsilon) and epsilon > 0
        output_scale, output_zp = tensor_int8_quant(model['tensors'][output_id])
        params = {'input_frac_bits': 16, 'square_shift': 23, 'normalize_shift': 23, 'epsilon_q': int(np.rint(epsilon * (1 << 30))), 'rsqrt_numerator': 1 << 62, 'gamma': checked_integer(np.rint(gamma * (1 << 16)), np.int32, 'LN2 gamma'), 'beta': checked_integer(np.rint(beta * (1 << 16)), np.int32, 'LN2 beta'), 'output_scale': output_scale, 'output_zero_point': output_zp, 'output_multiplier': int(np.rint(2.0 ** (-32) / output_scale * (1 << 47))), 'output_shift': 47, 'input_tensor': input_id, 'output_tensor': output_id}
        assert 0 < params['epsilon_q'] <= np.iinfo(np.int32).max
        assert 0 < params['output_multiplier'] <= np.iinfo(np.int64).max
        ln2_params[block_index] = params
    mlp_expand_params = {}
    for block_index in range(8):
        params = read_projection_params(block_models[block_index], op_index=58, expected_weight_shape=(512, 128))
        assert params['bias'] is not None
        assert params['input_tensor'] == ln2_params[block_index]['output_tensor']
        mlp_expand_params[block_index] = params
    gelu_params = {}
    for block_index in range(8):
        model = block_models[block_index]
        ops = model['ops']
        tensors = model['tensors']
        half = make_gelu_scalar_mul(model, 60)
        weighted = make_gelu_scalar_mul(model, 63)
        argument = make_gelu_scalar_mul(model, 65)
        assert half['input_tensor'] == mlp_expand_params[block_index]['output_tensor']
        input_scale, input_zp = tensor_int8_quant(tensors[half['input_tensor']])
        input_bound = max(abs(-128 - input_zp), abs(127 - input_zp))
        assert ops[61]['op_name'] == 'POW'
        exponent_id = int(ops[61]['inputs'][1])
        assert float(model['interpreter'].get_tensor(exponent_id).item()) == 3.0
        cube_id = int(ops[62]['outputs'][0])
        cube = make_gelu_conversion(model, cube_id, input_scale ** 3, input_bound ** 3, shift=45)
        assert weighted['input_tensor'] == cube_id
        assert ops[64]['op_name'] == 'ADD'
        sum_id = int(ops[64]['outputs'][0])
        sum_scale, sum_zp = tensor_int8_quant(tensors[sum_id])
        assert argument['input_tensor'] == sum_id
        linear_multiplier = int(np.rint(input_scale / sum_scale * (1 << 31)))
        cubic_multiplier = int(np.rint(weighted['output_scale'] / sum_scale * (1 << 31)))
        assert 0 < linear_multiplier <= np.iinfo(np.int64).max
        assert 0 < cubic_multiplier <= np.iinfo(np.int64).max
        cubic_zp = weighted['output_zero_point']
        cubic_bound = max(abs(-128 - cubic_zp), abs(127 - cubic_zp))
        combined_bound = input_bound * linear_multiplier + cubic_bound * cubic_multiplier
        assert combined_bound + (1 << 30) <= np.iinfo(np.int64).max
        sum_params = {'linear_zero_point': input_zp, 'cubic_zero_point': cubic_zp, 'linear_multiplier': linear_multiplier, 'cubic_multiplier': cubic_multiplier, 'multiplier': 1, 'shift': 31, 'output_tensor': sum_id, 'output_scale': sum_scale, 'output_zero_point': sum_zp}
        gelu_params[block_index] = {'input_scale': input_scale, 'input_zero_point': input_zp, 'half': half, 'cube': cube, 'weighted_cube': weighted, 'sum': sum_params, 'argument': argument}
    for block_index in range(8):
        model = block_models[block_index]
        params = gelu_params[block_index]
        ops = model['ops']
        assert ops[67]['op_name'] == 'TANH'
        assert int(ops[66]['inputs'][0]) == params['argument']['output_tensor']
        one_id = int(ops[68]['inputs'][1])
        assert float(model['interpreter'].get_tensor(one_id).item()) == 1.0
        argument_scale = params['argument']['output_scale']
        argument_zp = params['argument']['output_zero_point']
        argument_codes = np.arange(-128, 128, dtype=np.int32)
        argument_values = (argument_codes.astype(np.float64) - argument_zp) * argument_scale
        tanh_lut = checked_integer(np.rint(np.tanh(argument_values) * (1 << 24)), np.int32, 'tanh table')
        assert np.all((tanh_lut >= -(1 << 24)) & (tanh_lut <= 1 << 24))
        factor_id = int(ops[69]['outputs'][0])
        factor = make_gelu_conversion(model, factor_id, source_scale=2.0 ** (-24), source_bound=1 << 25, shift=47)
        assert [int(v) for v in ops[70]['inputs']] == [params['half']['output_tensor'], factor_id]
        output_id = int(ops[70]['outputs'][0])
        output_scale, output_zp = tensor_int8_quant(model['tensors'][output_id])
        ratio = params['half']['output_scale'] * factor['output_scale'] / output_scale
        assert math.isfinite(ratio) and ratio > 0
        fraction, exponent = math.frexp(ratio)
        q31_multiplier = int(math.floor(fraction * (1 << 31) + 0.5))
        if q31_multiplier == 1 << 31:
            q31_multiplier //= 2
            exponent += 1
        second_shift = -exponent
        assert 0 < q31_multiplier <= np.iinfo(np.int32).max
        assert 0 <= second_shift <= 31
        half_zp = params['half']['output_zero_point']
        factor_zp = factor['output_zero_point']
        half_bound = max(abs(-128 - half_zp), abs(127 - half_zp))
        factor_bound = max(abs(-128 - factor_zp), abs(127 - factor_zp))
        product_bound = half_bound * factor_bound
        assert product_bound <= np.iinfo(np.int32).max
        assert product_bound * q31_multiplier + (1 << 30) <= np.iinfo(np.int64).max
        params.update({'tanh_lut': tanh_lut, 'factor': factor, 'final': {'q31_multiplier': q31_multiplier, 'second_shift': second_shift, 'half_zero_point': half_zp, 'factor_zero_point': factor_zp, 'output_tensor': output_id, 'output_scale': output_scale, 'output_zero_point': output_zp}})
    mlp_out_params = {}
    for block_index in range(8):
        model = block_models[block_index]
        params = read_projection_params(block_models[block_index], 71, expected_weight_shape=(128, 512))
        assert params['bias'] is not None
        assert params['input_tensor'] == gelu_params[block_index]['final']['output_tensor']
        branch_multiplier = int(np.rint(params['output_scale'] * (1 << 16) * (1 << 22)))
        centered_bound = max(abs(-128 - params['output_zero_point']), abs(127 - params['output_zero_point']))
        assert 0 < branch_multiplier <= np.iinfo(np.int64).max
        assert centered_bound * branch_multiplier + (1 << 21) <= np.iinfo(np.int64).max
        boundary = block_boundary_specs[block_index]['output']
        assert boundary['zero_point'] == 0
        output_frac_bits = -int(boundary['scale_exponent'])
        assert output_frac_bits in (7, 8)
        params.update({'branch_multiplier': branch_multiplier, 'branch_shift': 22, 'reference_residual_tensor': int(model['ops'][42]['outputs'][0]), 'reference_output_tensor': int(model['ops'][73]['outputs'][0]), 'boundary': {'multiplier': 1, 'shift': 16 - output_frac_bits, 'output_zero_point': 0, 'scale': boundary['scale']}})
        mlp_out_params[block_index] = params
    return [{
        "block_index":i,
        "input_scale":block_boundary_specs[i]["input"]["scale"],
        "output_scale":block_boundary_specs[i]["output"]["scale"],
        "ln1":ln1_params[i], "qkv":qkv_params[i], "attention":attention_params[i],
        "out_residual":out_residual_params[i], "ln2":ln2_params[i],
        "mlp_expand":mlp_expand_params[i], "gelu":gelu_params[i], "mlp_out":mlp_out_params[i],
    } for i in range(8)]


def load_profile(path):
    profile=json.loads(Path(path).read_text())
    boundary_specs(profile)
    return profile


def load_parameters(models_dir, profile_path):
    profile=load_profile(profile_path)
    directory=Path(models_dir)
    views=[]
    for i in range(8):
        path=directory/f"block{i}_w8a8.tflite"
        graph=load_model(path)
        require(graph.sha256==profile["model_sha256"][str(i)], f"Block {i}: model hash differs from frozen profile")
        validate_layout(graph)
        views.append(model_view(graph))
    return build_from_views(views,profile), profile
