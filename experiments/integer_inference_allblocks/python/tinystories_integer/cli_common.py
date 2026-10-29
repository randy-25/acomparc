"""Small command-line helpers; no host-model dependencies."""
import hashlib
import json
from pathlib import Path
import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[2]
PROJECT = EXPERIMENT.parents[1]
MODELS = PROJECT / 'models/tinystories_3m_blocks_w8a8/20261011T134954617091Z'
FP32 = PROJECT / 'models/tinystories_3m_fp32'
PROFILE = EXPERIMENT / 'profiles/int8_boundaries_v1.json'


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_value(value), indent=2, allow_nan=False)+'\n')


def check_validation_file(path, profile):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != profile['validation_inputs_sha256']:
        raise ValueError('Validation-input archive differs from the frozen profile')


def check_cases(values):
    values = np.asarray(values)
    if values.shape == (1,32,128):
        values = values[None, ...]
    if values.ndim != 4 or values.shape[1:] != (1,32,128) or values.dtype != np.int8:
        raise ValueError('Input must be INT8 [1,32,128] or [cases,1,32,128]')
    if values.shape[0] == 0:
        raise ValueError('At least one input case is required')
    return values


def run_chain(inputs, bundles):
    from .integer_reference import integer_block_forward
    inputs = check_cases(inputs)
    outputs = np.empty((len(inputs),8,1,32,128), np.int8)
    pre_boundary = np.empty(outputs.shape, np.int32)
    diagnostics = []
    for case_index, initial in enumerate(inputs):
        codes = initial.copy()
        stages = []
        for i,bundle in enumerate(bundles):
            result = integer_block_forward(codes,bundle)
            codes = result['output']
            outputs[case_index,i] = codes
            pre_boundary[case_index,i] = result['output_i32']
            stages.append({k:v for k,v in result.items() if k not in ('output','output_i32')})
        diagnostics.append(stages)
        print(f'Completed integer case {case_index}: blocks 0–7', flush=True)
    return outputs, pre_boundary, diagnostics
