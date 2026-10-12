"""Interpreter access and serializable tensor metadata; no model download."""
import numpy as np

def reference_interpreter(model_path):
    try:
        from ai_edge_litert.interpreter import Interpreter, OpResolverType
    except ImportError:
        from tensorflow.lite import Interpreter
        from tensorflow.lite.experimental import OpResolverType
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing {model_path}. Run notebook 02 or copy your existing export there.")
    interpreter = Interpreter(
        model_path=str(model_path),
        experimental_preserve_all_tensors=True,
        experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
    )
    interpreter.allocate_tensors()
    return interpreter

def tensor_record(tensor):
    q = tensor['quantization_parameters']
    return {
        'index': int(tensor['index']), 'name': tensor['name'],
        'shape': tensor['shape'].tolist(), 'dtype': np.dtype(tensor['dtype']).name,
        'scales': q['scales'].tolist(), 'zero_points': q['zero_points'].tolist(),
        'quantized_dimension': int(q['quantized_dimension']),
    }
