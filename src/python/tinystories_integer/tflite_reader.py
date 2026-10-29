"""Read the inspected TFLite family using the official generated schema."""
from dataclasses import dataclass, field
import hashlib
import importlib
import importlib.metadata
import math
from pathlib import Path
import numpy as np

GRAPH = (
    ("MEAN", (0, 10), 31), ("RESHAPE", (31, 11), 32),
    ("SQUARED_DIFFERENCE", (0, 32), 33), ("MEAN", (33, 10), 34),
    ("ADD", (34, 2), 35), ("RSQRT", (35,), 36),
    ("MUL", (36, 31), 37), ("SUB", (5, 37), 38),
    ("RESHAPE", (36, 11), 39), ("MUL", (0, 39), 40),
    ("RESHAPE", (38, 11), 41), ("ADD", (40, 41), 42),
    ("MUL", (42, 6), 43), ("ADD", (43, 7), 44),
    ("QUANTIZE", (44,), 45), ("FULLY_CONNECTED", (45, 27), 46),
    ("FULLY_CONNECTED", (45, 25), 47), ("FULLY_CONNECTED", (45, 26), 48),
    ("RESHAPE", (46, 13), 49), ("TRANSPOSE", (49, 14), 50),
    ("RESHAPE", (47, 13), 51), ("RESHAPE", (48, 13), 52),
    ("TRANSPOSE", (52, 14), 53), ("TRANSPOSE", (51, 1), 54),
    ("RESHAPE", (50, 15), 55), ("DEQUANTIZE", (55,), 56),
    ("RESHAPE", (54, 16), 57), ("DEQUANTIZE", (57,), 58),
    ("BATCH_MATMUL", (56, 58), 59), ("RESHAPE", (59, 17), 60),
    ("SELECT_V2", (21, 60, 18), 61), ("SOFTMAX", (61,), 62),
    ("RESHAPE", (62, 19), 63), ("RESHAPE", (53, 15), 64),
    ("DEQUANTIZE", (64,), 65), ("BATCH_MATMUL", (63, 65), 66),
    ("QUANTIZE", (66,), 67), ("RESHAPE", (67, 20), 68),
    ("TRANSPOSE", (68, 14), 69), ("RESHAPE", (69, 12), 70),
    ("FULLY_CONNECTED", (70, 28, 24), 71), ("DEQUANTIZE", (71,), 72),
    ("ADD", (72, 0), 73), ("MEAN", (73, 10), 74),
    ("RESHAPE", (74, 11), 75), ("SQUARED_DIFFERENCE", (73, 75), 76),
    ("MEAN", (76, 10), 77), ("ADD", (77, 2), 78),
    ("RSQRT", (78,), 79), ("MUL", (79, 74), 80),
    ("SUB", (5, 80), 81), ("RESHAPE", (79, 11), 82),
    ("MUL", (73, 82), 83), ("RESHAPE", (81, 11), 84),
    ("ADD", (83, 84), 85), ("MUL", (85, 8), 86),
    ("ADD", (86, 9), 87), ("QUANTIZE", (87,), 88),
    ("FULLY_CONNECTED", (88, 29, 23), 89), ("DEQUANTIZE", (89,), 90),
    ("MUL", (89, 91), 92), ("POW", (90, 3), 93),
    ("QUANTIZE", (93,), 94), ("MUL", (94, 95), 96),
    ("ADD", (89, 96), 97), ("MUL", (97, 98), 99),
    ("DEQUANTIZE", (99,), 100), ("TANH", (100,), 101),
    ("ADD", (101, 4), 102), ("QUANTIZE", (102,), 103),
    ("MUL", (92, 103), 104), ("FULLY_CONNECTED", (104, 30, 22), 105),
    ("DEQUANTIZE", (105,), 106), ("ADD", (73, 106), 107),
)

class ExportError(ValueError):
    pass

def require(condition, message):
    if not condition:
        raise ExportError(message)

@dataclass
class Tensor:
    actual_id: int
    name: str
    shape: tuple
    dtype: np.dtype
    scales: np.ndarray = field(default_factory=lambda: np.empty(0, np.float64))
    zero_points: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    quantized_dimension: int = 0
    values: np.ndarray | None = None

class GraphModel:
    def __init__(self, tensors, mapping, sha256, backend, source_name):
        self.tensors = tensors
        self.mapping = mapping
        self.sha256 = sha256
        self.backend = backend
        self.source_name = source_name

    def tensor(self, role):
        return self.tensors[self.mapping[role]]

    def constant(self, role, dtype=None, shape=None):
        tensor = self.tensor(role)
        require(tensor.values is not None, f"T{role} role: not a serialized constant")
        if dtype is not None:
            require(tensor.dtype == np.dtype(dtype), f"T{role} role: expected {np.dtype(dtype)}, got {tensor.dtype}")
        if shape is not None:
            require(tensor.shape == tuple(shape), f"T{role} role: expected shape {shape}, got {tensor.shape}")
        require(np.all(np.isfinite(tensor.values)), f"T{role} role: non-finite constant")
        return tensor.values.copy()

    def quant(self, role):
        t = self.tensor(role)
        require(t.dtype == np.dtype('int8'), f"T{role} role: expected signed INT8")
        require(t.scales.size == t.zero_points.size == 1,
                f"T{role} role: expected per-tensor activation quantization")
        scale, zp = float(t.scales[0]), int(t.zero_points[0])
        require(math.isfinite(scale) and scale > 0 and -128 <= zp <= 127,
                f"T{role} role: invalid INT8 quantization")
        return scale, zp

def load_schema():
    for module, distribution in (
        ('ai_edge_litert.schema_py_generated', 'ai-edge-litert'),
        ('tensorflow.lite.python.schema_py_generated', 'tensorflow'),
    ):
        try:
            schema = importlib.import_module(module)
        except ImportError:
            continue
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = 'unknown'
        return schema, f'{distribution} {version}'
    raise ExportError('Cannot import the TFLite schema. Use the notebook environment containing ai-edge-litert, or install numpy and ai-edge-litert there.')

def read_option(schema, operator, kind):
    expected_type = getattr(schema.BuiltinOptions, kind)
    require(operator.BuiltinOptionsType() == expected_type,
            f"Expected {kind} builtin options")
    table = operator.BuiltinOptions()
    require(table is not None, f"Missing {kind} options")
    result = getattr(schema, kind)()
    result.Init(table.Bytes, table.Pos)
    return result

def validate_options(schema, operator, name):
    if name in ('FULLY_CONNECTED', 'ADD', 'MUL', 'SUB'):
        kind = {'FULLY_CONNECTED': 'FullyConnectedOptions', 'ADD': 'AddOptions',
                'MUL': 'MulOptions', 'SUB': 'SubOptions'}[name]
        option = read_option(schema, operator, kind)
        require(option.FusedActivationFunction() == schema.ActivationFunctionType.NONE,
                f'{name}: fused activation is unsupported')
        if name == 'FULLY_CONNECTED':
            require(option.WeightsFormat() == schema.FullyConnectedOptionsWeightsFormat.DEFAULT,
                    'Packed/shuffled FC weights are unsupported')
    elif name == 'BATCH_MATMUL':
        option = read_option(schema, operator, 'BatchMatMulOptions')
        require(not option.AdjX() and not option.AdjY(), 'Unexpected BATCH_MATMUL adjoint layout')
    elif name == 'SOFTMAX':
        require(read_option(schema, operator, 'SoftmaxOptions').Beta() == 1.0,
                'Only softmax beta=1 matches the current C kernel')
    elif name == 'MEAN':
        require(not read_option(schema, operator, 'ReducerOptions').KeepDims(),
                'Unexpected MEAN keep_dims layout')

def load_model(path):
    raw = path.read_bytes()
    require(len(raw) >= 8 and raw[4:8] == b'TFL3', 'Input is not a TFLite FlatBuffer')
    schema, backend = load_schema()
    model = schema.Model.GetRootAsModel(raw, 0)
    require(model.Version() == 3, 'Only TFLite schema version 3 is supported')
    require(model.SubgraphsLength() == 1, 'Expected one subgraph')
    graph = model.Subgraphs(0)
    require(graph.InputsLength() == graph.OutputsLength() == 1, 'Expected one block input and output')
    require(graph.OperatorsLength() == len(GRAPH),
            f'Expected {len(GRAPH)} operations; found {graph.OperatorsLength()}. Inspect this graph before export.')
    produced = {int(graph.Operators(i).Outputs(j))
                for i in range(graph.OperatorsLength())
                for j in range(graph.Operators(i).OutputsLength())}
    mapping = {0: int(graph.Inputs(0))}
    for index, (expected_name, roles, output_role) in enumerate(GRAPH):
        op = graph.Operators(index)
        opcode = model.OperatorCodes(op.OpcodeIndex())
        code = max(int(opcode.BuiltinCode()), int(opcode.DeprecatedBuiltinCode()))
        require(code == getattr(schema.BuiltinOperator, expected_name),
                f'Op {index}: expected {expected_name}; unexpected graph family')
        inputs = [int(op.Inputs(i)) for i in range(op.InputsLength()) if op.Inputs(i) >= 0]
        require(len(inputs) == len(roles) and op.OutputsLength() == 1,
                f'Op {index}: unexpected input/output count')
        for role, actual in zip(roles, inputs):
            if role in mapping:
                require(mapping[role] == actual, f'Op {index}: unexpected connectivity for T{role} role')
            else:
                require(actual not in produced and actual != mapping[0],
                        f'Op {index}: expected a static constant for T{role} role')
                mapping[role] = actual
        mapping[output_role] = int(op.Outputs(0))
        validate_options(schema, op, expected_name)
    require(int(graph.Outputs(0)) == mapping[107], 'Unexpected graph output')

    dtypes = {schema.TensorType.INT8: np.dtype('i1'), schema.TensorType.INT32: np.dtype('<i4'),
              schema.TensorType.FLOAT32: np.dtype('<f4'), schema.TensorType.BOOL: np.dtype('?')}
    tensors = {}
    for actual in sorted(set(mapping.values())):
        t = graph.Tensors(actual)
        require(t.Type() in dtypes, f'Tensor {actual}: unsupported dtype')
        require(not t.IsVariable() and t.Sparsity() is None, f'Tensor {actual}: variable/sparse storage unsupported')
        shape = tuple(int(t.Shape(i)) for i in range(t.ShapeLength()))
        require(all(n > 0 for n in shape), f'Tensor {actual}: unsupported shape {shape}')
        if hasattr(t, 'ShapeSignatureLength'):
            require(all(t.ShapeSignature(i) >= 0 for i in range(t.ShapeSignatureLength())),
                    f'Tensor {actual}: dynamic shapes unsupported')
        q = t.Quantization()
        scales = np.array([q.Scale(i) for i in range(q.ScaleLength())], np.float64) if q else np.empty(0, np.float64)
        zps = np.array([q.ZeroPoint(i) for i in range(q.ZeroPointLength())], np.int64) if q else np.empty(0, np.int64)
        values = None
        if actual not in produced and actual != mapping[0]:
            buffer = model.Buffers(t.Buffer())
            data = bytes(buffer.DataAsNumpy()) if buffer.DataLength() else b''
            count = math.prod(shape)
            require(len(data) == count * dtypes[t.Type()].itemsize,
                    f'Tensor {actual}: missing, external, or incorrectly sized constant buffer')
            values = np.frombuffer(data, dtype=dtypes[t.Type()]).reshape(shape).copy()
        name = t.Name().decode('utf-8', errors='replace') if t.Name() else ''
        tensors[actual] = Tensor(actual, name, shape, dtypes[t.Type()], scales, zps,
                                 int(q.QuantizedDimension()) if q else 0, values)
    return GraphModel(tensors, mapping, hashlib.sha256(raw).hexdigest(), backend, path.name)

def validate_layout(model):
    shapes = {0: (1, 32, 128), 45: (1, 32, 128), 46: (1, 32, 128), 47: (1, 32, 128), 48: (1, 32, 128), 55: (16, 32, 8), 57: (16, 8, 32), 59: (16, 32, 32), 63: (16, 32, 32), 64: (16, 32, 8), 67: (16, 32, 8), 70: (1, 32, 128), 73: (1, 32, 128), 88: (1, 32, 128), 89: (1, 32, 512), 104: (1, 32, 512), 105: (1, 32, 128), 107: (1, 32, 128)}
    for role, shape in shapes.items():
        require(model.tensor(role).shape == shape, f'T{role} role: expected {shape}, got {model.tensor(role).shape}')
    require(model.tensor(0).dtype == model.tensor(107).dtype == np.dtype('float32'), 'Expected the original mixed TFLite graph with FP32 block boundaries')
    constants = {1: (0, 2, 3, 1), 11: (1, 32, 1), 12: (1, 32, 128), 13: (1, 32, 16, 8), 14: (0, 2, 1, 3), 15: (16, 32, 8), 16: (16, 8, 32), 17: (1, 16, 32, 32), 19: (16, 32, 32), 20: (1, 16, 32, 8)}
    for role, expected in constants.items():
        values = model.constant(role, np.int32).reshape(-1)
        require(tuple((int(v) for v in values)) == expected, f'T{role} role: unexpected shape/permutation constant')
    mean_axes = tuple((int(v) for v in model.constant(10, np.int32).reshape(-1)))
    input_shape = model.tensor(0).shape
    rank = len(input_shape)
    require(all((-rank <= axis < rank for axis in mean_axes)), f'MEAN axes {mean_axes} are outside input rank {rank}')
    normalized_axes = tuple((axis % rank for axis in mean_axes))
    require(len(set(normalized_axes)) == len(normalized_axes) and set(normalized_axes) in ({rank - 1}, {0, rank - 1}) and (input_shape[0] == 1), f'MEAN axes {mean_axes} must reduce hidden features and may also reduce the singleton batch axis; token-axis reduction is unsupported')
    mask = model.constant(21, np.bool_, (1, 1, 32, 32))
    require(np.array_equal(mask[0, 0], np.tri(32, dtype=bool)), 'Expected a causal attention mask')
    masked = model.constant(18, np.float32)
    require(masked.size == 1 and float(masked.reshape(-1)[0]) < -1e+20, 'Unexpected masked-score constant')
    for role, expected in ((3, 3.0), (4, 1.0), (5, 0.0)):
        value = model.constant(role, np.float32)
        require(value.size == 1 and float(value.reshape(-1)[0]) == expected, f'T{role} role: unexpected arithmetic constant')
    for roles in ((46, 49, 50, 55), (47, 51, 54, 57), (48, 52, 53, 64), (67, 68, 69, 70)):
        expected = model.quant(roles[0])
        require(all((model.quant(r) == expected for r in roles)), 'Layout operation changes quantization')

class ConstantView:
    def __init__(self, graph):
        self.graph = graph

    def get_tensor(self, actual_id):
        tensor = self.graph.tensors[int(actual_id)]
        require(tensor.values is not None, f"Tensor {actual_id} is not a constant")
        return tensor.values.copy()


def model_view(graph):
    """Notebook-shaped read-only metadata view; no simulator inference."""
    return {
        "interpreter": ConstantView(graph),
        "tensors": {
            actual: {
                "index": actual, "name": tensor.name,
                "shape": np.array(tensor.shape, dtype=np.int32), "dtype": tensor.dtype,
                "quantization_parameters": {
                    "scales": tensor.scales, "zero_points": tensor.zero_points,
                    "quantized_dimension": tensor.quantized_dimension,
                },
            }
            for actual, tensor in graph.tensors.items()
        },
        "ops": [
            {"index": index, "op_name": name,
             "inputs": np.array([graph.mapping[r] for r in inputs],dtype=np.int32),
             "outputs": np.array([graph.mapping[output]],dtype=np.int32)}
            for index,(name,inputs,output) in enumerate(GRAPH)
        ],
        "sha256": graph.sha256,
        "graph": graph,
    }
