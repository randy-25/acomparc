"""JSON/NPZ parameter archives; no pickle or notebook execution."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def save_archive(directory, bundles, profile, overwrite=False):
    directory = Path(directory)
    paths = [directory / 'parameters.json', directory / 'parameters.npz']
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError('Parameter archive already exists; use --overwrite explicitly')
    arrays, descriptions, identical = {}, {}, {}

    def pack(value):
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in 'iub':
                raise ValueError('Only integer parameter arrays may be archived')
            array = np.ascontiguousarray(value.astype(value.dtype.newbyteorder('<')))
            digest = array_digest(array)
            identity = (array.dtype.str, array.shape, digest)
            if identity not in identical:
                key = f'a{len(arrays):04d}'
                identical[identity] = key
                arrays[key] = array
                descriptions[key] = {'dtype': array.dtype.str, 'shape': list(array.shape), 'sha256': digest}
            return {'array': identical[identity]}
        if isinstance(value, dict):
            return {key: pack(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [pack(item) for item in value]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Non-finite parameter metadata')
        if value is None or isinstance(value, (int, float, str, bool)):
            return value
        raise TypeError(f'Unsupported parameter type: {type(value)}')

    packed = pack(bundles)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(paths[1], **arrays)
    manifest = {
        'schema_version': 1, 'profile': profile, 'array_file': 'parameters.npz',
        'array_file_sha256': hashlib.sha256(paths[1].read_bytes()).hexdigest(),
        'arrays': descriptions, 'blocks': packed,
    }
    paths[0].write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def load_archive(path):
    path = Path(path)
    if path.is_dir():
        path = path / 'parameters.json'
    manifest = json.loads(path.read_text())
    if manifest.get('schema_version') != 1 or manifest.get('array_file') != 'parameters.npz':
        raise ValueError('Unsupported parameter archive')
    array_path = path.parent / 'parameters.npz'
    if hashlib.sha256(array_path.read_bytes()).hexdigest() != manifest['array_file_sha256']:
        raise ValueError('Parameter archive checksum mismatch')
    with np.load(array_path, allow_pickle=False) as archive:
        if set(archive.files) != set(manifest['arrays']):
            raise ValueError('Parameter archive array set mismatch')
        arrays = {}
        for key, description in manifest['arrays'].items():
            array = archive[key].copy()
            if array.dtype.str != description['dtype'] or list(array.shape) != description['shape']:
                raise ValueError(f'Parameter array type/shape mismatch: {key}')
            if array_digest(array) != description['sha256']:
                raise ValueError(f'Parameter array checksum mismatch: {key}')
            array.setflags(write=False)
            arrays[key] = array

    def unpack(value):
        if isinstance(value, dict):
            if set(value) == {'array'}:
                return arrays[value['array']]
            return {key: unpack(item) for key, item in value.items()}
        if isinstance(value, list):
            return [unpack(item) for item in value]
        return value

    bundles = unpack(manifest['blocks'])
    from .parameters import boundary_specs
    boundaries = boundary_specs(manifest['profile'])
    if len(bundles) != 8 or [b['block_index'] for b in bundles] != list(range(8)):
        raise ValueError('Expected blocks 0..7')
    for i,bundle in enumerate(bundles):
        if (bundle['input_scale'] != boundaries[i]['input']['scale']
                or bundle['output_scale'] != boundaries[i]['output']['scale']):
            raise ValueError('Archive scales differ from its profile')
    for i in range(7):
        if bundles[i]['output_scale'] != bundles[i + 1]['input_scale']:
            raise ValueError('Incompatible adjacent block scales')
    return bundles, manifest['profile']
