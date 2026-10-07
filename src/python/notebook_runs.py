"""Create an independent saved record for each notebook execution (standard library only)."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def start_run(root, notebook_path, parameters, inputs=()):
    root = Path(root)
    notebook = root / notebook_path
    run_id = notebook.stem + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = root / 'results/notebook_runs' / run_id
    destination.mkdir(parents=True, exist_ok=False)
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True)
    status = subprocess.run(['git', 'status', '--porcelain'], cwd=root, capture_output=True, text=True)
    versions = {}
    for name in ['torch', 'transformers', 'torchao', 'litert-torch', 'ai-edge-litert', 'numpy']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    manifest = {'schema_version': 1, 'run_id': run_id, 'notebook': notebook_path,
                'status': 'started', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
                'parameters': parameters, 'git_commit': git.stdout.strip() if git.returncode == 0 else None,
                'git_status_porcelain': status.stdout if status.returncode == 0 else None,
                'package_versions': versions,
                'python': sys.version, 'source_files': {}, 'inputs': {}, 'outputs': {}}
    for path in [notebook, *sorted((root / 'src/python').glob('*.py')), root / 'requirements.txt']:
        relative = str(path.relative_to(root))
        manifest['source_files'][relative] = _hash(path)
        copied = destination / 'snapshot' / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, copied)
    for path in inputs:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest['inputs'][str(path.relative_to(root))] = _hash(path)
    (destination / 'run.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Notebook run:', run_id)
    return destination


def finish_run(root, run_dir, outputs):
    root, run_dir = Path(root), Path(run_dir)
    path = run_dir / 'run.json'
    manifest = json.loads(path.read_text())
    for relative, expected in manifest['source_files'].items():
        if _hash(root / relative) != expected:
            raise ValueError('Notebook or shared source changed during execution: ' + relative)
    for relative, expected in manifest['inputs'].items():
        if _hash(root / relative) != expected:
            raise ValueError('Input changed during execution: ' + relative)
    for output in outputs:
        output = Path(output)
        manifest['outputs'][str(output.relative_to(root))] = {
            'sha256': _hash(output), 'size_bytes': output.stat().st_size}
    manifest['status'] = 'completed'  # Completion is not a claim about quantization accuracy.
    manifest['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(manifest, indent=2) + '\n')
