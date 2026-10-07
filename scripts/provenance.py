"""Small standard-library helpers for experiment provenance."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


def now():
    return datetime.now(timezone.utc).isoformat()


def identifier(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError("Use letters, numbers, underscores, or hyphens for IDs.")
    return value


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def record(path):
    path = Path(path)
    return {'sha256': digest(path), 'size_bytes': path.stat().st_size}


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def git_state(root):
    def git(*args):
        result = subprocess.run(['git', *args], cwd=root, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    try:
        return {'commit': git('rev-parse', 'HEAD'),
                'status_porcelain': git('status', '--porcelain'),
                'diff_stat': git('diff', 'HEAD', '--stat')}
    except FileNotFoundError:
        return {'commit': None, 'status_porcelain': None}


def root_file(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('File path must be inside the project: ' + str(relative))
    return path
