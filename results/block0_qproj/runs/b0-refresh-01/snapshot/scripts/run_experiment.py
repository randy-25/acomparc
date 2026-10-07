#!/usr/bin/env python3
"""Execute a configured experiment and save a unique run with logs and snapshots."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from provenance import digest, git_state, identifier, json_write, now, record, root_file

ROOT = Path(__file__).resolve().parents[1]


def capture(argv, cwd):
    try:
        result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=20)
        return {'command': argv, 'returncode': result.returncode,
                'stdout': result.stdout[:12000], 'stderr': result.stderr[:12000]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'command': argv, 'error': str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment')
    parser.add_argument('--run-id')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--make-var', action='append', default=[], metavar='NAME=VALUE')
    parser.add_argument('--keep-trace', action='store_true', help='Copy the large raw trace into the saved run.')
    args = parser.parse_args()
    exp_id = identifier(args.experiment)
    exp = ROOT / 'experiments' / exp_id
    config_path = exp / 'experiment.json'
    config = json.loads(config_path.read_text())
    if config.get('id') != exp_id or not config.get('commands'):
        raise ValueError('Fill in the experiment ID and commands before running it.')
    for item in args.make_var:
        key, separator, _ = item.partition('=')
        identifier(key)
        if not separator:
            raise ValueError('--make-var requires NAME=VALUE.')
        if key == 'BUILD_DIR':
            raise ValueError('BUILD_DIR is reserved for unique run directories.')
    run_id = identifier(args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    run_dir = ROOT / 'results' / exp_id / 'runs' / run_id
    build_dir = exp / 'build' / 'runs' / run_id
    values = {'python': sys.executable, 'run_dir': str(run_dir),
              'build_dir': str(build_dir), 'riscv_prefix': os.environ.get('RISCV_PREFIX', 'riscv64-unknown-elf-')}
    def expand(raw):
        command = []
        for part in raw:
            for key, value in values.items():
                part = part.replace('{' + key + '}', value)
            command.append(part)
        if command[0] == 'make':
            command.extend(args.make_var)
        return command
    if args.dry_run:
        if run_dir.exists() or build_dir.exists():
            raise ValueError('Run ID already exists; choose a new ID.')
        for command in config['commands']:
            print(json.dumps(expand(command)))
        return 0
    if build_dir.exists():
        raise ValueError('Build directory already exists; choose a new run ID.')
    run_dir.mkdir(parents=True, exist_ok=False)
    build_dir.mkdir(parents=True, exist_ok=False)
    manifest = {'schema_version': 1, 'experiment_id': exp_id, 'run_id': run_id,
                'origin': 'executed', 'status': 'running', 'started_at_utc': now(),
                'git': git_state(ROOT), 'config': config, 'make_variables': args.make_var,
                'source_files': {}, 'input_files': {}, 'external_files': {}, 'commands': [],
                'outputs': {}, 'build_directory': str(build_dir.relative_to(ROOT))}
    json_write(run_dir / 'run.json', manifest)
    tracked = {}
    def snapshot(relative, bucket, expected=None):
        path = root_file(ROOT, relative)
        info = record(path)
        if expected and info['sha256'] != expected:
            raise ValueError('Input changed from the configured baseline: ' + relative)
        destination = run_dir / 'snapshot' / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        manifest[bucket][relative] = info
        tracked[path] = info['sha256']
    def write_state():
        json_write(run_dir / 'run.json', manifest)
    try:
        snapshot(str(config_path.relative_to(ROOT)), 'source_files')
        for relative in config['source_files']:
            snapshot(relative, 'source_files')
        for item in config['inputs']:
            snapshot(item['path'], 'input_files', item.get('sha256'))
        settings = capture(['make', '-s', 'print-config', 'BUILD_DIR=' + str(build_dir), *args.make_var], exp) if config.get('capture_make_settings') else {'returncode': 0, 'stdout': ''}
        manifest['make_settings_capture'] = settings
        if settings.get('returncode') != 0:
            raise ValueError('Makefile must provide a working print-config target.')
        settings_map = dict(line.split('=', 1) for line in settings['stdout'].splitlines() if '=' in line)
        manifest['make_settings'] = settings_map
        values['riscv_prefix'] = settings_map.get('RISCV_PREFIX', values['riscv_prefix'])
        for filename in (['crt.S', 'syscalls.c', 'test.ld'] if config.get('capture_cvw_runtime') else []):
            path = Path(settings_map['CVW_COMMON']) / filename
            if not path.is_file():
                raise ValueError('Missing CV-Wally runtime input: ' + str(path))
            destination = run_dir / 'snapshot' / 'external' / 'cvw_common' / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            manifest['external_files'][str(path)] = record(path)
            tracked[path] = digest(path)
        for path in (Path(settings_map['CVW_COMMON']).glob('*.h') if config.get('capture_cvw_runtime') else []):
            destination = run_dir / 'snapshot' / 'external' / 'cvw_common' / path.name
            shutil.copy2(path, destination)
            manifest['external_files'][str(path)] = record(path)
            tracked[path] = digest(path)
        manifest['cvw_git'] = git_state(Path(settings_map['CVW_ROOT'])) if config.get('capture_cvw_runtime') else None
        manifest['tools'] = [capture([settings_map.get('HOST_CC', 'gcc'), '--version'], exp),
                             capture([values['riscv_prefix'] + 'gcc', '--version'], exp),
                             capture([settings_map.get('SPIKE', 'spike'), '--help'], exp)]
        manifest['python'] = sys.version
        manifest['python_packages'] = capture([sys.executable, '-m', 'pip', 'freeze'], exp)
        for number, raw in enumerate(config['commands'], 1):
            command = expand(raw)
            log = run_dir / f'{number:02d}-command.log'
            entry = {'argv': command, 'cwd': str(exp), 'started_at_utc': now(), 'log': log.name}
            manifest['commands'].append(entry)
            write_state()
            print('Running:', ' '.join(command), flush=True)
            # Stream logs and screen output; do not buffer a potentially long test in memory.
            with log.open('w') as stream:
                with subprocess.Popen(command, cwd=exp, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True) as process:
                    for line in process.stdout:
                        stream.write(line)
                        stream.flush()
                        print(line, end='', flush=True)
                    entry['returncode'] = process.wait()
            entry['finished_at_utc'] = now()
            if entry['returncode'] != 0:
                raise ValueError(f'Command {number} failed; see {log.name}.')
        changed = [str(p) for p, expected in tracked.items() if not p.is_file() or digest(p) != expected]
        if changed:
            raise ValueError('Inputs/source changed during the run: ' + ', '.join(changed))
        profile = json.loads((run_dir / config['profile_filename']).read_text()) if config.get('profile_filename') else {}
        for key in config.get('required_profile_checks', []):
            if profile.get(key) is not True:
                raise ValueError('Profile validation did not pass: ' + key)
        elf_name = config.get('build_outputs', {}).get('elf')
        if elf_name and profile.get('elf_sha256') is not None and profile.get('elf_sha256') != digest(build_dir / elf_name):
            raise ValueError('Profile references a different ELF.')
        manifest['metrics'] = {key: profile.get(key) for key in [
            'kernel_instructions', 'macs', 'output_elements', 'instructions_per_mac',
            'loads_per_mac', 'inner_loop_percent', 'observed_scalar_macs', 'requantizations']}
        for role, filename in config.get('build_outputs', {}).items():
            output = build_dir / filename
            manifest['outputs'][str(output.relative_to(ROOT))] = record(output)
            if role == 'trace':
                manifest['raw_trace'] = {'path': str(output.relative_to(ROOT)), **record(output),
                                         'included_in_run': args.keep_trace}
                if args.keep_trace:
                    shutil.copy2(output, run_dir / output.name)
            else:
                shutil.copy2(output, run_dir / output.name)
        for path in sorted(run_dir.iterdir()):
            if path.is_file() and path.name != 'run.json':
                manifest['outputs'][path.name] = record(path)
        manifest['status'] = 'passed'
    except (Exception, KeyboardInterrupt) as exc:
        manifest['status'] = 'failed' if not isinstance(exc, KeyboardInterrupt) else 'interrupted'
        manifest['error'] = str(exc)
        print('Run did not complete:', exc, file=sys.stderr)
    finally:
        manifest['finished_at_utc'] = now()
        write_state()
    print('Saved run:', run_dir)
    return 0 if manifest['status'] == 'passed' else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print('Run error:', exc, file=sys.stderr)
        sys.exit(1)
