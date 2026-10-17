"""Assemble a local inference-only bundle; omit base weights, optimizer state and raw data."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess

from possum.checkpoints import atomic_write, save_json
from possum.option_a import digest


def package(root, output):
    if json.loads((root / 'run.json').read_text())['status'] != 'complete':
        raise ValueError('Main training/evaluation must be complete')
    export = root / 'export'
    for check in ('reload-check.json', 'raw-inference-check.json'):
        if not json.loads((export / check).read_text())['passed']:
            raise ValueError(f'Export check failed: {check}')
    config = json.loads((export / 'config.json').read_text())
    files = [Path('export') / name for name in
             ('config.json', 'self_head.joblib', 'validation.json', 'reload-check.json', 'raw-inference-check.json')]
    files.extend(Path('export') / p.name for p in sorted(export.glob('*.pt')))
    files.extend(Path('bert_adapter') / name for name in ('adapter_config.json', 'adapter_model.safetensors'))
    files.extend(Path('mistral_adapter') / name for name in ('adapter_config.json', 'adapters.safetensors'))
    for name, filename in [('bert', 'adapter_model.safetensors'), ('mistral', 'adapters.safetensors')]:
        info = json.loads((root / f'{name}_adapter/training.json').read_text())
        if digest(root / f'{name}_adapter' / filename) != info['adapter_sha256']:
            raise ValueError('Adapter changed after training validation')
    hashes = {}
    for relative in files:
        source, target = root / relative, output / relative
        expected = digest(source)
        if target.is_symlink() or (target.exists() and digest(target) != expected):
            raise ValueError(f'Existing bundle file differs: {relative}')
        if not target.exists():
            atomic_write(target, lambda stream, source=source: stream.write(source.read_bytes()))
        if digest(target) != expected:
            raise ValueError('Bundle copy verification failed')
        hashes[str(relative)] = dict(sha256=expected, bytes=target.stat().st_size)
    versions = {name: importlib.metadata.version(name) for name in
                ('numpy', 'scipy', 'scikit-learn', 'torch', 'transformers', 'peft', 'mlx', 'mlx-lm', 'gliner2')}
    manifest = dict(format_version=1, model='PoSSUM', title='Political Slant Scoring Using Mixtures',
                    selected=config['selected'], labels={'0': 'Left', '1': 'Right'},
                    status='Research prototype; poor human-text transfer, no validated abstention or production release',
                    repository_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    required_backbones=config['backbones'], tested_runtime=versions,
                    entrypoint='scripts/predict_option_a.py --export <bundle>/export --text <passage>',
                    runtime='Use the PoSSUM source checkout and installed dependencies; pinned base models must be cached separately.',
                    files=hashes, total_bytes=sum(v['bytes'] for v in hashes.values()))
    save_json(output / 'manifest.json', manifest)
    print(json.dumps({'bundle': str(output), 'files': len(files), 'bytes': manifest['total_bytes']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    package(root, args.output.resolve() if args.output else root / 'bundle')
