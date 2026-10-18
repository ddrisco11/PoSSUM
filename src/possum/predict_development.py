"""Run a verified development export without hard-coding its version directory."""
import argparse
import json
from pathlib import Path

from .improvement import digest


def resolve_bundle(root, variant='selected'):
    root = Path(root)
    if variant not in {'selected', 'compact', 'single_expert'}:
        raise ValueError('Unknown model variant')
    frozen = root / 'selection-freeze.json'
    if frozen.exists():
        records = json.loads(frozen.read_text())['bundles']
        if variant not in records:
            raise ValueError(f'Variant {variant} was not included in the frozen evaluation')
        record = records[variant]
        bundle = Path(record['path'])
        if digest(bundle / 'model.json') != record['manifest_sha256']:
            raise ValueError('Frozen model manifest changed')
    elif variant == 'selected':
        bundle = Path(json.loads((root / 'export.json').read_text())['path'])
    else:
        records = json.loads((root / 'final-reference-bundles.json').read_text())
        if variant not in records:
            raise ValueError(f'No verified {variant} export is available')
        bundle = Path(records[variant])
    if not json.loads((bundle / 'verification.json').read_text()).get('passed'):
        raise ValueError('Export has not passed reload verification')
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--variant', choices=['selected', 'compact', 'single_expert'], default='selected')
    parser.add_argument('--text', required=True)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    args = parser.parse_args()
    from .improvement_inference import predict
    print(json.dumps(predict(resolve_bundle(args.run, args.variant), [args.text], args.device), indent=2))


if __name__ == '__main__':
    main()
