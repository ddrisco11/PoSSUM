import json
from pathlib import Path
from importlib.resources import files


def resolve(model_id, revision=None):
    if revision:
        return revision
    known=json.loads(files('possum').joinpath('model_revisions.json').read_text())
    if model_id in known:
        return known[model_id]
    from huggingface_hub import model_info
    return model_info(model_id).sha


def local_snapshot(model_id, revision):
    """Find a complete pinned snapshot in configured or standard local caches."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE
    from huggingface_hub.errors import LocalEntryNotFoundError
    caches = dict.fromkeys([Path(HF_HUB_CACHE).resolve(), Path.home() / '.cache/huggingface/hub'])
    for cache in caches:
        try:
            path = Path(snapshot_download(model_id, revision=revision, cache_dir=str(cache), local_files_only=True))
        except LocalEntryNotFoundError:
            continue
        if not (path / 'config.json').is_file():
            continue
        index = path / 'model.safetensors.index.json'
        weights = set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
        if weights and all((path / weight).is_file() for weight in weights):
            return str(path)
    raise FileNotFoundError(f'Complete local snapshot unavailable: {model_id}@{revision}')
