import json
from importlib.resources import files


def resolve(model_id, revision=None):
    if revision:
        return revision
    known=json.loads(files('possum').joinpath('model_revisions.json').read_text())
    if model_id in known:
        return known[model_id]
    from huggingface_hub import model_info
    return model_info(model_id).sha
