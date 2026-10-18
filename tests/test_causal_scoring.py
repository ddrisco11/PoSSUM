"""Exact candidate scoring without loading a model or requiring Apple hardware."""
import sys
import types
import numpy as np
import pytest
from scipy.special import logsumexp
from possum.components.causal import MLXCausal


def test_single_token_labels_share_prefix_and_keep_full_vocabulary_normalization(monkeypatch):
    core=types.ModuleType('mlx.core')
    core.array=np.array
    core.float32=np.float32
    core.logsumexp=logsumexp
    core.clear_cache=lambda:None
    package=types.ModuleType('mlx');package.core=core
    monkeypatch.setitem(sys.modules,'mlx',package)
    monkeypatch.setitem(sys.modules,'mlx.core',core)
    class Tokenizer:
        def encode(self,text,add_special_tokens=True):
            suffix=[1] if text.endswith(' left') else [2] if text.endswith(' right') else []
            return [0,3]+suffix
    calls=[]
    def model(ids):
        calls.append(ids.tolist())
        return np.broadcast_to(np.array([0.,1.,3.,5.]),(1,ids.shape[1],4))
    scorer=MLXCausal.__new__(MLXCausal)
    scorer.model=model;scorer.tokenizer=Tokenizer()
    scores=scorer.score(['A passage'])
    assert calls==[[[0,3]]]
    np.testing.assert_allclose(scores,[[1-logsumexp([0,1,3,5]),3-logsumexp([0,1,3,5])]],rtol=1e-6)
    scorer.max_input_tokens=2
    np.testing.assert_allclose(scorer.score(['A passage']), scores)
    calls.clear()
    scorer.max_input_tokens=1
    with pytest.raises(ValueError, match='segment it explicitly'):
        scorer.score(['A passage'])
    assert not calls  # Reject before any forward pass; no truncation.
    class MultiTokenTokenizer:
        def encode(self,text,add_special_tokens=True):
            suffix=[1,2] if text.endswith((' left',' right')) else []
            return [0,3]+suffix
    scorer.tokenizer=MultiTokenTokenizer();scorer.max_input_tokens=2
    with pytest.raises(ValueError, match='causal prompt limit'):
        scorer.score(['A passage'])
    assert not calls  # The fallback scores full continuations, so they count too.
