import numpy as np
import pytest
from possum.checkpoints import atomic_write, save_scores


def test_interrupted_write_preserves_previous_checkpoint(tmp_path):
    target=tmp_path/'scores.npz'
    save_scores(target,[[1.,2.]],['a'],[.3])
    def interrupted(f):
        f.write(b'incomplete')
        raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError):
        atomic_write(target,interrupted)
    with np.load(target,allow_pickle=False) as data:
        assert data['ids'].tolist()==['a']
        assert data['chunk_seconds'].tolist()==[.3]
    save_scores(target,[[1.,2.],[3.,4.]],['a','b'],[.3,.4])
    with np.load(target,allow_pickle=False) as data:
        assert data['ids'].tolist()==['a','b']
    assert not list(tmp_path.glob('*.tmp'))
