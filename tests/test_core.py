import numpy as np
import pytest
import torch
from scipy.special import softmax
from sklearn.metrics import log_loss
from possum.data import prepare, split_rows
from possum.learning import temperature, Gate, fit_gate, predict_gate, bootstrap_delta
from possum.components.causal import candidate_ids


def row(text, label=0, group='doc:a'):
    return dict(text=text,label=label,source='test',label_kind='human',group_ids=[group])


def test_duplicate_conflict_and_group_union():
    a=' '.join(['healthcare public insurance universal access']*6)
    b=' '.join(['federal fiscal spending budget income regulation']*5)
    data=prepare([row(a,0,'doc:a'),row(a.upper(),0,'doc:b'),row(b,1,'doc:b')])
    assert len(data)==2
    assert data[0]['group']==data[1]['group']
    assert len(prepare([row(a,0),row(a,1),row(b,1)]))==1


def test_reject_missing_provenance():
    r=row(' '.join(['policy']*20));r['group_ids']=[]
    with pytest.raises(ValueError,match='group'):
        prepare([r])


def test_split_never_leaks_groups():
    rows=[dict(group=f'g{g}',label=y) for g in range(40) for y in (0,1)]
    parts=split_rows(rows)
    sets=[set(rows[i]['group'] for i in ids) for ids in parts.values()]
    for i,s in enumerate(sets):
        for t in sets[i+1:]:
            assert not s&t
    assert sorted(np.concatenate(list(parts.values())))==list(range(len(rows)))
    assert len(parts['test'])==16


def test_temperature_reduces_overconfidence():
    logits=np.array([[0,8],[0,8],[0,8],[0,8],[8,0],[8,0],[8,0],[8,0]])
    y=np.array([1,1,1,0,0,0,0,1])
    t=temperature(logits,y)
    assert t>1
    assert log_loss(y,softmax(logits/t,axis=1))<log_loss(y,softmax(logits,axis=1))


def test_gate_convex_and_learns_signal():
    rng=np.random.default_rng(10)
    x=rng.normal(size=(64,384)).astype('float32')
    y=np.tile([0,1],32)
    p=np.column_stack([.1+.8*y,.9-.8*y]).astype('float32')
    gate=fit_gate(x,p,y,fixed=True,epochs=150)
    result=predict_gate(gate,x,p)
    assert np.mean((result>=.5)==y)==1
    w=gate.weights(torch.tensor(x)).detach().numpy()
    assert np.allclose(w.sum(1),1)
    assert np.all(result>=p.min(1)) and np.all(result<=p.max(1))


def test_bootstrap_refuses_one_group():
    result=bootstrap_delta(np.array([0,1]),np.array([.2,.8]),np.array([.3,.7]),np.array(['g','g']))
    assert result['interval'] is None


def test_multitoken_candidate_and_boundary_guard():
    class Tokenizer:
        def encode(self,text,add_special_tokens=True):
            return [0]+[ord(c) for c in text]
    a,b=candidate_ids(Tokenizer(),'prefix',' left')
    assert b[len(a):]==[ord(c) for c in ' left']
    class Bad:
        def encode(self,text,add_special_tokens=True):return [len(text)]
    with pytest.raises(ValueError,match='boundary'):
        candidate_ids(Bad(),'prefix',' left')


def test_self_branch_assigns_own_mass_and_receives_gradients():
    torch.manual_seed(3)
    gate=Gate(2,self_branch=True)
    x=torch.randn(12,384)
    experts=torch.full((12,2),.5)
    y=torch.tensor([0.,1.]*6)
    w=gate.weights(x)
    assert w.shape==(12,3)
    assert torch.allclose(w.sum(1),torch.ones(12))
    own=torch.sigmoid(gate.self_head(x)).squeeze(1)
    expected=(w[:,:2]*experts).sum(1)+w[:,2]*own
    assert torch.allclose(gate(x,experts),expected)
    torch.nn.functional.binary_cross_entropy(gate(x,experts),y).backward()
    assert gate.self_head[0].weight.grad.abs().sum()>0
    assert gate.net[0].weight.grad.abs().sum()>0
