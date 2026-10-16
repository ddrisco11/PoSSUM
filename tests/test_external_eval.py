import importlib.util
import io
import pickle
from pathlib import Path
import pytest


def module(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).parents[1]/'scripts'/f'{name}.py')
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_ibc_reader_rejects_executable_globals():
    fetch=module('fetch_external_eval')
    payload=b"cos\nsystem\n(S'echo unsafe'\ntR."
    with pytest.raises(pickle.UnpicklingError,match='Forbidden'):
        fetch.RestrictedTreeReader(io.BytesIO(payload)).load()


def test_ibc_tree_leaf_order():
    fetch=module('fetch_external_eval')
    left,right,root=[fetch.TreeRecord() for _ in range(3)]
    left.word='public';right.word='policy';root.c1=left;root.c2=right
    assert fetch.tree_words(root)==['public','policy']


def test_external_overlap_screen_removes_exact_and_near_duplicates():
    evaluate=module('evaluate_external')
    text='Public investment in schools and affordable health insurance should be available to every family regardless of their income or employment status.'
    source=[{'text':text}]
    external=[{'text':text.upper()},{'text':text+' Today.'},{'text':'Lower corporate taxes and deregulation are necessary to encourage private enterprise, support business owners, and increase competition throughout the economy.'}]
    kept,near=evaluate.screen_overlap(external,source,threshold=.9)
    assert len(kept)==1 and near==1
    assert kept[0]['text'].startswith('Lower corporate')
