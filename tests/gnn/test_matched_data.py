import copy
import hashlib
import pickle
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from models.gnn.matched.data import checked_targets, select_members, extract_program, upstream
from models.gnn.data.looperset import normalize_schedule
from models.gnn.graph import collate_records
from models.gnn.hetero_data import tensor_template, hetero_batch
from models.gnn.efficient_model import tensor_batch
from models.gnn.factory import build_model
from .fixtures import program


def raw_fixture():
    raw=program('fixture')
    raw['node_name']='test-machine'
    raw['schedules_list']=[normalize_schedule(raw['program_annotation'],s) | {'execution_times':s['execution_times']} for s in raw['schedules_list']]
    def tree(name):
        node=raw['program_annotation']['iterators'][name]
        return dict(loop_name=name,computations_list=node['computations_list'],child_list=[tree(c) for c in node['child_iterators']])
    roots={'roots':[tree(n) for n,v in raw['program_annotation']['iterators'].items() if v['parent_iterator'] is None]}
    for schedule in raw['schedules_list']: schedule['tree_structure']=copy.deepcopy(roots)
    u=upstream(); template=u.get_representation_template(raw)
    row=dict(name='fixture',schedule_ids=[0,1],schedules=2,footprint=u.get_tree_footprint(template[0]),
             target_fp32_sha256=hashlib.sha256(np.array([1,2],dtype='<f4').tobytes()).hexdigest())
    return raw,row


def test_membership_targets_and_rejection():
    raw,row=raw_fixture()
    assert checked_targets(raw,row).tolist()==[1,2]
    for bad in (float('nan'),0,-1):
        changed=copy.deepcopy(raw);changed['schedules_list'][0]['execution_times']=[bad]
        with pytest.raises(ValueError):checked_targets(changed,row)
    changed=copy.deepcopy(raw);changed['initial_execution_time']=11
    with pytest.raises(ValueError,match='target mismatch'):checked_targets(changed,row)
    repeated={**row,'schedule_ids':[0,0]}
    with pytest.raises(ValueError):checked_targets(raw,repeated)


def test_footprint_filter_exact_boundary():
    rows={'a':dict(footprint='x',schedules=199),'b':dict(footprint='y',schedules=200),
          'c':dict(footprint='z',schedules=1),'d':dict(footprint='z',schedules=1)}
    assert set(select_members(rows,2))=={'b','c','d'}
    assert set(select_members(rows,0))==set(rows)


def test_pinned_features_and_graph_additions_agree():
    raw,row=raw_fixture(); u=upstream()
    record,info=extract_program('fixture',raw,row,u)
    template=u.get_representation_template(raw)
    for i,s in enumerate(raw['schedules_list']):
        c,l,e=u.get_schedule_representation(raw['program_annotation'],s,*template[1:])
        assert torch.equal(record['comps'][i],c[0])
        assert torch.equal(record['loops'][i],l[0])
        assert torch.equal(record['expr'],e[0])
    assert info['samples']==2
    base=collate_records([(record,0),(record,1)])
    graph=hetero_batch(base,SimpleNamespace(get=lambda _:tensor_template(record['topology'])))
    model=build_model('heterogat',efficient=True).eval()
    predicted=model(tensor_batch(graph));assert torch.isfinite(predicted).all()
    changed=copy.deepcopy(raw);changed['initial_execution_time']*=2
    changed_row={**row,'target_fp32_sha256':hashlib.sha256(np.array([2,4],dtype='<f4').tobytes()).hexdigest()}
    other,_=extract_program('fixture',changed,changed_row,u)
    assert torch.equal(record['comps'],other['comps'])
    assert record['topology']==other['topology']


def test_copy_is_nonoverwriting_and_detects_changed_source(tmp_path):
    from models.gnn.matched.copy_raw import retain
    src=tmp_path/'source.pkl';dest=tmp_path/'copy'/'raw.pkl'
    src.write_bytes(b'original')
    assert retain(src,dest)['sha256']==hashlib.sha256(b'original').hexdigest()
    retain(src,dest)
    src.write_bytes(b'changed')
    with pytest.raises(ValueError,match='refusing to overwrite'):retain(src,dest)
    assert dest.read_bytes()==b'original'


def test_preparation_resume_and_stale_identity(tmp_path):
    import gzip,json
    from models.gnn.matched.data import prepare
    raw,row=raw_fixture()
    source=tmp_path/'raw.pkl';source.write_bytes(pickle.dumps({'fixture':raw}))
    audit=tmp_path/'programs.jsonl.gz'
    with gzip.open(audit,'wt') as stream:stream.write(json.dumps(row)+'\n')
    out=tmp_path/'cache'
    prepare(source,audit,out,workers=1,parts=1,min_functions=0)
    first=(out/'part_000.sqlite').read_bytes()
    prepare(source,audit,out,workers=1,parts=1,min_functions=0)
    assert (out/'part_000.sqlite').read_bytes()==first
    assert json.loads((out/'manifest.json').read_text())['membership']['selected_schedules']==2
    with pytest.raises(ValueError,match='identity changed'):prepare(source,audit,out,workers=1,parts=1,min_functions=2)


def test_archived_baseline_matches_upstream_recursive_class():
    import ast
    from pathlib import Path
    upstream_path=Path(__file__).resolve().parents[2]/'utils/modeling.py'
    archive_path=Path(__file__).resolve().parents[2]/'models/gnn/pact25_archive/modeling.py'
    def definition(path):
        tree=ast.parse(path.read_text())
        return ast.dump(next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Model_Recursive_LSTM_v2'),include_attributes=False)
    assert definition(upstream_path)==definition(archive_path)
