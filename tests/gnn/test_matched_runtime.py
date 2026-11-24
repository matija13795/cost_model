import copy
import gzip
import hashlib
import json
import pickle
from pathlib import Path
import sqlite3
import zlib
import numpy as np
import pytest
import torch
from models.gnn.matched.data import prepare
from models.gnn.matched.batches import make_plan, pack, BatchStore
from models.gnn.matched.train import train, model_input
from models.gnn.full_gnn_data import FullGraphBatch
from models.gnn.factory import build_model
from .test_matched_data import raw_fixture


def prepared_batches(tmp_path):
    raw,row=raw_fixture();source=tmp_path/'raw.pkl';source.write_bytes(pickle.dumps({'fixture':raw}))
    audit=tmp_path/'audit.gz'
    with gzip.open(audit,'wt') as f:f.write(json.dumps(row)+'\n')
    prepare(source,audit,tmp_path/'programs',workers=1,parts=1,min_functions=0)
    pack(tmp_path/'programs',tmp_path/'batches',batch_size=2,seed=42,workers=1,parts=1)
    return tmp_path/'batches'


def test_plan_is_exhaustive_deterministic_and_does_not_mix_footprints():
    index={'a':dict(footprint='x',samples=3),'b':dict(footprint='x',samples=2),'c':dict(footprint='y',samples=3)}
    names,rows,offsets=make_plan(index,2,42)
    assert sorted(map(tuple,rows))==[(i,j) for i,n in enumerate(names) for j in range(index[n]['samples'])]
    assert np.array_equal(rows,make_plan(index,2,42)[1])
    for start,end in zip(offsets,offsets[1:]):
        assert 0<end-start<=2
        assert len({index[names[i]]['footprint'] for i,j in rows[start:end]})==1


def test_full_resume_and_checked_batches(tmp_path):
    batches=prepared_batches(tmp_path)
    packet=BatchStore(batches)[0]
    assert len(packet['base'].labels)==2
    for name in ('lstm','hybrid_rgcn','full_rgcn','heterogat'):
        torch.manual_seed(42);model=build_model(name,efficient=name=='heterogat').eval()
        with torch.no_grad():
            result=model(model_input(packet,name,'cpu'))
            separate=torch.cat([model(model_input(packet,name,'cpu',i,i+1)) for i in range(2)])
        torch.testing.assert_close(result,separate,atol=2e-5,rtol=2e-5)
    base=dict(model='heterogat',train_batches=str(batches),valid_batches=str(batches),
        learning_rate=1e-4,weight_decay=.015,epochs=2,batch_size=2,microbatch=1,seed=42,loader_workers=0,diagnostic=True)
    # The repeated train/valid fixture is only a resume test, never experiment data.
    for name in ('continuous','resumed'):
        config=base|dict(run_root=str(tmp_path/name));path=tmp_path/(name+'.json');path.write_text(json.dumps(config))
        if name=='resumed':train(path,stop_after=1,device='cpu')
        train(path,device='cpu')
    full=torch.load(tmp_path/'continuous/epoch_0002.pt',map_location='cpu',weights_only=False)
    resumed=torch.load(tmp_path/'resumed/epoch_0002.pt',map_location='cpu',weights_only=False)
    for key in full['model']:assert torch.equal(full['model'][key],resumed['model'][key])
    assert full['history'][-1]['valid_mape']==resumed['history'][-1]['valid_mape']
    assert (tmp_path/'resumed/epoch_0001.pt').exists()
    path=tmp_path/'resumed.json';config=json.loads(path.read_text());config['learning_rate']=.01;path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='identity differs'):train(path,device='cpu')
    db=sqlite3.connect(batches/'batches_000.sqlite');db.execute("UPDATE batches SET payload=? WHERE id=0",(b'broken',));db.commit();db.close()
    with pytest.raises(ValueError,match='payload changed'):BatchStore(batches)[0]


def test_microbatch_gradients_keep_upstream_nominal_batch_weighting():
    from models.gnn.matched.train import phase
    from models.gnn.graph import collate_records
    from .fixtures import fixture
    record,_=fixture();base=collate_records([(record,0),(record,1)])
    base.inputs[1][:,0,0]=torch.tensor([2.,3.]);base.labels[:]=torch.tensor([1.,2.])
    class Scalar(torch.nn.Module):
        def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.tensor(0.))
        def forward(self,x):return x[1][:,0,0]*self.w
    for micro in (1,2):
        model=Scalar();optimizer=torch.optim.SGD(model.parameters(),lr=.001)
        mape,n=phase(model,[{'base':base}],'lstm','cpu',optimizer,batch_size=4,microbatch=micro)
        assert n==2 and mape==100.
        torch.testing.assert_close(model.w,torch.tensor(.0875))


def test_split_audit_detects_ids_and_renamed_exact_annotations():
    from models.gnn.matched.audit_splits import compare
    train={'same_id':{'annotation_sha256':'a'},'different_id':{'annotation_sha256':'b'}}
    valid={'same_id':{'annotation_sha256':'c'},'copy':{'annotation_sha256':'b'}}
    result=compare(train,valid)
    assert result['program_id_overlap']==1
    assert result['annotation_matches']=={'different_id':['copy']}
    assert not result['no_exact_overlap']
    assert compare(train,{'separate':{'annotation_sha256':'z'}})['no_exact_overlap']
