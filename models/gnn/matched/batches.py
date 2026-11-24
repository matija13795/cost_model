"""Freeze shared logical batches and prepare graph tensors once on CPU."""
import argparse
from collections import OrderedDict,defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import sqlite3
import time
import zlib
from types import SimpleNamespace
import numpy as np
import torch
from ..io_utils import sha256,atomic_json
from ..graph import collate_records
from ..hetero_data import hetero_batch,tensor_template
from ..efficient_model import tensor_batch


def connect(path):
    return sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)


def program_index(root, verify=True):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text());index={}
    for part in manifest['parts']:
        path=root/part['file']
        if verify and sha256(path)!=part['sha256']:raise ValueError('Prepared program shard changed')
        db=connect(path)
        try:
            for name,info in db.execute('SELECT name,info FROM programs'):
                if name in index:raise ValueError('Repeated prepared program')
                row=json.loads(info);index[name]=dict(part=part['file'],samples=row['samples'],footprint=row['footprint'],annotation_sha256=row['annotation_sha256'])
        finally:db.close()
    if sum(r['samples'] for r in index.values())!=manifest['membership']['selected_schedules']:
        raise ValueError('Program index sample count mismatch')
    return index


def make_plan(index,batch_size=512,seed=42):
    if batch_size<=0:raise ValueError('Invalid batch size')
    if not index:raise ValueError('Cannot batch an empty dataset')
    names=sorted(index);groups=defaultdict(list)
    for i,name in enumerate(names):groups[index[name]['footprint']].append(i)
    rng=np.random.default_rng(seed);chunks=[];offsets=[0]
    for footprint in sorted(groups):
        rows=np.concatenate([np.column_stack((np.full(index[names[i]]['samples'],i,dtype=np.int32),
                              np.arange(index[names[i]]['samples'],dtype=np.int32))) for i in groups[footprint]])
        rng.shuffle(rows);chunks.append(rows)
        start=offsets[-1]
        offsets.extend(start+j for j in range(batch_size,len(rows),batch_size));offsets.append(start+len(rows))
    return names,np.concatenate(chunks),np.asarray(offsets,dtype=np.int64)


class ProgramStore:
    def __init__(self,root,index,max_bytes=512*1024**2):
        self.root=Path(root);self.index=index;self.max_bytes=max_bytes;self.dbs={};self.cache=OrderedDict();self.bytes=0
    def get(self,name):
        if name in self.cache:
            result,size=self.cache.pop(name);self.cache[name]=(result,size);return result
        part=self.index[name]['part']
        if part not in self.dbs:self.dbs[part]=connect(self.root/part)
        blob,digest=self.dbs[part].execute('SELECT payload,sha256 FROM programs WHERE name=?',(name,)).fetchone()
        if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('Program payload checksum mismatch')
        decoded=zlib.decompress(blob);result=pickle.loads(decoded);size=len(decoded)
        while self.cache and self.bytes+size>self.max_bytes:
            _,(_,old)=self.cache.popitem(last=False);self.bytes-=old
        if size<=self.max_bytes:self.cache[name]=(result,size);self.bytes+=size
        return result
    def close(self):
        for db in self.dbs.values():db.close()
        self.dbs.clear();self.cache.clear();self.bytes=0


_store=None
_names=None
_rows=None
_offsets=None
_target=None


def initialize(root,out):
    global _store,_names,_rows,_offsets,_target
    torch.set_num_threads(1);_target=Path(out)
    _store=ProgramStore(root,json.loads((_target/'program_index.json').read_text()))
    _names=json.loads((_target/'names.json').read_text())
    with np.load(_target/'plan.npz') as plan:_rows=plan['rows'];_offsets=plan['offsets']


def pack_part(task):
    part,numbers=task;path=_target/f'batches_{part:03d}.sqlite';sidecar=path.with_suffix('.json')
    if sidecar.exists():
        entry=json.loads(sidecar.read_text())
        if sha256(path)!=entry['sha256']:raise ValueError('Batch shard changed')
        return entry
    if path.exists():raise ValueError('Unmanifested batch shard exists')
    temp=path.with_name(path.name+f'.building-{os.getpid()}-{time.time_ns()}');db=sqlite3.connect(temp)
    db.execute('CREATE TABLE batches (id INTEGER PRIMARY KEY,samples INTEGER,payload BLOB,sha256 TEXT)')
    entries=[]
    try:
        for number in numbers:
            rows=_rows[_offsets[number]:_offsets[number+1]]
            records={int(i):_store.get(_names[int(i)]) for i in np.unique(rows[:,0])}
            base=collate_records([(records[int(i)],int(j)) for i,j in rows])
            topologies={_names[i]:tensor_template(r['topology']) for i,r in records.items()}
            graph=hetero_batch(base,SimpleNamespace(get=lambda name:topologies[name]))
            packet=dict(base=base,expressions=graph.expressions,efficient=tensor_batch(graph))
            blob=zlib.compress(pickle.dumps(packet,protocol=5),level=1)
            digest=hashlib.sha256(blob).hexdigest()
            db.execute('INSERT INTO batches VALUES (?,?,?,?)',(number,len(rows),blob,digest))
            entries.append(dict(id=number,samples=len(rows),sha256=digest))
            if len(entries)%50==0:
                db.commit();print(json.dumps(dict(phase='packing',part=part,completed=len(entries),total=len(numbers))),flush=True)
        db.commit()
    finally:db.close()
    os.link(temp,path)
    entry=dict(file=path.name,batches=entries,sha256=sha256(path),bytes=path.stat().st_size)
    atomic_json(entry,sidecar);return entry


def pack(root,out,batch_size=512,seed=42,workers=8,parts=32):
    import fcntl
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with open(out/'pack.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=dict(program_manifest_sha256=sha256(root/'manifest.json'),builder_sha256=sha256(__file__),
                      batch_size=batch_size,seed=seed,parts=parts)
        if (out/'identity.json').exists() and json.loads((out/'identity.json').read_text())!=identity:
            raise ValueError('Batch cache identity changed')
        if (out/'manifest.json').exists():
            existing=json.loads((out/'manifest.json').read_text())
            if sha256(out/'plan.npz')!=existing['plan_sha256']:raise ValueError('Frozen batch-plan bytes changed')
        atomic_json(identity,out/'identity.json')
        index=program_index(root);names,rows,offsets=make_plan(index,batch_size,seed)
        if (out/'plan.npz').exists():
            with np.load(out/'plan.npz') as old:
                if not np.array_equal(old['rows'],rows) or not np.array_equal(old['offsets'],offsets):
                    raise ValueError('Frozen batch plan changed')
        else:
            with open(out/'plan.npz','xb') as stream:np.savez_compressed(stream,rows=rows,offsets=offsets)
        atomic_json(index,out/'program_index.json');atomic_json(names,out/'names.json')
        tasks=[(i,list(range(i,len(offsets)-1,parts))) for i in range(parts)];entries=[]
        with ProcessPoolExecutor(workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(str(root),str(out))) as pool:
            for entry in pool.map(pack_part,tasks):
                entries.append(entry);atomic_json(dict(phase='packing',completed_parts=len(entries),total_parts=parts,updated_unix=time.time()),out/'progress.json')
        atomic_json(dict(identity=identity,plan_sha256=sha256(out/'plan.npz'),parts=entries,
                         samples=len(rows),batches=len(offsets)-1),out/'manifest.json')
        atomic_json(dict(phase='packed',samples=len(rows),batches=len(offsets)-1,updated_unix=time.time()),out/'progress.json')


class BatchStore(torch.utils.data.Dataset):
    """Bounded loader; raw graph preparation is absent from training iterations."""
    def __init__(self,root):
        self.root=Path(root);self.manifest=json.loads((self.root/'manifest.json').read_text());self.dbs={};self.entries={}
        for part in self.manifest['parts']:
            for item in part['batches']:self.entries[item['id']]=(part['file'],item)
        if set(self.entries)!=set(range(self.manifest['batches'])):raise ValueError('Batch IDs are not contiguous')
    def __len__(self):return len(self.entries)
    def __getitem__(self,number):
        part,entry=self.entries[number]
        if part not in self.dbs:self.dbs[part]=connect(self.root/part)
        blob,digest=self.dbs[part].execute('SELECT payload,sha256 FROM batches WHERE id=?',(int(number),)).fetchone()
        if digest!=entry['sha256'] or hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('Batch payload changed')
        packet=pickle.loads(zlib.decompress(blob))
        if len(packet['base'].labels)!=entry['samples']:raise ValueError('Batch size mismatch')
        return packet


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--batch-size',type=int,default=512);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--parts',type=int,default=32);a=p.parse_args()
    pack(a.root,a.out,a.batch_size,a.seed,a.workers,a.parts)
