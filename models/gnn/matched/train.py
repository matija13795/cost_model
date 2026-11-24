"""Fresh matched training with immutable checkpoints and deterministic batches.

A pilot uses the same code but a separate output directory. Production requires
an explicit data/GPU/performance gate file; it cannot accidentally resume an old
9.24M experiment.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
from .batches import BatchStore
from ..factory import build_model, MODEL_NAMES
from ..full_gnn_data import FullGraphBatch
from ..io_utils import atomic_json, atomic_torch, sha256


def code_identity():
    root=Path(__file__).parents[1]
    files=['constants.py','factory.py','paper_models.py','gnn.py','model.py','graph.py','full_gnn.py','hetero_gat.py',
           'efficient_model.py','pact25_archive/modeling.py','matched/train.py','matched/batches.py']
    return {f:sha256(root/f) for f in files}


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])


def model_input(packet,architecture,device,start=0,end=None):
    base=packet['base'];end=len(base.labels) if end is None else end
    if architecture=='heterogat':return tuple(t[start:end].to(device,non_blocking=True) for t in packet['efficient'])
    sliced=base.slice(start,end).to(device)
    if architecture=='lstm':return sliced.inputs
    if architecture=='hybrid_rgcn':return sliced
    c=base.inputs[1].shape[1]
    return FullGraphBatch(sliced,tuple(t[start*c:end*c].to(device,non_blocking=True) for t in packet['expressions']))


class EpochOrder(torch.utils.data.Sampler):
    def __init__(self,size,seed,shuffle=True):self.size=size;self.seed=seed;self.shuffle=shuffle;self.epoch=0
    def __len__(self):return self.size
    def __iter__(self):
        return iter(np.random.default_rng(self.seed+self.epoch).permutation(self.size).tolist() if self.shuffle else range(self.size))


def unchanged(value):return value


def phase(model,loader,architecture,device,optimizer,batch_size,microbatch,progress=None,epoch=0):
    model.train(optimizer is not None);total=0.;samples=0;started=time.monotonic()
    for number,packet in enumerate(loader):
        labels=packet['base'].labels;n=len(labels)
        if optimizer:optimizer.zero_grad(set_to_none=True)
        for lo in range(0,n,microbatch):
            hi=min(n,lo+microbatch);target=labels[lo:hi].to(device)
            with torch.set_grad_enabled(optimizer is not None):
                prediction=model(model_input(packet,architecture,device,lo,hi)).reshape(-1)
                ape=100*(prediction-target).abs()/target.abs().clamp_min(1e-4)
                if not torch.isfinite(ape).all():raise ValueError('Nonfinite prediction/loss')
                if optimizer:
                    # Upstream scales partial logical batches by n / nominal batch size.
                    (ape.sum()/batch_size).backward()
            total+=float(ape.detach().double().sum());samples+=hi-lo
        if optimizer:
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise ValueError('Nonfinite gradient')
            optimizer.step()
        if progress and (number%100==0 or number+1==len(loader)):
            atomic_json(dict(phase='training' if optimizer else 'validation',epoch=epoch,
                batches_completed=number+1,total_batches=len(loader),samples=samples,
                partial_instance_mape=total/samples,seconds=time.monotonic()-started,updated_unix=time.time()),progress)
    if not samples:raise ValueError('Empty epoch')
    return total/samples,samples


def train(config_path,stop_after=None,device='cuda'):
    config=json.loads(Path(config_path).read_text());root=Path(config['run_root']);root.mkdir(parents=True,exist_ok=True)
    with open(root/'run.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is required for this run')
        if device!='cuda' and not config.get('diagnostic',False):raise ValueError('Production requires CUDA')
        if not config.get('diagnostic',False):
            gate_path=Path(config['gate_file'])
            if sha256(gate_path)!=config['gate_sha256']:raise ValueError('Production gate changed')
            gate=json.loads(gate_path.read_text())
            if not all(gate.get(k) is True for k in ('data_reconciled','no_train_validation_overlap','cuda_checks_passed','throughput_approved')):
                raise ValueError('Production gates are incomplete')
        torch.set_num_threads(config.get('torch_threads',1))
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.set_float32_matmul_precision('highest')
        seed=config.get('seed',42);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
        architecture=config['model'];model=build_model(architecture,device,efficient=architecture=='heterogat')
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        stores={s:BatchStore(config[s+'_batches']) for s in ('train','valid')}
        fingerprints={s:sha256(Path(config[s+'_batches'])/'manifest.json') for s in stores}
        if not config.get('diagnostic',False) and gate.get('batch_manifests')!=fingerprints:
            raise ValueError('Production gate does not cover these exact batch manifests')
        code=code_identity();loaders={};orders={}
        for split,store in stores.items():
            if store.manifest['identity']['batch_size']!=config['batch_size'] or store.manifest['identity']['seed']!=seed:
                raise ValueError('Batch plan and training configuration disagree')
            order=EpochOrder(len(store),seed,split=='train');orders[split]=order
            workers=config.get('loader_workers',2)
            extra=dict(persistent_workers=True,prefetch_factor=2,multiprocessing_context='spawn') if workers else {}
            loaders[split]=torch.utils.data.DataLoader(store,batch_size=None,sampler=order,num_workers=workers,
                collate_fn=unchanged,generator=torch.Generator().manual_seed(seed),**extra)
        history=[];best=float('inf');first=0
        last=root/'latest.json'
        if last.exists():
            pointer=json.loads(last.read_text());path=root/pointer['file']
            if sha256(path)!=pointer['sha256']:raise ValueError('Checkpoint checksum changed')
            saved=torch.load(path,map_location='cpu',weights_only=False)
            if saved['config']!=config or saved['data']!=fingerprints or saved['code']!=code:
                raise ValueError('Checkpoint config/data/code identity differs')
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
            history=saved['history'];best=saved['best_mape'];first=saved['completed_epochs']
        atomic_json(dict(config=config,data=fingerprints,code=code,torch=torch.__version__,
            device=torch.cuda.get_device_name(0) if device=='cuda' else 'cpu',slurm_job=os.getenv('SLURM_JOB_ID'),
            parameters=sum(p.numel() for p in model.parameters()),precision='FP32; TF32 off'),root/'environment.json')
        stop=[False]
        def request_stop(*_):stop[0]=True
        for sig in (signal.SIGTERM,signal.SIGUSR1):signal.signal(sig,request_stop)
        end=min(config['epochs'],stop_after) if stop_after is not None else config['epochs']
        try:
            for completed in range(first,end):
                if stop[0]:break
                epoch=completed+1;started=time.monotonic();orders['train'].epoch=epoch
                if device=='cuda':torch.cuda.reset_peak_memory_stats()
                train_mape,train_n=phase(model,loaders['train'],architecture,device,optimizer,config['batch_size'],config['microbatch'],root/'progress.json',epoch)
                valid_mape,valid_n=phase(model,loaders['valid'],architecture,device,None,config['batch_size'],config['microbatch'],root/'progress.json',epoch)
                if train_n!=stores['train'].manifest['samples'] or valid_n!=stores['valid'].manifest['samples']:
                    raise ValueError('Epoch did not visit every declared schedule')
                row=dict(epoch=epoch,train_mape=train_mape,valid_mape=valid_mape,train_samples=train_n,valid_samples=valid_n,
                         seconds=time.monotonic()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0)
                improved=valid_mape<best;best=min(best,valid_mape);history.append(row)
                path=root/f'epoch_{epoch:04d}.pt'
                if path.exists():raise ValueError('Refusing to overwrite existing epoch checkpoint')
                atomic_torch(dict(completed_epochs=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),
                    rng=rng_state(),config=config,data=fingerprints,code=code,history=history,best_mape=best),path)
                pointer=dict(epoch=epoch,file=path.name,sha256=sha256(path),valid_mape=valid_mape)
                atomic_json(pointer,root/'latest.json')
                if improved:atomic_json(pointer,root/'best.json')
                atomic_json(history,root/'history.json')
                atomic_json(dict(phase='running',completed_epochs=epoch,target_epochs=config['epochs'],best_mape=best,
                    latest_valid_mape=valid_mape,epoch_seconds=row['seconds'],updated_unix=time.time()),root/'status.json')
                print(json.dumps(row),flush=True)
            atomic_json(dict(phase='complete' if len(history)==config['epochs'] else 'needs_resume',
                completed_epochs=len(history),target_epochs=config['epochs'],best_mape=best if history else None,
                updated_unix=time.time()),root/'status.json')
        except BaseException as exc:
            atomic_json(dict(phase='failed',completed_epochs=len(history),error=repr(exc),updated_unix=time.time()),root/'status.json')
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--stop-after',type=int)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args();train(a.config,a.stop_after,a.device)
