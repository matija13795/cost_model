"""Allocated-GPU compatibility, tiny-fit and full-state resume gates.

This uses a small raw fixture; its losses are diagnostics, never validation
scores or an end-to-end dataset throughput estimate.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
from types import SimpleNamespace
import numpy as np
import torch
from .data import upstream, extract_program
from ..factory import build_model, MODEL_NAMES
from ..graph import collate_records
from ..hetero_data import hetero_batch, tensor_template
from ..full_gnn_data import full_graph_batch
from ..efficient_model import tensor_batch
from ..io_utils import atomic_json, atomic_torch, sha256


def mape(prediction, target):
    return 100 * ((prediction.reshape(-1)-target).abs()/target.abs().clamp_min(1e-4)).mean()


def inputs(base, record, architecture, device):
    if architecture == 'lstm':
        return base.to(device).inputs
    if architecture == 'hybrid_rgcn':
        return base.to(device)
    if architecture == 'full_rgcn':
        return full_graph_batch(base, SimpleNamespace(get=lambda _:record['topology']['expressions'])).to(device)
    batch = hetero_batch(base, SimpleNamespace(get=lambda _:tensor_template(record['topology'])))
    return tuple(t.to(device) for t in tensor_batch(batch))


def run(fixture, out, steps=200):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA diagnostics require an allocated GPU')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.set_float32_matmul_precision('highest')
    raw=pickle.load(open(fixture,'rb'));name=next(iter(raw));raw=raw[name]
    u=upstream();ids=list(range(len(raw['schedules_list'])))
    labels=np.asarray([max(.01,raw['initial_execution_time']/min(raw['schedules_list'][i]['execution_times'])) for i in ids],dtype='<f4')
    row=dict(name=name,schedule_ids=ids,schedules=len(ids),footprint=u.get_tree_footprint(u.get_representation_template(raw)[0]),target_fp32_sha256=hashlib.sha256(labels.tobytes()).hexdigest())
    record,_=extract_program(name,raw,row,u);base=collate_records([(record,i) for i in range(len(ids))])
    target=base.labels.cuda()
    results=dict(phase='cuda_diagnostics',fixture_sha256=sha256(fixture),samples=len(ids),steps=steps,
                 torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),
                 slurm_job=os.getenv('SLURM_JOB_ID'),models={})
    for architecture in MODEL_NAMES:
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        model=build_model(architecture,efficient=architecture=='heterogat').eval()
        cpu_input=inputs(base,record,architecture,'cpu')
        with torch.no_grad():cpu_prediction=model(cpu_input)
        model=model.to('cuda')
        if architecture=='lstm':model.device='cuda'
        gpu_input=inputs(base,record,architecture,'cuda')
        with torch.no_grad():gpu_prediction=model(gpu_input)
        torch.testing.assert_close(cpu_prediction,gpu_prediction.cpu(),rtol=2e-4,atol=2e-4)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.015)
        initial=float(mape(gpu_prediction,target));best=initial
        torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        def step():
            model.train();optimizer.zero_grad(set_to_none=True)
            loss=mape(model(gpu_input),target)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise ValueError('Nonfinite gradient')
            optimizer.step();return float(loss)
        for _ in range(steps):
            step();model.eval()
            with torch.no_grad():best=min(best,float(mape(model(gpu_input),target)))
        if best >= .7*initial:
            raise ValueError(f'{architecture}: fixed-sample loss did not decrease sufficiently: {initial} -> {best}')
        checkpoint=out/(architecture+'_resume_test.pt')
        atomic_torch(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                          cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),checkpoint)
        expected_loss=step()
        expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['cpu_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        resumed_loss=step()
        if abs(resumed_loss-expected_loss)>1e-5:raise ValueError('Resume loss differs')
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value.cpu(),expected[key],rtol=1e-6,atol=1e-7)
        results['models'][architecture]=dict(cpu_cuda_max_absolute_difference=float((cpu_prediction-gpu_prediction.cpu()).abs().max()),
            initial_tiny_sample_mape=initial,best_tiny_sample_mape=best,
            resume_continuity=True,seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated())
        atomic_json(results,out/'results.json');print(json.dumps(results['models'][architecture]),flush=True)
        del model,optimizer,cpu_input,gpu_input
        torch.cuda.empty_cache()
    results['phase']='passed';atomic_json(results,out/'results.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=200);a=p.parse_args()
    run(a.fixture,a.out,a.steps)
