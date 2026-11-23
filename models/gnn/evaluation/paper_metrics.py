"""Matija's reported instance MAPE and program-averaged ranking definitions."""
import argparse
import hashlib
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

REFERENCE = '5a18f1c6bd2104aee2f2d57fc234fcf9f5e990d9'


def calculate(frame):
    required = {'sample_id', 'name', 'target', 'prediction'}
    if not required <= set(frame):
        raise ValueError('Missing columns: ' + str(required - set(frame)))
    if len(frame) == 0 or frame['sample_id'].isna().any() or not frame['sample_id'].is_unique or frame['name'].isna().any():
        raise ValueError('Require nonempty data, unique schedule IDs and program identities')
    raw_y = frame.target.to_numpy(dtype=np.float32)
    raw_p = frame.prediction.to_numpy(dtype=np.float32)
    if not np.isfinite(raw_y).all() or not np.isfinite(raw_p).all() or (raw_y <= 0).any():
        raise ValueError('Require finite predictions and finite positive targets')
    # Same criterion mathematically, accumulated in float64 for reporting.
    ape = 100 * np.abs(raw_y.astype(np.float64) - raw_p.astype(np.float64)) / np.maximum(np.abs(raw_y.astype(np.float64)), 1e-4)
    work = frame[['name']].copy()
    work['target'] = np.around(raw_y, decimals=6)
    work['prediction'] = np.around(raw_p, decimals=6)
    work['raw_ape'] = ape
    if (work.target <= 0).any():
        raise ValueError('Six-decimal rounding produced a nonpositive target')
    rows = []
    for name, g in work.groupby('name', sort=True):
        y, pred = g.target.to_numpy(), g.prediction.to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rho = float(spearmanr(y, pred)[0]) if len(g) >= 2 else np.nan
        rows.append(dict(name=name, count=len(g), ape_sum=float(g.raw_ape.sum()),
                         MAPE_program=float(np.mean(np.abs(y-pred)/y*100)),
                         nDCG=float(ndcg_score([y.tolist()], [pred.tolist()], k=None)) if len(g) >= 2 else np.nan,
                         nDCG_5=float(ndcg_score([y.tolist()], [pred.tolist()], k=5)) if len(g) >= 2 else np.nan,
                         nDCG_1=float(ndcg_score([y.tolist()], [pred.tolist()], k=1)) if len(g) >= 2 else np.nan,
                         Spearman=rho))
    programs = pd.DataFrame(rows)
    def mean_or_none(column):
        return float(programs[column].mean()) if programs[column].notna().any() else None
    summary = dict(schedules=len(frame), programs=len(programs),
                   MAPE_instance_percent=float(ape.mean()),
                   MAPE_program_percent=mean_or_none('MAPE_program'),
                   nDCG_full=mean_or_none('nDCG'), Spearman=mean_or_none('Spearman'),
                   nDCG_5=mean_or_none('nDCG_5'), nDCG_1=mean_or_none('nDCG_1'),
                   undefined_nDCG=int(programs.nDCG.isna().sum()),
                   undefined_nDCG_5=int(programs.nDCG_5.isna().sum()),
                   undefined_nDCG_1=int(programs.nDCG_1.isna().sum()),
                   undefined_Spearman=int(programs.Spearman.isna().sum()),
                   reference_commit=REFERENCE,
                   precision='FP32 model outputs; float64 instance MAPE accumulation; ranking uses six-decimal FP32 rounding',
                   weighting='MAPE: every schedule equally; nDCG and Spearman: every defined program score equally')
    return summary, programs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('predictions', type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    frame = pd.read_csv(a.predictions, float_precision='round_trip')
    summary, programs = calculate(frame)
    a.out.mkdir(parents=True, exist_ok=False)
    h = hashlib.sha256()
    with a.predictions.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2), b''): h.update(block)
    summary['predictions_sha256'] = h.hexdigest()
    (a.out/'metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    programs.to_csv(a.out/'program_metrics.csv', index=False)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__': main()
