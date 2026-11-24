"""Batch-one adapter for the unchanged inference_timer 1.0.1 package."""
from pathlib import Path
import csv
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor/inference_timer'))
from inference_timer import benchmark


def measure(model, inputs, *, batch_size, device, csv_path):
    """Measure only forward inference. Prepare and transfer inputs beforehand."""
    if batch_size != 1:
        raise ValueError('Single-input latency requires exactly one schedule')
    if model.training:
        raise ValueError('Call model.eval() before measuring')
    path = Path(csv_path)
    if path.exists():
        raise FileExistsError('Preserve previous measurements: ' + str(path))
    with torch.inference_mode():
        elapsed = benchmark(lambda: model(inputs), device=device, csv_path=path)
        # Validate after all timed calls, so this is not an extra warmup.
        result = model(inputs)
    if result.numel() != 1 or not torch.isfinite(result).all():
        raise ValueError('Expected one finite speedup prediction')
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    measured = [float(r['elapsed_ms']) for r in rows if r['status'] == 'kept']
    assert len(rows) == 7 and len(measured) == 3
    assert rows[0]['status'] == 'warmup'
    assert abs(elapsed - sum(measured) / 3) < 1e-10
    return {'trimmed_mean_ms': elapsed, 'prediction': result.item(),
            'csv': str(path), 'kept_range_ms': [min(measured), max(measured)]}
