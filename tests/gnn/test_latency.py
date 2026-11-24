import csv
import pytest
import torch
from models.gnn.evaluation import single_input
import inference_timer

def test_exact_six_calls_and_trimmed_mean(tmp_path, monkeypatch):
    # Warmup 99 ms, then 1, 5, 3, 2, 4 ms: retain 2, 3, 4 => 3 ms.
    clock = iter(v for ms in [99, 1, 5, 3, 2, 4] for v in (0, ms * 1_000_000))
    monkeypatch.setattr(inference_timer, 'perf_counter_ns', lambda: next(clock))
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def forward(self, x):
            assert not torch.is_grad_enabled() and not self.training
            self.calls += 1
            return x
    model = Model().eval()
    result = single_input.measure(model, torch.ones(1), batch_size=1, device='cpu', csv_path=tmp_path/'raw.csv')
    assert result['trimmed_mean_ms'] == 3
    assert model.calls == 7  # six benchmark calls, one correctness check afterwards
    with (tmp_path/'raw.csv').open() as f:
        statuses = [r['status'] for r in csv.DictReader(f)]
    assert statuses == ['warmup', 'excluded_fastest', 'excluded_slowest', 'kept', 'kept', 'kept', 'trimmed_mean']

def test_reject_batch_training_mode_and_overwrite(tmp_path):
    model = torch.nn.Identity()
    with pytest.raises(ValueError, match='exactly one'):
        single_input.measure(model, None, batch_size=2, device='cpu', csv_path=tmp_path/'a')
    with pytest.raises(ValueError, match='eval'):
        single_input.measure(model, None, batch_size=1, device='cpu', csv_path=tmp_path/'a')
    (tmp_path/'a').write_text('preserved')
    with pytest.raises(FileExistsError):
        single_input.measure(model.eval(), None, batch_size=1, device='cpu', csv_path=tmp_path/'a')
    assert (tmp_path/'a').read_text() == 'preserved'
