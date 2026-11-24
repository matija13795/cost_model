# LOOPer graph model family

This branch packages the three existing graph variants and a historical LSTM checkpoint compatibility factory. It does not introduce another architecture. Matija's existing `utils/modeling.py`, training loop and Hydra configurations remain unchanged.

## Status and scope

The model library and evaluation utilities are tested locally. Original raw datasets and a hyperparameter table arrived on November 24. The new checked preparation path is documented in [matched/README.md](matched/README.md). Full matched training remains gated on complete data reconciliation and GPU checks. The graph factories cannot be selected by changing only `model` in the upstream trainer.

The baseline factory below is the archived baseline used for the historical experiments. It is provided to load those checkpoints and reproduce graph initialization; it does not replace the active upstream baseline or assert that the historical training protocol matches Matija's current runs.

## Architectures

| Factory name | Parameters | Structure |
|---|---:|---|
| `lstm` | 2,109,623 | Historical LOOPer LSTM, five recurrent modules |
| `hybrid_rgcn` | 2,128,703 | Local expression/transformation LSTMs; four relational graph layers for program aggregation |
| `full_rgcn` | 2,125,071 | Expression and transformation graph encoders; four relational graph layers; no recurrent modules |
| `heterogat` | 2,114,663 | Local graph encoders; four graph attention layers with explicit buffer nodes; no recurrent modules |

The RGCNs use containment, sibling order and shared-buffer computation links, each directed both ways. HeteroGAT replaces the shared-buffer computation links with buffer nodes and separate read/write edges, both directed ways. It has eight relation types, width 152 and four attention heads. Mean pooling includes actual loops and computations; buffer nodes contribute through message passing but are not pooled directly. Shared buffers indicate potential interaction, not proven dependencies.

All three models predict one speedup for one program/schedule input. IDs and timing measurements are metadata or targets, not node features. Topology comes from the original program; existing schedule features carry transformations. Model-only timing excludes graph preparation.

## Use

From the repository root, with dependencies in a project-local environment:

```python
import torch
from models.gnn.factory import build_model

torch.manual_seed(42)
model = build_model('heterogat', device='cpu', efficient=True)
model.eval()
```

`efficient=True` applies only to HeteroGAT and preserves the state-dict layout. The reference model takes a `HeteroBatch`; `models.gnn.efficient_model.tensor_batch` prepares the efficient tuple. This precomputation is outside its model-only timing. `hybrid_rgcn` takes `GraphBatch`; `full_rgcn` takes `FullGraphBatch`; historical `lstm` takes the six-element recursive input tuple.

`graph.py` batches computation and loop features, with adjacency indexed `[example, relation, destination, source]`. `full_gnn_data.py` constructs expression graphs and packs them; `hetero_data.py` adds typed buffers and masks. The included feature extractor is preserved for historical compatibility. Do not use it to regenerate the shared training sets before comparing its output, filtering and ordering with Matija's exact preprocessing. Flat expression vectors alone cannot recover the original expression tree.

## Metrics matching the shared sheet

```sh
python -m models.gnn.evaluation.paper_metrics predictions.csv --out evaluation/new_run
```

CSV columns: `sample_id,name,target,prediction`. Each sample ID must be unique and targets must be finite and positive. The command refuses to overwrite an existing output directory. It writes per-program scores and an aggregate JSON report.

Primary MAPE is instance weighted: `100 * mean(abs(target - prediction) / max(abs(target), 1e-4))`. Targets/predictions are interpreted as FP32 model outputs; the report accumulates MAPE in float64, so floating-point reduction differences from Torch can occur. Per-program MAPE is supplementary and must not populate the sheet's primary MAPE column.

Full nDCG, nDCG@5, nDCG@1 and Spearman follow the upstream definitions at commit `5a18f1c6bd2104aee2f2d57fc234fcf9f5e990d9`: predictions/targets rounded to six FP32 decimals for ranking, sklearn linear relevance and default tie averaging, and an equal-weight mean over defined program scores. Singleton rankings and undefined correlations are counted and omitted from those means. Tests execute the corresponding upstream functions on tied and singleton examples.

## Single-input timing

```python
from models.gnn.evaluation.single_input import measure
result = measure(model.eval(), prepared_single_input,
                 batch_size=1, device='cpu', csv_path='latency/new_input.csv')
```

Use exactly one function and one schedule. All tensors, graph preparation and transfers are prepared before the call. The unchanged instructor-supplied inference_timer 1.0.1 performs one excluded warmup, five timed forwards, discards exactly one fastest and one slowest run, and averages the other three. CUDA is synchronized before/after measurement. No cache flush is performed. The wrapper makes one correctness call after the measurements, not another warmup. Preserve raw CSVs and report the input, hardware, precision, CPU thread count and checkpoint hash. A mean over multiple single-input timings is supplementary and is not the same as timing a batch.

The timer was supplied for this project by the course team. Original attachment SHA256: `d8b797bd99d629016f27ce1d55bfefa9d52e9d61a9e4e2ae76b9fd7cd55d5528`. Core source hashes and the unchanged timer source are recorded in `SOURCE_PROVENANCE.json`.

## Verification

```sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/gnn
```

The current suite covers preserved parameter counts/recurrent-module counts, graph relations, attention weights, padding, graph reindexing, sample isolation, finite gradients, reference/efficient prediction and gradient parity, target isolation, metric agreement and timing protocol. It does not claim a matched full-data training or GPU test of the relocated package. Previously collected CUDA timings are separate historical artifacts, not tests of a newly trained matched checkpoint.

## Gates before new experiments

Follow the current gates and resolved assumptions in [matched/README.md](matched/README.md). The November 23 processed-folder audit is preserved; those shared folders were subsequently replaced by raw pickle files. Preparation must reconcile the raw files against that audit before new training. This PR stays in draft until training and evaluation integration passes its checks.

Historical 9.24M/1.07M results must remain separately labeled and cannot populate the new matched rows merely because both are called “10M”.
