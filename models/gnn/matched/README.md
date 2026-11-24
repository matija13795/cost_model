# Shared-data experiment, November 24

The owner supplied four original raw pickle files on November 24, replacing the previously inspected representation folders at the same shared path. The 1M probe confirms original buffer annotations and expression trees. These are new matched-data runs; old 9.24M/1.07M scores and checkpoints are not reused as results or initialization.

## Configuration authority

`configs/protocol.json` transcribes the supplied table and explicitly lists unresolved settings and our defaults. AdamW, LR 1e-4, batch 512 and 500 epochs are supplied. FCNN/CNN/SimpleRNN have different regularization values in that table; none defines GNN regularization. We preserve our four existing architectures and their initialization/activation/dropout, using common weight decay 0.015. The current upstream recursive LSTM also uses ELU. Changing these models to the table's GELU would create an additional variant.

Seed 42, constant LR, FP32 without TF32 and the upstream template's training footprint filter of 2 are declared choices, not verified Matija run settings. Original serialized batch files and shuffle order are unavailable. We therefore cannot claim bit-for-bit reproduction of his training. Every one of our models will receive identical accepted membership and logical batches.

## Data gates

The November 23 audit preserved all program IDs, schedule IDs, ordered FP32 target hashes and tree footprints from the supplied processed pickles. Preparation checks every pre-filter target against the new raw data, applies the stated footprint filter, and regenerates features with the **unchanged pinned upstream** extractor. All emitted features must be finite. Every program's graph is checked against the upstream tree indexing; local expression graph features must match the upstream flat features exactly. A mismatch fails preparation rather than dropping an example silently.

The historical loop-zeroing behavior in upstream `utils/data_utils.py` is deliberately retained for this shared-feature comparison. This distinguishes these runs from our earlier corrected-feature experiments. The new raw files allow recovery of graph topology; flattened expression vectors alone did not.

Expected counts from the processed audit, subject to raw reconciliation:

| Dataset | Audited schedules before footprint filter | Selected schedules |
|---|---:|---:|
| 1M training | 1,091,955 | 1,007,326 |
| 5M training | 5,384,838 | 5,077,478 |
| 10M training | 10,757,418 | 10,229,066 |
| Common validation | 5,550,533 | 5,550,533 |

The last column for training follows `programs_in_footprint < 2 AND schedules_in_footprint < 200`; validation has no footprint filter. No target-based subsampling is introduced. Cross-split program and canonical annotation overlap must be reported before production; do not silently change the supplied membership.

## Preparation commands

Run on an allocated CPU node with the pinned dependencies. The `audit` file is the preserved `programs.jsonl.gz` from the corresponding processed folder, not a newly invented subset.

```sh
python -m models.gnn.matched.copy_raw /shared/source.pkl /project/raw/source.pkl
python -m models.gnn.matched.data --source /project/raw/source.pkl \
  --audit /project/audit/programs.jsonl.gz --out /project/cache/train_1m \
  --workers 16 --parts 32 --min-functions 2
```

Use `--min-functions 0` for common validation. The worker implementation uses Linux fork to share the read-only raw dictionary and writes separate compressed SQLite shards. Expressions and topology are stored once per program; learned embeddings are never cached. Completed shard hashes and source/code identities are verified on resume. A different source or implementation requires a new cache version. Temporary files and incomplete shards are preserved for diagnosis.

## Completion gates

Preparation is followed by cross-split leakage auditing, immutable batch plans, CPU/CUDA prediction checks, a resume test and a GPU throughput pilot. Production must not start before these gates. Use actual end-to-end epoch timings to request allocations. The final evaluation uses the shared sheet's instance MAPE, full nDCG, nDCG@5, nDCG@1 and Spearman, the supplied single-input timer, and all 30 PolyBench Medium kernels with identical inputs/settings. No historical score is to be entered as a matched result.

## Batch preparation and trainer

After raw preparation completes:

```sh
python -m models.gnn.matched.audit_splits --root /project/datasets --out /project/results/split_audit.json
python -m models.gnn.matched.batches --root /project/datasets/1m --out /project/batches/1m \
  --batch-size 512 --seed 42 --workers 8 --parts 32
```

Repeat packing for the other training sets and validation. Batch membership is frozen by footprint. Sample order within a footprint uses a declared NumPy seed; training shuffles the resulting logical batches deterministically each epoch. Packed tensors include only fixed input features/topology, not learned embeddings. Training reads bounded batches from compressed SQLite packs. All architectures share the exact plan and labels. CPU extraction and graph construction are done before GPU iterations; performance still needs a measured GPU pilot.

The standalone entry point is `python -m models.gnn.matched.train --config /project/config.json`. It does not change the upstream Hydra entry point. Example configuration fields:

```json
{
  "model": "heterogat",
  "train_batches": "/project/batches/1m",
  "valid_batches": "/project/batches/validation",
  "run_root": "/project/runs/heterogat_1m",
  "learning_rate": 0.0001,
  "weight_decay": 0.015,
  "epochs": 500,
  "batch_size": 512,
  "microbatch": 512,
  "seed": 42,
  "loader_workers": 2,
  "gate_file": "/project/results/production_gate.json",
  "gate_sha256": "SHA256_OF_REVIEWED_GATE"
}
```

Start with the full 512-schedule microbatch; reduce it only if measured memory or throughput requires accumulation, while retaining logical batch size 512.

`model` also accepts `lstm`, `hybrid_rgcn` and `full_rgcn`. Partial logical batches retain the upstream nominal-batch loss scaling (`sum(APE)/512`). Validation MAPE is weighted by actual examples. Each epoch saves an immutable checkpoint with model, optimizer, RNG states, resolved config, dataset/code hashes and history. `latest.json` and `best.json` point to checked checkpoint files. Reusing the same command resumes; changed configuration, source code or data is rejected. `--stop-after N` stops at an epoch boundary without changing the experiment identity.

A separate diagnostic run can set `diagnostic: true`; this bypass is labeled and cannot be resumed into a different production configuration. CPU is allowed only for diagnostics. Production requires a checksummed gate with `data_reconciled`, `no_train_validation_overlap`, `cuda_checks_passed` and `throughput_approved` set to true, and `batch_manifests` containing the exact train/valid manifest hashes. Do not fill these flags speculatively. This gate does not assert that Matija's unspecified run settings have been recovered.

Local tests cover batch completeness, footprint isolation, non-overwriting source copies, cache rejection, CPU batch equivalence for all models, nominal-batch gradient weighting and exact interrupted/resumed model states. GPU checks separately compare CPU/CUDA predictions, fixed-sample learning and optimizer/RNG resume. The four-schedule learning diagnostic allows 1,000 updates with the unchanged optimizer and requires a 30% loss reduction; the original 200 updates were too short for HeteroGAT. A separate, fresh one-schedule memorization test requires MAPE below 1% within 1,000 updates. Both results are retained: a reduction on the four schedules is not reported as successful memorization. These are diagnostics, not the 500-epoch experiment. Their tiny-sample losses are not validation results.
