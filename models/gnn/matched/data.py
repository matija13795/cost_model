"""Checked raw-pickle preparation using the pinned upstream feature extractor.

Run on allocated CPU nodes. One SQLite file per worker shard avoids millions
of small files. A completed shard is immutable and reusable after interruption.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import sqlite3
import time
from types import SimpleNamespace
import zlib

import numpy as np
import torch
from ..graph import graph_from_annotation, collate_records
from ..full_gnn_data import program_templates, full_graph_batch
from ..hetero_data import buffer_graph, tensor_template, hetero_batch
from ..io_utils import atomic_json, sha256, annotation_hash

SCHEMA = 1
# Never silently regenerate matched features with a modified extractor.
UPSTREAM_SHA256 = '7db58aff647adae90c12ee23cb3b9b0e600eadd00f840a53a50137e8baa82dd8'
_raw = None
_rows = None
_out = None
_identity = None


def upstream():
    path = Path(__file__).resolve().parents[3] / 'utils' / 'data_utils.py'
    if sha256(path) != UPSTREAM_SHA256:
        raise ValueError('Upstream feature extractor changed; version and review the cache first')
    spec = importlib.util.spec_from_file_location('matched_upstream_features', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_audit(path):
    with gzip.open(path, 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    if len({r['name'] for r in rows}) != len(rows):
        raise ValueError('Duplicate program IDs in accepted-membership audit')
    return {r['name']: r for r in rows}


def select_members(rows, min_functions):
    footprints = defaultdict(lambda: [0, 0])
    for row in rows.values():
        footprints[row['footprint']][0] += 1
        footprints[row['footprint']][1] += row['schedules']
    return {name: row for name, row in rows.items()
            if not (footprints[row['footprint']][0] < min_functions and
                    footprints[row['footprint']][1] < 100 * min_functions)
            and row['schedules'] > 0}


def checked_targets(raw, row):
    ids = row['schedule_ids']
    if len(ids) != len(set(ids)) or any(i < 0 or i >= len(raw['schedules_list']) for i in ids):
        raise ValueError('Invalid schedule membership: ' + row['name'])
    initial = float(raw['initial_execution_time'])
    if not math.isfinite(initial) or initial <= 0:
        raise ValueError('Invalid initial execution time')
    labels = []
    for i in ids:
        times = np.asarray(raw['schedules_list'][i]['execution_times'], dtype=np.float64)
        if not len(times) or not np.isfinite(times).all() or (times <= 0).any():
            raise ValueError('Invalid execution times for accepted schedule')
        labels.append(max(.01, initial / float(times.min())))
    values = np.asarray(labels, dtype='<f4')
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Invalid FP32 target')
    if hashlib.sha256(values.tobytes()).hexdigest() != row['target_fp32_sha256']:
        raise ValueError('Raw/processed target mismatch: ' + row['name'])
    return values


def validate_tree(tree, graph):
    edges = set()
    def visit(node, parent=None):
        i = int(node['loop_index'])
        if parent is not None:
            edges.add((parent, i))
        if node['has_comps']:
            edges.update((i, graph['n_loops'] + int(c)) for c in node['computations_indices'])
        for child in node['child_list']:
            visit(child, i)
    for root in tree['roots']:
        visit(root)
    if edges != set(map(tuple, graph['edges'][0])):
        raise ValueError('Annotated graph and upstream tree use different node indexing')


def extract_program(name, raw, row, feature_module=None):
    u = feature_module or upstream()
    targets = checked_targets(raw, row)
    template = u.get_representation_template(raw, train_device='cpu')
    if u.get_tree_footprint(template[0]) != row['footprint']:
        raise ValueError('Raw/processed tree footprint mismatch: ' + name)
    annotation = raw['program_annotation']
    graph = graph_from_annotation(annotation)
    validate_tree(template[0], graph)
    payload = dict(expressions=program_templates(annotation), graph=buffer_graph(annotation))
    cs, ls, expr = [], [], None
    for schedule_id in row['schedule_ids']:
        c, l, e = u.get_schedule_representation(annotation, raw['schedules_list'][schedule_id], *template[1:])
        if not all(torch.isfinite(t).all() for t in (c, l, e)):
            raise ValueError('Nonfinite upstream features: ' + name)
        if expr is None:
            expr = e[0]
        elif not torch.equal(expr, e[0]):
            raise ValueError('Expression features changed between schedules')
        cs.append(c); ls.append(l)
    record = dict(tree=template[0], comps=torch.cat(cs), loops=torch.cat(ls), expr=expr,
                  labels=torch.from_numpy(targets.copy()), adjacency=graph['adjacency'],
                  metadata=[dict(name=name, schedule_id=i, sample_id=f'{name}:{i}')
                            for i in row['schedule_ids']], topology=payload)
    # Check expression edges/features and graph node indexing before publishing.
    base = collate_records([(record, 0)])
    hetero_batch(base, SimpleNamespace(get=lambda _: tensor_template(payload)))
    full_graph_batch(base, SimpleNamespace(get=lambda _: payload['expressions']))
    info = dict(name=name, samples=len(targets), footprint=row['footprint'],
                schedule_ids=row['schedule_ids'], target_fp32_sha256=row['target_fp32_sha256'],
                annotation_sha256=annotation_hash(annotation),
                raw_program_sha256=hashlib.sha256(json.dumps(raw, sort_keys=True,
                    separators=(',', ':'), allow_nan=True).encode()).hexdigest(),
                nodes=graph['n_loops']+graph['n_comps'], computations=graph['n_comps'],
                max_expression_nodes=max(len(t['features']) for t in payload['expressions']))
    return record, info


def _prepare_part(task):
    number, names = task
    path = _out / f'part_{number:03d}.sqlite'
    manifest = path.with_suffix('.json')
    if manifest.exists():
        entry = json.loads(manifest.read_text())
        if entry['identity'] != _identity or entry['names'] != names or sha256(path) != entry['sha256']:
            raise ValueError('Completed shard failed its integrity check')
        return entry
    if path.exists():
        raise ValueError('Unmanifested shard exists; preserve and investigate: ' + str(path))
    # Retain failed temporary files for diagnosis; never replace a published shard.
    tmp = path.with_name(path.name + f'.building-{os.getpid()}-{time.time_ns()}')
    db = sqlite3.connect(tmp)
    db.execute('CREATE TABLE programs (name TEXT PRIMARY KEY, info TEXT, payload BLOB, sha256 TEXT)')
    u = upstream()
    count = 0
    started = time.monotonic()
    try:
        for n, name in enumerate(names, 1):
            record, info = extract_program(name, _raw[name], _rows[name], u)
            blob = zlib.compress(pickle.dumps(record, protocol=5), level=1)
            digest = hashlib.sha256(blob).hexdigest()
            db.execute('INSERT INTO programs VALUES (?,?,?,?)',
                       (name, json.dumps(info, separators=(',', ':')), blob, digest))
            count += info['samples']
            if n % 50 == 0 or n == len(names):
                db.commit()
                progress = dict(phase='features', shard=number, programs=n, total=len(names),
                                samples=count, seconds=time.monotonic()-started, updated_unix=time.time())
                atomic_json(progress, _out / f'part_{number:03d}.progress.json')
                print(json.dumps(progress), flush=True)
        db.commit()
    finally:
        db.close()
    os.link(tmp, path)  # Fails if a different process published this path.
    # Temporary hard link intentionally retained until operator review; no data deletion.
    entry = dict(file=path.name, identity=_identity, names=names, programs=len(names), samples=count,
                 sha256=sha256(path), bytes=path.stat().st_size, seconds=time.monotonic()-started)
    atomic_json(entry, manifest)
    return entry


def prepare(source, audit, destination, workers=8, parts=32, min_functions=2):
    global _raw, _rows, _out, _identity
    if os.uname().nodename.startswith('login') and not os.getenv('SLURM_JOB_ID'):
        raise RuntimeError('Raw preparation must run on an allocated compute node')
    torch.set_num_threads(1)
    _out = Path(destination)
    _out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with open(_out / 'prepare.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        identity = dict(schema=SCHEMA, source_sha256=sha256(source), audit_sha256=sha256(audit),
                        feature_sha256=UPSTREAM_SHA256, builder_sha256=sha256(__file__),
                        min_functions=min_functions, parts=parts,
                        topology_builders={f: sha256(Path(__file__).parents[1] / f)
                            for f in ('graph.py','full_gnn_data.py','hetero_data.py','data/features.py')})
        identity_path = _out / 'identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError('Preparation identity changed; use a new version')
        atomic_json(identity, identity_path)
        _identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        atomic_json(dict(phase='loading_raw', updated_unix=time.time()), _out / 'progress.json')
        with open(source, 'rb') as stream:
            _raw = pickle.load(stream)  # Trusted owner-supplied local research dataset.
        audit_rows = read_audit(audit)
        if not set(audit_rows).issubset(_raw):
            raise ValueError('Raw file does not contain every audited program')
        # Verify all pre-batch-filter targets, not only the selected training rows.
        for name, row in audit_rows.items():
            checked_targets(_raw[name], row)
        _rows = select_members(audit_rows, min_functions)
        membership = dict(raw_programs=len(_raw), raw_schedules=sum(len(p['schedules_list']) for p in _raw.values()),
                          audited_programs=len(audit_rows), audited_schedules=sum(r['schedules'] for r in audit_rows.values()),
                          selected_programs=len(_rows), selected_schedules=sum(r['schedules'] for r in _rows.values()),
                          raw_programs_absent_from_processed_audit=sorted(set(_raw)-set(audit_rows)),
                          excluded_by_batch_filter=sorted(set(audit_rows)-set(_rows)),
                          min_functions=min_functions, all_audited_targets_match=True)
        atomic_json(membership, _out / 'membership.json')
        print(json.dumps(membership, default=str)[:2000], flush=True)
        names = sorted(_rows)
        chunks = [(i, names[i::parts]) for i in range(parts)]
        # Linux fork shares the immutable loaded raw dataset. Workers write separate DBs.
        entries = []
        with mp.get_context('fork').Pool(workers) as pool:
            for entry in pool.imap_unordered(_prepare_part, chunks):
                entries.append(entry)
                atomic_json(dict(phase='features', completed_shards=len(entries), total_shards=parts,
                                 completed_samples=sum(x['samples'] for x in entries),
                                 total_samples=membership['selected_schedules'], updated_unix=time.time()),
                            _out / 'progress.json')
        if sum(e['samples'] for e in entries) != membership['selected_schedules']:
            raise ValueError('Prepared sample count changed')
        atomic_json(dict(schema=SCHEMA, identity=identity, membership=membership,
                         parts=sorted(entries, key=lambda e:e['file']), seconds=time.monotonic()-started),
                    _out / 'manifest.json')
        atomic_json(dict(phase='prepared', samples=membership['selected_schedules'], updated_unix=time.time()),
                    _out / 'progress.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--parts', type=int, default=32)
    p.add_argument('--min-functions', type=int, default=2)
    a = p.parse_args()
    prepare(a.source, a.audit, a.out, a.workers, a.parts, a.min_functions)
