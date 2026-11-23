"""Loop/computation/buffer incidence graphs and unchanged local expression graphs."""
import fcntl
import json
import math
import os
import sqlite3
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import torch
from torch.nn import functional as F
from .constants import MAX_EXPR_LEN, ExpressionTooLongException
from .data.looperset import stream_jsonl_gz
from .full_gnn_data import program_templates, pack_expressions
from .graph import graph_from_annotation
from .io_utils import sha256

RELATIONS = ('contains', 'contained_by', 'next_sibling', 'previous_sibling',
             'reads_buffer', 'read_by', 'writes_buffer', 'written_by')


def buffer_graph(annotation):
    original = graph_from_annotation(annotation)
    loops, comps = original['n_loops'], original['n_comps']
    names = sorted(annotation['computations'],
                   key=lambda name: annotation['computations'][name]['absolute_order'])
    buffers, readers, writers, accesses, reductions = {}, {}, {}, {}, {}
    def register(value):
        key = str(value)
        if key not in buffers:
            buffers[key] = loops + comps + len(buffers)
            readers[key], writers[key], accesses[key], reductions[key] = set(), set(), 0, False
        return key
    # First-use order makes the graph independent of the spelling of buffer IDs.
    edges = [set(map(tuple, pairs)) for pairs in original['edges'][:4]] + [set() for _ in range(4)]
    for i, name in enumerate(names):
        comp, node = annotation['computations'][name], loops + i
        for access in comp['accesses']:
            key = register(access['buffer_id'])
            readers[key].add(i)
            accesses[key] += 1
            reductions[key] |= bool(access.get('access_is_reduction', False))
            edges[4].add((node, buffers[key])); edges[5].add((buffers[key], node))
        key = register(comp['write_buffer_id'])
        writers[key].add(i)
        reductions[key] |= bool(comp.get('comp_is_reduction', False))
        edges[6].add((node, buffers[key])); edges[7].add((buffers[key], node))
    features = [[math.log1p(len(readers[key])), math.log1p(len(writers[key])),
                 math.log1p(accesses[key]), float(reductions[key])] for key in buffers]
    return {'n_loops': loops, 'n_comps': comps, 'buffer_features': features,
            'edges': [sorted(pairs) for pairs in edges], 'buffer_ids_for_audit_only': list(buffers)}


def index_signature(source, accepted_sha256=None):
    root = Path(__file__).parent
    return {'schema': 1, 'source_sha256': sha256(source), 'accepted_sha256': accepted_sha256,
            'builders': {name: sha256(root / name) for name in
                         ('hetero_data.py', 'full_gnn_data.py', 'graph.py', 'data/features.py')}}


def build_topology_index(source, destination, accepted_manifest=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    signature = index_signature(source, sha256(accepted_manifest) if accepted_manifest else None)
    allowed_names = None
    if accepted_manifest:
        with open(accepted_manifest) as stream:
            allowed_names = {json.loads(line)['name'] for line in stream}
    with open(destination.with_suffix('.lock'), 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if destination.exists():
            with TopologyIndex(destination, signature) as index:
                return index.meta
        temporary = destination.with_name(destination.name + '.building-' + uuid.uuid4().hex)
        connection = sqlite3.connect(temporary)
        try:
            connection.execute('CREATE TABLE programs (name TEXT PRIMARY KEY, payload TEXT NOT NULL)')
            connection.execute('CREATE TABLE meta (payload TEXT NOT NULL)')
            count, excluded, largest = 0, [], []
            for number, program in enumerate(stream_jsonl_gz(source), 1):
                if allowed_names is not None and program['program_name'] not in allowed_names:
                    continue
                try:
                    expressions = program_templates(program['program_annotation'])
                except ExpressionTooLongException:
                    excluded.append(program['program_name'])
                    continue
                payload = {'expressions': expressions, 'graph': buffer_graph(program['program_annotation'])}
                connection.execute('INSERT INTO programs VALUES (?, ?)',
                                   (program['program_name'], json.dumps(payload)))
                count += 1
                graph = payload['graph']
                n = graph['n_loops'] + graph['n_comps'] + len(graph['buffer_features'])
                expression_nodes = max(len(t['features']) for t in expressions)
                largest.append({'name': program['program_name'], 'nodes': n, 'computations': graph['n_comps'],
                                'expression_nodes': expression_nodes,
                                'score': 8 * n * n + graph['n_comps'] * expression_nodes ** 2})
                largest = sorted(largest, key=lambda x: (-x['score'], x['name']))[:10]
                if number % 1000 == 0:
                    connection.commit()
                    print(f'TOPOLOGY programs={number}', flush=True)
            if allowed_names is not None and count != len(allowed_names):
                raise ValueError('The graph index did not cover every accepted program')
            meta = {'signature': signature, 'programs': count, 'expression_too_long': excluded,
                    'largest_programs': largest,
                    'relations': RELATIONS, 'buffer_features': ['log1p_readers', 'log1p_writers',
                                                               'log1p_read_accesses', 'reduction_access']}
            connection.execute('INSERT INTO meta VALUES (?)', (json.dumps(meta),))
            connection.commit()
        finally:
            connection.close()
        os.replace(temporary, destination)
        return meta


class TopologyIndex:
    def __init__(self, path, expected_signature=None):
        self.path = Path(path)
        self.connection = sqlite3.connect(self.path.resolve().as_uri() + '?mode=ro', uri=True)
        self.connection.execute('PRAGMA query_only=ON')
        self.meta = json.loads(self.connection.execute('SELECT payload FROM meta').fetchone()[0])
        if expected_signature is not None and self.meta['signature'] != expected_signature:
            self.connection.close()
            raise ValueError('Stale graph topology index')
        self.get = lru_cache(maxsize=512)(self._get)

    def _get(self, name):
        row = self.connection.execute('SELECT payload FROM programs WHERE name=?', (name,)).fetchone()
        if row is None:
            raise ValueError(f'Missing graph topology for {name}')
        return tensor_template(json.loads(row[0]))

    def close(self):
        self.get.cache_clear()
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def tensor_template(payload):
    graph = payload['graph']
    n = graph['n_loops'] + graph['n_comps'] + len(graph['buffer_features'])
    edges = torch.zeros(8, n, n, dtype=torch.bool)
    for relation, pairs in enumerate(graph['edges']):
        for source, destination in pairs:
            if not 0 <= source < n or not 0 <= destination < n or source == destination:
                raise ValueError('Invalid incidence-graph edge')
            edges[relation, destination, source] = True
    return {'expressions': pack_expressions(payload['expressions']), 'edges': edges,
            'buffer_features': torch.tensor(graph['buffer_features'], dtype=torch.float32).reshape(-1, 4),
            'n_loops': graph['n_loops'], 'n_comps': graph['n_comps']}


@dataclass
class HeteroBatch:
    base: object
    expressions: tuple
    buffer_features: torch.Tensor
    edges: torch.Tensor
    node_mask: torch.Tensor
    pool_mask: torch.Tensor

    def to(self, device):
        return HeteroBatch(self.base.to(device), tuple(t.to(device) for t in self.expressions),
                           self.buffer_features.to(device), self.edges.to(device),
                           self.node_mask.to(device), self.pool_mask.to(device))


def hetero_batch(base, index):
    b, c = base.inputs[1].shape[:2]
    loops = base.inputs[4].shape[1]
    templates = [index.get(row['name']) for row in base.metadata]
    if any(t['n_loops'] != loops or t['n_comps'] != c for t in templates):
        raise ValueError('Graph topology does not match the cached footprint')
    max_expr = max(t['expressions'][0].shape[1] for t in templates)
    max_buffers = max(len(t['buffer_features']) for t in templates)
    n = loops + c + max_buffers
    expr_x, expr_a, expr_m, expr_r, buffer_x, edges, masks = [], [], [], [], [], [], []
    for t in templates:
        x, a, m, r = t['expressions']
        if len(x) != c:
            raise ValueError('Expression computation count mismatch')
        pad = max_expr - x.shape[1]
        expr_x.append(F.pad(x, (0, 0, 0, pad)))
        expr_a.append(F.pad(a, (0, pad, 0, pad)))
        expr_m.append(F.pad(m, (0, pad)))
        expr_r.append(r)
        missing = max_buffers - len(t['buffer_features'])
        buffer_x.append(F.pad(t['buffer_features'], (0, 0, 0, missing)))
        edges.append(F.pad(t['edges'], (0, missing, 0, missing)))
        masks.append(torch.arange(n) < n - missing)
    expressions = tuple(torch.cat(values) for values in (expr_x, expr_a, expr_m, expr_r))
    old = base.inputs[-1].reshape(b * c, MAX_EXPR_LEN, 11)
    if not torch.equal(old[:, :max_expr], expressions[0][:, :, :11]) or torch.count_nonzero(old[:, max_expr:]):
        raise ValueError('Expression topology does not match cached features')
    edges = torch.stack(edges)
    if not torch.equal(edges[:, :4, :loops+c, :loops+c], base.adjacency[:, :4].bool()):
        raise ValueError('Containment/order topology does not match cached features')
    pool_mask = (torch.arange(n) < loops + c).expand(b, -1)
    return HeteroBatch(base, expressions, torch.stack(buffer_x), edges, torch.stack(masks), pool_mask)
