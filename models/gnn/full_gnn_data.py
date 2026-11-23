"""Versioned expression topology sidecars; existing feature shards are read-only."""
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
from .constants import MAX_EXPR_LEN, ExpressionTooLongException
from .data.features import expr_op_repr
from .data.looperset import stream_jsonl_gz
from .io_utils import sha256


def expression_template(node, dtype):
    features, edges = [], []

    def visit(current, slot=0, siblings=0):
        children = [visit(child, i, len(current['children']))
                    for i, child in enumerate(current['children'])]
        index = len(features)
        structural = [slot / max(1, siblings - 1), math.log1p(slot), math.log1p(siblings)]
        features.append(expr_op_repr(current['expr_type'], dtype) + structural)
        edges.extend((index, child) for child in children)
        return index

    root = visit(node)
    if len(features) > MAX_EXPR_LEN:
        raise ExpressionTooLongException(len(features))
    return {'features': features, 'edges': edges, 'root': root}


def program_templates(annotation):
    names = sorted(annotation['computations'],
                   key=lambda name: annotation['computations'][name]['absolute_order'])
    return [expression_template(annotation['computations'][name]['expression_representation'],
                                annotation['computations'][name]['data_type']) for name in names]


def index_signature(source):
    return {'schema': 1, 'source_sha256': sha256(source), 'builder_sha256': sha256(__file__)}


def build_expression_index(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    signature = index_signature(source)
    with open(destination.with_suffix('.lock'), 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if destination.exists():
            index = ExpressionIndex(destination)
            try:
                if index.meta['signature'] != signature:
                    raise ValueError('Stale expression index; choose a new version')
                return index.meta
            finally:
                index.close()
        temporary = destination.with_name(destination.name + '.building-' + uuid.uuid4().hex)
        connection = sqlite3.connect(temporary)
        try:
            connection.execute('CREATE TABLE programs (name TEXT PRIMARY KEY, payload TEXT NOT NULL)')
            connection.execute('CREATE TABLE meta (payload TEXT NOT NULL)')
            accepted = 0
            rejected = []
            for number, program in enumerate(stream_jsonl_gz(source), 1):
                try:
                    templates = program_templates(program['program_annotation'])
                except ExpressionTooLongException:
                    rejected.append(program['program_name'])
                    continue
                connection.execute('INSERT INTO programs VALUES (?, ?)',
                                   (program['program_name'], json.dumps(templates)))
                accepted += 1
                if number % 1000 == 0:
                    connection.commit()
                    print(f'Expression topology: {number} programs read', flush=True)
            meta = {'signature': signature, 'programs': accepted,
                    'excluded_expression_too_long': rejected}
            connection.execute('INSERT INTO meta VALUES (?)', (json.dumps(meta),))
            connection.commit()
        finally:
            connection.close()
        os.replace(temporary, destination)
        return meta


class ExpressionIndex:
    def __init__(self, path, expected_signature=None):
        self.path = Path(path)
        self.connection = sqlite3.connect(self.path.resolve().as_uri() + '?mode=ro', uri=True)
        self.connection.execute('PRAGMA query_only = ON')
        self.meta = json.loads(self.connection.execute('SELECT payload FROM meta').fetchone()[0])
        if expected_signature is not None and self.meta['signature'] != expected_signature:
            self.close()
            raise ValueError('Stale expression index')
        self.get = lru_cache(maxsize=256)(self._get)

    def _get(self, name):
        result = self.connection.execute('SELECT payload FROM programs WHERE name=?', (name,)).fetchone()
        if result is None:
            raise ValueError(f'Missing expression topology for {name}')
        return json.loads(result[0])

    def close(self):
        if hasattr(self, 'get'):
            self.get.cache_clear()
        self.connection.close()


def pack_expressions(templates):
    length = max(len(t['features']) for t in templates)
    x = torch.zeros(len(templates), length, 14)
    adjacency = torch.zeros(len(templates), 2, length, length)
    mask = torch.zeros(len(templates), length, dtype=torch.bool)
    roots = torch.zeros(len(templates), dtype=torch.long)
    for i, template in enumerate(templates):
        n = len(template['features'])
        x[i, :n] = torch.tensor(template['features'])
        mask[i, :n] = True
        roots[i] = template['root']
        for parent, child in template['edges']:
            adjacency[i, 0, child, parent] = 1
            adjacency[i, 1, parent, child] = 1
    return x, adjacency, mask, roots


@dataclass
class FullGraphBatch:
    base: object
    expressions: tuple

    def to(self, device):
        return FullGraphBatch(self.base.to(device), tuple(t.to(device) for t in self.expressions))


def full_graph_batch(base, index):
    b, c = base.inputs[1].shape[:2]
    templates = []
    for row in base.metadata:
        current = index.get(row['name'])
        if len(current) != c:
            raise ValueError('Expression computation count mismatch')
        templates.extend(current)
    expressions = pack_expressions(templates)
    # Confirm that the sidecar only adds topology to the intended feature rows.
    old = base.inputs[-1].reshape(b * c, MAX_EXPR_LEN, 11)
    features, _, mask, _ = expressions
    if not torch.equal(old[:, :features.shape[1]], features[:, :, :11]):
        raise ValueError('Expression sidecar does not match cached computation features')
    if torch.count_nonzero(old[:, features.shape[1]:]):
        raise ValueError('Expression sidecar omitted real nodes')
    return FullGraphBatch(base, expressions)
