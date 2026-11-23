"""Six-relation graphs from annotations only; no labels or timing inputs."""
from dataclasses import dataclass
import torch

RELATIONS = ("contains", "contained_by", "next_sibling", "previous_sibling",
             "shared_buffer_forward", "shared_buffer_backward")


def graph_from_annotation(annotation):
    loops = list(annotation["iterators"])
    comps = sorted(annotation["computations"],
                   key=lambda c: annotation["computations"][c]["absolute_order"])
    li = {n: i for i, n in enumerate(loops)}
    ci = {n: len(loops) + i for i, n in enumerate(comps)}
    edges = [set() for _ in RELATIONS]

    def add(a, b, relation):
        if a != b:
            edges[relation].add((a, b))
            edges[relation + 1].add((b, a))

    def comp_order(name):
        return annotation["computations"][name]["absolute_order"]

    def loop_order(name):
        it = annotation["iterators"][name]
        orders = [comp_order(c) for c in it.get("computations_list", [])]
        orders += [loop_order(c) for c in it.get("child_iterators", [])]
        return min(orders, default=float("inf"))

    def siblings(nodes):
        nodes.sort(key=lambda x: (x[1], x[0]))
        for (a, _), (b, _) in zip(nodes, nodes[1:]):
            add(a, b, 2)

    roots = []
    for name, it in annotation["iterators"].items():
        if it.get("parent_iterator") is None:
            roots.append((li[name], loop_order(name)))
        children = [(li[c], loop_order(c)) for c in it.get("child_iterators", [])]
        children += [(ci[c], comp_order(c)) for c in it.get("computations_list", [])]
        for child, _ in children:
            add(li[name], child, 0)
        siblings(children)
    siblings(roots)
    buffers = {}
    for name in comps:
        comp = annotation["computations"][name]
        buffers[name] = {str(comp["write_buffer_id"])} | {
            str(a["buffer_id"]) for a in comp["accesses"]}
    for i, a in enumerate(comps):
        for b in comps[i + 1:]:
            if buffers[a] & buffers[b]:
                add(ci[a], ci[b], 4)

    n = len(loops) + len(comps)
    adjacency = torch.zeros(6, n, n)
    for relation, pairs in enumerate(edges):
        for source, destination in pairs:
            adjacency[relation, destination, source] = 1
    adjacency /= adjacency.sum(-1, keepdim=True).clamp_min(1)
    return {"edges": [sorted(e) for e in edges], "adjacency": adjacency,
            "n_loops": len(loops), "n_comps": len(comps)}


@dataclass
class GraphBatch:
    inputs: tuple
    adjacency: torch.Tensor
    labels: torch.Tensor
    metadata: list

    def to(self, device):
        return GraphBatch((self.inputs[0],) + tuple(t.to(device) for t in self.inputs[1:]),
                          self.adjacency.to(device), self.labels.to(device), self.metadata)

    def slice(self, start, end):
        tree, first, tags, third, loops, expr = self.inputs
        b, c = first.shape[:2]
        tags = tags.reshape(b, c, 4, 16)[start:end].reshape(-1, 4, 16)
        return GraphBatch((tree, first[start:end], tags, third[start:end], loops[start:end],
                           expr[start:end]), self.adjacency[start:end],
                          self.labels[start:end], self.metadata[start:end])


def collate_records(items):
    """Items are (program record, schedule index), all with the same tree footprint."""
    comps = torch.stack([r["comps"][i] for r, i in items])
    b, c = comps.shape[:2]
    inputs = (items[0][0]["tree"], comps[:, :, :33].contiguous(),
              comps[:, :, 33:97].reshape(b * c, 4, 16), comps[:, :, 97:].contiguous(),
              torch.stack([r["loops"][i] for r, i in items]),
              torch.stack([r["expr"] for r, _ in items]))
    return GraphBatch(inputs, torch.stack([r["adjacency"] for r, _ in items]),
                      torch.stack([r["labels"][i] for r, i in items]),
                      [r["metadata"][i] for r, i in items])
