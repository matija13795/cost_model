"""Small annotated programs; never used as experiment data."""
import torch
from types import SimpleNamespace
from models.gnn.data import features as F
from models.gnn.data.looperset import normalize_schedule
from models.gnn.graph import graph_from_annotation, collate_records
from models.gnn.full_gnn_data import program_templates
from models.gnn.hetero_data import buffer_graph, tensor_template, hetero_batch

def program(name="fixture"):
    annotation = {"memory_size": "1", "iterators": {}, "computations": {}}
    for n in range(2):
        outer, inner, comp = f"i{2*n}", f"i{2*n+1}", f"comp{n}"
        annotation["iterators"][outer] = {"lower_bound": "0", "upper_bound": "8", "parent_iterator": None, "child_iterators": [inner], "computations_list": []}
        annotation["iterators"][inner] = {"lower_bound": "0", "upper_bound": "8", "parent_iterator": outer, "child_iterators": [], "computations_list": [comp]}
        annotation["computations"][comp] = {
            "absolute_order": n, "iterators": [outer, inner], "comp_is_reduction": False,
            "write_buffer_id": n, "write_access_relation": f"{{ {comp}[{outer}, {inner}] -> buf{n}[{outer}, {inner}] }}",
            "data_type": "float64", "accesses": [{"buffer_id": 0, "access_is_reduction": False, "access_matrix": [[1,0,0],[0,1,0]]}],
            "expression_representation": {"expr_type": "access", "children": []}}
    return {"program_name": name, "program_annotation": annotation, "initial_execution_time": 10.,
            "schedules_list": [{"transformations_list": [], "execution_times": [10.], "schedule_str": ""},
                               {"transformations_list": [{"type": "parallelization", "computations": ["comp0"], "loop_levels": [0], "parameters": []}], "execution_times": [5.], "schedule_str": "P"}]}


def fixture(name='p', extra_buffer=False):
    p = program(name)
    if extra_buffer:
        p['program_annotation']['computations']['comp1']['accesses'][0]['buffer_id'] = 7
    annotation = p['program_annotation']
    template = F.build_representation_template(p)
    values = [F.fill_schedule_representation(annotation, normalize_schedule(annotation, s), template, 1)
              for s in p['schedules_list']]
    record = dict(tree=template.tree, comps=torch.cat([v[0] for v in values]),
                  loops=torch.cat([v[1] for v in values]), expr=torch.tensor(template.expr_repr,dtype=torch.float32),
                  labels=torch.tensor([1., 2.]), adjacency=graph_from_annotation(annotation)['adjacency'],
                  metadata=[dict(name=name,sample_id=f'{name}:{i}') for i in range(2)])
    topology = tensor_template({'graph': buffer_graph(annotation), 'expressions': program_templates(annotation)})
    return record, topology


def sample_batch():
    record, template = fixture()
    return hetero_batch(collate_records([(record,0),(record,1)]), SimpleNamespace(get=lambda _:template))
