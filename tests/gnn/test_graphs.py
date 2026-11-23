import copy
from types import SimpleNamespace
import pytest
import torch
from .fixtures import program, fixture, sample_batch
from models.gnn.graph import collate_records
from models.gnn.hetero_data import buffer_graph, hetero_batch
from models.gnn.hetero_gat import create_hetero_gat, RelationalAttentionLayer
from models.gnn.efficient_model import create_efficient_model, tensor_batch
from models.gnn.full_gnn_data import full_graph_batch, program_templates
from models.gnn.factory import build_model

def test_graph_has_explicit_buffers_and_separate_read_write_relations():
    g = buffer_graph(program()['program_annotation'])
    assert g['n_loops'] == 4 and g['n_comps'] == 2
    assert g['buffer_ids_for_audit_only'] == ['0', '1']
    assert set(g['edges'][4]) == {(4, 6), (5, 6)}
    assert set(g['edges'][6]) == {(4, 6), (5, 7)}
    for r in (0, 2, 4, 6):
        assert set(g['edges'][r+1]) == {(b, a) for a, b in g['edges'][r]}
    assert (4, 5) not in g['edges'][4]  # The previous shared-buffer clique is gone.
    renamed = program()
    for c in renamed['program_annotation']['computations'].values():
        c['write_buffer_id'] = 'renamed_' + str(c['write_buffer_id'])
        for access in c['accesses']:
            access['buffer_id'] = 'renamed_' + str(access['buffer_id'])
    other = buffer_graph(renamed['program_annotation'])
    assert other['edges'] == g['edges'] and other['buffer_features'] == g['buffer_features']
    changed = copy.deepcopy(program())
    changed['initial_execution_time'] = 9999
    changed['schedules_list'][1]['execution_times'] = [.001]
    assert buffer_graph(changed['program_annotation']) == g

def test_attention_normalizes_neighbors_and_handles_empty_relations():
    layer = RelationalAttentionLayer(width=8, heads=2, dropout=0)
    nodes = torch.randn(2, 4, 8)
    edges = torch.zeros(2, 8, 4, 4, dtype=torch.bool)
    edges[:, 0, 2, :2] = True
    mask = torch.tensor([[True, True, True, False]] * 2)
    with torch.no_grad():
        layer.source_attention.zero_()
        layer.destination_attention.zero_()
    _, weights = layer.attention(nodes, edges, mask, 0)
    torch.testing.assert_close(weights[:, :, 2, :2], torch.full((2, 2, 2), .5))
    assert torch.count_nonzero(weights[:, :, :2]) == 0
    empty = layer(nodes, torch.zeros_like(edges), mask)
    assert torch.isfinite(empty).all() and torch.count_nonzero(empty[:, 3]) == 0
    # Nonzero learned scores must be able to give two neighbors different weights.
    with torch.no_grad():
        layer.relations[0].weight.copy_(torch.eye(8))
        layer.source_attention.fill_(1)
        nodes[:, 0].zero_(); nodes[:, 1].fill_(1)
    _, learned = layer.attention(nodes, edges, mask, 0)
    assert torch.all(learned[:, :, 2, 1] > learned[:, :, 2, 0])

def test_batch_isolation_padding_and_finite_gradients():
    model = create_hetero_gat().eval()
    r, t = fixture('a')
    r2, t2 = fixture('b', extra_buffer=True)
    index = SimpleNamespace(get=lambda name: {'a': t, 'b': t2}[name])
    one = hetero_batch(collate_records([(r, 0)]), index)
    two = hetero_batch(collate_records([(r, 0), (r2, 1)]), index)
    assert two.edges.shape[-1] == one.edges.shape[-1] + 1
    torch.testing.assert_close(model(one), model(two)[:1], atol=2e-5, rtol=2e-5)
    two.buffer_features[0, -1].fill_(1000)
    torch.testing.assert_close(model(one), model(two)[:1], atol=2e-5, rtol=2e-5)
    model(two).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

def test_program_node_reindexing_and_padding_preserve_readout():
    model, batch = create_hetero_gat().eval(), sample_batch()
    nodes = model.encode_nodes(batch)
    result = model.readout(model.message_pass(nodes, batch.edges, batch.node_mask), batch.pool_mask)
    permutation = torch.randperm(nodes.shape[1])
    reindexed = model.readout(model.message_pass(nodes[:, permutation],
        batch.edges[:, :, permutation][:, :, :, permutation], batch.node_mask[:, permutation]),
        batch.pool_mask[:, permutation])
    torch.testing.assert_close(result, reindexed, atol=2e-5, rtol=2e-5)
    padded = model.readout(model.message_pass(torch.nn.functional.pad(nodes, (0, 0, 0, 2), value=33),
        torch.nn.functional.pad(batch.edges, (0, 2, 0, 2)),
        torch.nn.functional.pad(batch.node_mask, (0, 2))), torch.nn.functional.pad(batch.pool_mask, (0, 2)))
    torch.testing.assert_close(result, padded, atol=2e-5, rtol=2e-5)

@pytest.mark.parametrize('name,parameters,recurrent',[
    ('lstm',2109623,5),('hybrid_rgcn',2128703,2),('full_rgcn',2125071,0),('heterogat',2114663,0)])
def test_preserved_models_and_batch_independence(name,parameters,recurrent):
    torch.manual_seed(42)
    model=build_model(name).eval()
    assert sum(p.numel() for p in model.parameters())==parameters
    assert sum(isinstance(m,torch.nn.LSTM) for m in model.modules())==recurrent
    record, template=fixture()
    base=collate_records([(record,0),(record,1)])
    def adapt(b):
        if name=='lstm':return b.inputs
        if name=='hybrid_rgcn':return b
        if name=='full_rgcn':return full_graph_batch(b,SimpleNamespace(get=lambda _:program_templates(program()['program_annotation'])))
        return hetero_batch(b,SimpleNamespace(get=lambda _:template))
    with torch.inference_mode():
        torch.testing.assert_close(model(adapt(base.slice(0,1))),model(adapt(base))[:1],atol=3e-4,rtol=3e-4)


def test_efficient_predictions_gradients_and_target_isolation():
    torch.manual_seed(42)
    reference=create_hetero_gat().eval()
    efficient=create_efficient_model().eval()
    efficient.load_state_dict(reference.state_dict(),strict=True)
    batch=sample_batch();tensors=tensor_batch(batch)
    left,right=reference(batch),efficient(tensors)
    torch.testing.assert_close(left,right,atol=3e-4,rtol=3e-4)
    left.sum().backward();right.sum().backward()
    for (name,a),(_,b) in zip(reference.named_parameters(),efficient.named_parameters()):
        assert a.grad is not None and b.grad is not None,name
        torch.testing.assert_close(a.grad,b.grad,atol=5e-5,rtol=2e-3,msg=name)
    torch.testing.assert_close(right,efficient((*tensors[:-1],tensors[-1]*1000)),atol=0,rtol=0)
