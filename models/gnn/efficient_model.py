"""The frozen HeteroGAT mathematics with prevalidated inputs and batched relations."""
import torch
from torch.nn import functional as F
from .hetero_gat import HeteroGATCostModel, create_hetero_gat
from .full_gnn import transformation_graph


def tensor_batch(batch):
    """Prepare invariant graph inputs on CPU once; every tensor has a sample axis."""
    _, first, tags, third, loops, _ = batch.base.inputs
    b, c = first.shape[:2]
    expressions = tuple(t.reshape(b, c, *t.shape[1:]) for t in batch.expressions)
    transformations = tuple(t.reshape(b, c, *t.shape[1:]) for t in transformation_graph(tags))
    return tuple(t.contiguous().clone() for t in (
        first, third, loops, *expressions, *transformations, batch.buffer_features,
        batch.edges, batch.node_mask, batch.pool_mask, batch.base.labels))


def batched_attention(layer, nodes, edges, mask):
    b, n, width = nodes.shape
    r, h, d = len(layer.relations), layer.heads, layer.head_width
    # One matrix multiply replaces eight launches; all existing weights remain trainable.
    weight = torch.cat([relation.weight for relation in layer.relations], dim=0)
    projected = F.linear(nodes, weight).reshape(b, n, r, h, d).permute(0, 2, 3, 1, 4)
    source = (projected * layer.source_attention[None, :, :, None]).sum(-1)
    destination = (projected * layer.destination_attention[None, :, :, None]).sum(-1)
    logits = F.leaky_relu(destination.unsqueeze(-1) + source.unsqueeze(-2), .2)
    allowed = edges & mask[:, None, :, None] & mask[:, None, None, :]
    weights = logits.masked_fill(~allowed[:, :, None], torch.finfo(logits.dtype).min).softmax(-1)
    weights = weights * allowed[:, :, None]
    messages = torch.matmul(weights, projected).permute(0, 1, 3, 2, 4).reshape(b, r, n, width)
    # Preserve the reference's relation summation order.
    result = layer.self_linear(nodes)
    for relation in range(r):
        result = result + messages[:, relation]
    return layer.norm(nodes + layer.dropout(F.elu(result))) * mask.unsqueeze(-1)


class EfficientHeteroGAT(HeteroGATCostModel):
    def forward(self, tensors):
        first, third, loops = tensors[:3]
        b, c = first.shape[:2]
        expression = tuple(t.flatten(0, 1) for t in tensors[3:7])
        transformation = tuple(t.flatten(0, 1) for t in tensors[7:11])
        expr = self.expression_encoder(expression).reshape(b, c, 100)
        affine = self.transformation_encoder(transformation).reshape(b, c, 200)
        comps = torch.cat((first, affine, third, expr), dim=-1)
        for layer, dropout in zip(self.comp_embedding_layers, self.comp_embedding_dropouts):
            comps = dropout(self.ELU(layer(comps)))
        nodes = torch.cat((self.loop_projection(loops) + self.node_types.weight[0],
                           self.comp_projection(comps) + self.node_types.weight[1],
                           self.buffer_projection(tensors[11]) + self.node_types.weight[2]), dim=1)
        mask = tensors[13]
        nodes = nodes * mask.unsqueeze(-1)
        for layer in self.graph_layers:
            nodes = batched_attention(layer, nodes, tensors[12], mask)
        return self.readout(nodes, tensors[14])


def create_efficient_model():
    reference = create_hetero_gat()
    # Constructing the replacement must not advance the reference initialization RNG.
    with torch.random.fork_rng(devices=[]):
        model = EfficientHeteroGAT()
    model.load_state_dict(reference.state_dict(), strict=True)
    return model
