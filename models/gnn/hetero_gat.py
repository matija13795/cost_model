"""Fully non-recurrent GNN with buffer-incidence graphs and relational attention."""
import torch
from torch import nn
from torch.nn import functional as F
from .full_gnn import FullGNNCostModel, create_full_gnn, transformation_graph


class RelationalAttentionLayer(nn.Module):
    def __init__(self, width=152, heads=4, relations=8, dropout=.05):
        super().__init__()
        if width % heads:
            raise ValueError('Hidden width must be divisible by attention heads')
        self.width, self.heads, self.head_width = width, heads, width // heads
        self.relations = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(relations)])
        self.source_attention = nn.Parameter(torch.empty(relations, heads, self.head_width))
        self.destination_attention = nn.Parameter(torch.empty(relations, heads, self.head_width))
        self.self_linear = nn.Linear(width, width)
        self.norm, self.dropout = nn.LayerNorm(width), nn.Dropout(dropout)
        for layer in [*self.relations, self.self_linear]:
            nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.source_attention)
        nn.init.xavier_uniform_(self.destination_attention)

    def attention(self, nodes, edges, node_mask, relation):
        b, n, _ = nodes.shape
        projected = self.relations[relation](nodes).view(b, n, self.heads, self.head_width).transpose(1, 2)
        source = (projected * self.source_attention[relation][None, :, None]).sum(-1)
        destination = (projected * self.destination_attention[relation][None, :, None]).sum(-1)
        logits = F.leaky_relu(destination.unsqueeze(-1) + source.unsqueeze(-2), negative_slope=.2)
        allowed = edges[:, relation] & node_mask[:, :, None] & node_mask[:, None, :]
        masked = logits.masked_fill(~allowed[:, None], torch.finfo(logits.dtype).min)
        # An empty neighborhood contributes zero rather than a NaN softmax row.
        weights = masked.softmax(-1) * allowed[:, None]
        return projected, weights

    def forward(self, nodes, edges, node_mask):
        messages = self.self_linear(nodes)
        for relation in range(len(self.relations)):
            projected, weights = self.attention(nodes, edges, node_mask, relation)
            contribution = torch.matmul(weights, projected).transpose(1, 2).reshape_as(nodes)
            messages = messages + contribution
        updated = self.norm(nodes + self.dropout(F.elu(messages)))
        return updated * node_mask.unsqueeze(-1)


class HeteroGATCostModel(FullGNNCostModel):
    def __init__(self, width=152, layers=4, heads=4, dropout=.05):
        super().__init__(dropout)
        self.comp_projection = nn.Linear(180, width)
        self.loop_projection = nn.Linear(8, width)
        self.buffer_projection = nn.Linear(4, width)
        self.node_types = nn.Embedding(3, width)
        self.graph_layers = nn.ModuleList([RelationalAttentionLayer(width, heads, dropout=dropout)
                                           for _ in range(layers)])
        self.to_head = nn.Linear(width, 180)
        for layer in (self.comp_projection, self.loop_projection, self.buffer_projection, self.to_head):
            nn.init.xavier_uniform_(layer.weight)

    def encode_nodes(self, batch):
        _, first, tags, third, loops, _ = batch.base.inputs
        b, c = first.shape[:2]
        expr = self.expression_encoder(batch.expressions).reshape(b, c, 100)
        affine = self.transformation_encoder(transformation_graph(tags)).reshape(b, c, 200)
        comps = torch.cat((first, affine, third, expr), dim=-1)
        for layer, dropout in zip(self.comp_embedding_layers, self.comp_embedding_dropouts):
            comps = dropout(self.ELU(layer(comps)))
        nodes = torch.cat((self.loop_projection(loops) + self.node_types.weight[0],
                           self.comp_projection(comps) + self.node_types.weight[1],
                           self.buffer_projection(batch.buffer_features) + self.node_types.weight[2]), dim=1)
        return nodes * batch.node_mask.unsqueeze(-1)

    def message_pass(self, nodes, edges, node_mask):
        for layer in self.graph_layers:
            nodes = layer(nodes, edges, node_mask)
        return nodes

    def readout(self, nodes, pool_mask):
        # Buffer nodes relay memory interactions; readout retains loop/computation pooling.
        pooled = (nodes * pool_mask.unsqueeze(-1)).sum(1) / pool_mask.sum(1, keepdim=True).clamp_min(1)
        x = self.to_head(pooled)
        for layer, dropout in zip(self.regression_layers, self.regression_dropouts):
            x = dropout(self.ELU(layer(x)))
        return self.LeakyReLU(self.predict(x)[:, 0])

    def forward(self, batch):
        nodes = self.message_pass(self.encode_nodes(batch), batch.edges, batch.node_mask)
        return self.readout(nodes, batch.pool_mask)


def create_hetero_gat():
    reference = create_full_gnn()
    model = HeteroGATCostModel()
    prefixes = ('expression_encoder.', 'transformation_encoder.', 'comp_embedding_layers.',
                'regression_layers.', 'predict.')
    shared = {key: value for key, value in reference.state_dict().items() if key.startswith(prefixes)}
    model.load_state_dict(shared, strict=False)
    return model
