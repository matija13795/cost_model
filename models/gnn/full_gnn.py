"""Hierarchical graph encoders for expressions, affine paths, and programs."""
import torch
from torch import nn
from .gnn import RelationalLayer


class LocalLayer(nn.Module):
    def __init__(self, width, dropout=.05):
        super().__init__()
        self.relations = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(2)])
        self.self_linear = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        for layer in [*self.relations, self.self_linear]:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, nodes, adjacency, mask):
        message = self.self_linear(nodes)
        for relation, layer in enumerate(self.relations):
            message = message + layer(torch.bmm(adjacency[:, relation], nodes))
        nodes = self.norm(nodes + self.dropout(torch.nn.functional.elu(message)))
        return nodes * mask.unsqueeze(-1)


class LocalEncoder(nn.Module):
    def __init__(self, input_size, width, output_size, dropout=.05):
        super().__init__()
        self.project = nn.Linear(input_size, width)
        self.layers = nn.ModuleList([LocalLayer(width, dropout) for _ in range(3)])
        self.readout = nn.Linear(2 * width, output_size)
        nn.init.xavier_uniform_(self.project.weight)
        nn.init.xavier_uniform_(self.readout.weight)

    def forward(self, graph):
        x, adjacency, mask, anchor = graph
        nodes = torch.nn.functional.elu(self.project(x)) * mask.unsqueeze(-1)
        for layer in self.layers:
            nodes = layer(nodes, adjacency, mask)
        selected = nodes[torch.arange(len(nodes), device=nodes.device), anchor]
        return self.readout(torch.cat((nodes.sum(1), selected), dim=-1))


def transformation_graph(tags):
    """Keep sequence order and distinguish an empty sequence from padding."""
    if tags.ndim != 3 or tags.shape[1:] != (4, 16):
        raise ValueError('Expected four padded 16-dimensional affine tags')
    active = tags[:, :, 0] != 0
    lengths = active.sum(1)
    positions = torch.arange(4, device=tags.device)
    if not torch.equal(active, positions[None, :] < lengths[:, None]):
        raise ValueError('Affine tags must precede trailing padding')
    if torch.any(tags[~active] != 0):
        raise ValueError('Nonzero data in padded affine tag')
    if torch.any(active & ~((tags[:, :, 0] >= 1) & (tags[:, :, 0] <= 3))):
        raise ValueError('Unknown affine tag type')
    mask = active.clone()
    mask[:, 0] = True
    position_features = torch.eye(4, device=tags.device, dtype=tags.dtype)[None] * active[:, :, None]
    empty = torch.zeros((*tags.shape[:2], 1), device=tags.device, dtype=tags.dtype)
    empty[:, 0, 0] = (lengths == 0).to(tags.dtype)
    x = torch.cat((tags, position_features, empty), dim=-1)
    adjacency = tags.new_zeros((len(tags), 2, 4, 4))
    valid_edges = (positions[None, 1:] < lengths[:, None]).to(tags.dtype)
    adjacency[:, 0, positions[1:], positions[:-1]] = valid_edges
    adjacency[:, 1] = adjacency[:, 0].transpose(1, 2)
    return x, adjacency, mask, (lengths - 1).clamp_min(0)


class FullGNNCostModel(nn.Module):
    """No recurrent modules; the surrounding MLP and program graph stay fixed."""
    def __init__(self, dropout=.05):
        super().__init__()
        self.expression_encoder = LocalEncoder(14, 64, 100, dropout)
        self.transformation_encoder = LocalEncoder(21, 76, 200, dropout)
        comp_sizes = [1146, 600, 350, 200, 180]
        regression_sizes = [180, 200, 180]
        self.comp_embedding_layers = nn.ModuleList([
            nn.Linear(a, b) for a, b in zip(comp_sizes, comp_sizes[1:])])
        self.comp_embedding_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(4)])
        self.regression_layers = nn.ModuleList([
            nn.Linear(a, b) for a, b in zip(regression_sizes, regression_sizes[1:])])
        self.regression_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(2)])
        self.predict = nn.Linear(180, 1)
        self.loop_projection = nn.Linear(8, 180)
        self.node_types = nn.Embedding(2, 180)
        self.graph_layers = nn.ModuleList([RelationalLayer() for _ in range(4)])
        self.ELU = nn.ELU()
        self.LeakyReLU = nn.LeakyReLU(.01)
        for layer in [*self.comp_embedding_layers, *self.regression_layers,
                      self.predict, self.loop_projection]:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, batch):
        _, first, tags, third, loops, _ = batch.base.inputs
        b, c = first.shape[:2]
        expr = self.expression_encoder(batch.expressions).reshape(b, c, 100)
        affine = self.transformation_encoder(transformation_graph(tags)).reshape(b, c, 200)
        computations = torch.cat((first, affine, third, expr), dim=-1)
        for layer, dropout in zip(self.comp_embedding_layers, self.comp_embedding_dropouts):
            computations = dropout(self.ELU(layer(computations)))
        nodes = torch.cat((self.loop_projection(loops) + self.node_types.weight[0],
                           computations + self.node_types.weight[1]), dim=1)
        for layer in self.graph_layers:
            nodes = layer(nodes, batch.base.adjacency)
        x = nodes.mean(1)
        for layer, dropout in zip(self.regression_layers, self.regression_dropouts):
            x = dropout(self.ELU(layer(x)))
        return self.LeakyReLU(self.predict(x)[:, 0])


def create_full_gnn():
    # Copy only randomly initialized common modules, never trained checkpoints.
    from .paper_models import create_model
    reference = create_model('rgcn')
    model = FullGNNCostModel()
    shared = {k: v for k, v in reference.state_dict().items() if k in model.state_dict()}
    model.load_state_dict(shared, strict=False)
    return model
