"""Parameter-matched RGCN with the unchanged LOOPer local encoders and head."""
import torch
from torch import nn
from .model import LOOPerCostModel


class RelationalLayer(nn.Module):
    def __init__(self, width=180, dropout=0.05):
        super().__init__()
        self.relations = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(6)])
        self.self_linear = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        for linear in [*self.relations, self.self_linear]:
            nn.init.xavier_uniform_(linear.weight)

    def forward(self, nodes, adjacency):
        messages = self.self_linear(nodes)
        for r, linear in enumerate(self.relations):
            messages = messages + linear(torch.bmm(adjacency[:, r], nodes))
        return self.norm(nodes + self.dropout(torch.nn.functional.elu(messages)))


class RGCNCostModel(LOOPerCostModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for name in ("concat_layers", "concat_dropouts", "no_comps_tensor", "no_nodes_tensor",
                     "comps_lstm", "nodes_lstm", "roots_lstm"):
            delattr(self, name)
        self.loop_projection = nn.Linear(8, 180)
        self.node_types = nn.Embedding(2, 180)
        self.graph_layers = nn.ModuleList([RelationalLayer() for _ in range(4)])
        nn.init.xavier_uniform_(self.loop_projection.weight)

    def forward(self, batch):
        computations = self.encode_computations(batch.inputs) + self.node_types.weight[1]
        loops = self.loop_projection(batch.inputs[4]) + self.node_types.weight[0]
        nodes = torch.cat((loops, computations), dim=1)
        return self.readout(self.message_pass(nodes, batch.adjacency))

    def message_pass(self, nodes, adjacency):
        for layer in self.graph_layers:
            nodes = layer(nodes, adjacency)
        return nodes

    def readout(self, nodes):
        x = nodes.mean(1)
        for layer, dropout in zip(self.regression_layers, self.regression_dropouts):
            x = dropout(self.ELU(layer(x)))
        return self.LeakyReLU(self.predict(x)[:, 0])


def create_model(cfg):
    kwargs = dict(input_size=cfg.model.input_size,
                  comp_embed_layer_sizes=cfg.model.comp_embed_layer_sizes,
                  drops=cfg.model.drops)
    if cfg.model.architecture == "lstm":
        return LOOPerCostModel(**kwargs)
    if cfg.model.architecture == "rgcn":
        if list(cfg.model.comp_embed_layer_sizes) != [600, 350, 200, 180]:
            raise ValueError("The frozen RGCN experiment requires the matched 180-wide encoder")
        return RGCNCostModel(**kwargs)
    raise ValueError(f"Unknown architecture: {cfg.model.architecture}")


def forward_batch(model, batch):
    return model(batch) if isinstance(model, RGCNCostModel) else model(batch.inputs)
