"""Archived LSTM initialization and the matched RGCN comparator."""
from .pact25_archive.modeling import Model_Recursive_LSTM_v2
from .gnn import RGCNCostModel, forward_batch


def create_model(architecture):
    original = Model_Recursive_LSTM_v2(846, drops=[0.05] * 4)
    if architecture == 'lstm':
        return original
    if architecture != 'rgcn':
        raise ValueError(architecture)
    graph = RGCNCostModel(drops=[0.05] * 4)
    shared = {name: value for name, value in original.state_dict().items()
              if name in graph.state_dict()}
    graph.load_state_dict(shared, strict=False)
    return graph


def set_device(model, device):
    model.to(device)
    if isinstance(model, Model_Recursive_LSTM_v2):
        model.device = str(device)
    return model
