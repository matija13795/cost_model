"""Convenience entry points for the four unchanged experimental architectures."""
import json
import torch

from .paper_models import create_model, set_device
from .full_gnn import create_full_gnn
from .hetero_gat import create_hetero_gat
from .efficient_model import create_efficient_model


MODEL_NAMES = ('lstm', 'hybrid_rgcn', 'full_rgcn', 'heterogat')


def build_model(name, device='cpu', *, efficient=False):
    """Build a model without loading weights. Set the Torch seed before calling."""
    if name not in MODEL_NAMES:
        raise ValueError(f'Choose one of {MODEL_NAMES}, received {name!r}')
    if efficient and name != 'heterogat':
        raise ValueError('The efficient execution option applies to heterogat only')
    if name == 'lstm':
        return set_device(create_model('lstm'), device)
    if name == 'hybrid_rgcn':
        model = create_model('rgcn')
    elif name == 'full_rgcn':
        model = create_full_gnn()
    else:
        model = create_efficient_model() if efficient else create_hetero_gat()
    return model.to(device)


if __name__ == '__main__':
    torch.set_num_threads(1)
    recurrent = (torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM,
                 torch.nn.RNNCell, torch.nn.GRUCell, torch.nn.LSTMCell)
    for name in MODEL_NAMES:
        torch.manual_seed(42)
        model = build_model(name)
        print(json.dumps(dict(model=name, parameters=sum(p.numel() for p in model.parameters()),
                              recurrent_modules=sum(isinstance(m, recurrent) for m in model.modules()))))
