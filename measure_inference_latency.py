import os
import time
import torch
import hydra
import argparse
from tqdm import tqdm
from pathlib import Path
from utils.modeling import *
from hydra.core.config_store import ConfigStore

def define_and_load_model(conf, device):
    if conf.model.name == "recursive_lstm":
        model = Model_Recursive_LSTM_v2(
            input_size=conf.model.input_size,
            comp_embed_layer_sizes=list(conf.model.comp_embed_layer_sizes),
            drops=list(conf.model.drops),
            loops_tensor_size=8,
            device=device,
        )
    elif conf.model.name == "fcnn":
        model = Model_FF_v1(
            input_size=conf.model.input_size,
            max_comps=conf.model.max_comps,
            comp_embed_layer_sizes=list(conf.model.comp_embed_layer_sizes),
            drops=list(conf.model.drops),
            mlp_num_layers=conf.model.mlp_num_layers,
            mlp_width=conf.model.mlp_width,
            mlp_dropout=conf.model.mlp_dropout,
            activation=conf.model.activation,
            norm_type=conf.model.norm_type,
            device=device,
        )

    elif conf.model.name == "residual-fcnn":
        model = Model_FF_v2(
            input_size=conf.model.input_size,
            max_comps=conf.model.max_comps,
            comp_embed_layer_sizes=list(conf.model.comp_embed_layer_sizes),
            drops=list(conf.model.drops),
            mlp_num_layers=conf.model.mlp_num_layers,
            mlp_width=conf.model.mlp_width,
            mlp_dropout=conf.model.mlp_dropout,
            activation=conf.model.activation,
            norm_type=conf.model.norm_type,
            device=device,
        )

    elif conf.model.name == "cnn":
        model = Model_CNN_v1(
            input_size=conf.model.input_size,
            max_comps=conf.model.max_comps,
            comp_embed_layer_sizes=list(conf.model.comp_embed_layer_sizes),
            drops=list(conf.model.drops),
            mlp_num_layers=conf.model.mlp_num_layers,
            mlp_width=conf.model.mlp_width,
            mlp_dropout=conf.model.mlp_dropout,
            activation=conf.model.activation,
            norm_type=conf.model.norm_type,
            cnn_channels=conf.model.get("cnn_channels", None),
            cnn_kernel_sizes=conf.model.get("cnn_kernel_sizes", None),
            cnn_num_blocks=conf.model.cnn_num_blocks,
            cnn_width=conf.model.cnn_width,
            cnn_use_bn=conf.model.cnn_use_bn,
            cnn_dropout=conf.model.cnn_dropout,
            cnn_activation=conf.model.cnn_activation,
            pooling=conf.model.pooling,
            device=device,
        )

    elif conf.model.name == "residual-cnn":
        model = Model_CNN_v2(
            input_size=conf.model.input_size,
            max_comps=conf.model.max_comps,
            comp_embed_layer_sizes=list(conf.model.comp_embed_layer_sizes),
            drops=list(conf.model.drops),
            mlp_num_layers=conf.model.mlp_num_layers,
            mlp_width=conf.model.mlp_width,
            mlp_dropout=conf.model.mlp_dropout,
            activation=conf.model.activation,
            norm_type=conf.model.norm_type,
            cnn_channels=conf.model.get("cnn_channels", None),
            cnn_kernel_sizes=conf.model.get("cnn_kernel_sizes", None),
            cnn_num_blocks=conf.model.cnn_num_blocks,
            cnn_width=conf.model.cnn_width,
            cnn_use_bn=conf.model.cnn_use_bn,
            cnn_dropout=conf.model.cnn_dropout,
            cnn_activation=conf.model.cnn_activation,
            pooling=conf.model.pooling,
            mlp_use_residual=conf.model.mlp_use_residual,
            device=device,
        )

    elif conf.model.name == "simple-rnn":
        model = Model_RNN_v1(
            input_size=conf.model.input_size,
            comp_embed_layer_sizes=list(conf.model.comp_embed_layer_sizes),
            drops=list(conf.model.drops),
            activation=conf.model.activation,
            rnn_type=conf.model.rnn_type,
            rnn_hidden_size=conf.model.rnn_hidden_size,
            rnn_num_layers=conf.model.rnn_num_layers,
            rnn_bidirectional=conf.model.rnn_bidirectional,
            rnn_dropout=conf.model.rnn_dropout,
            rnn_pooling=conf.model.rnn_pooling,
            attn_type=conf.model.attn_type,
            attn_dropout=conf.model.attn_dropout,
            device=device,
        )
    
    elif conf.model.name == "looper_attn":
        model = Model_Recursive_LSTM_v3(
            input_size=conf.model.input_size,
            comp_embed_layer_sizes=list(conf.model.comp_embed_layer_sizes),
            drops=list(conf.model.drops),
            loops_tensor_size=8,
            use_attn_comps=True,
            use_attn_nodes=True,
            use_attn_roots=True,
            use_cross_attn_children=True,
            use_cross_attn_comps=True,
            use_attn_tags=True,
            use_attn_exprs=True,
            device=device,
        )

    # Load the trained model weights
    state_dict = torch.load(
        conf.testing.testing_model_weights_path,
        map_location=device,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


@hydra.main(config_path="conf", config_name="config")
def main(conf):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = define_and_load_model(conf, device)

    dataset_file_path = os.path.join(conf.experiment.base_path, "batched/valid/", f"{Path(conf.data_generation.valid_dataset_file).parts[-1][:-4]}_GPU.pt")
    # Load dataset directly on the same device as the model
    dataset = torch.load(dataset_file_path, map_location=device)

    total_inference_time = 0.0
    num_batches = len(dataset)
    with torch.inference_mode():
        # optional warm-up for GPU
        if device.type == "cuda":
            for _ in range(5):
                inputs, _ = dataset[0]
                _ = model(inputs)
            torch.cuda.synchronize()

        for batch in tqdm(dataset, desc="measuring inference latency"):
            inputs, labels = batch
            if device.type == "cuda":
                torch.cuda.synchronize()
                
            t0 = time.perf_counter()
            _ = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            total_inference_time += (t1 - t0)

    avg_inference_time = total_inference_time / num_batches
    print(
        f"Average inference time per batch: {avg_inference_time:.6f}s "
        f"({avg_inference_time*1e3:.3f} ms)"
    )

if __name__ == "__main__":
    main()