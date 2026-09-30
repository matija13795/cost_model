import os
import io
import logging
import random
import gc
import hydra
from hydra.core.config_store import ConfigStore
from utils.data_utils import *
from utils.modeling import *
from utils.train_utils import *
def load_batches_from_path(train_paths, train_devices, val_paths):
    assert(len(train_paths)>0 and len(val_paths)>0)
    assert(len(train_paths) == len(train_devices))
    train_bl = []
    val_bl = []
    # Read batches from specified paths
    for index, path in enumerate(train_paths):
        if os.path.exists(path):
            print(f"Loading second part of the training set {path} into {train_devices[index]}")
            with open(path, "rb") as file:
                train_bl_2 = torch.load(path, map_location=train_devices[index])
                train_bl += train_bl_2
    for path in val_paths:
        if os.path.exists(path):
            print(f"Loading second part of the validation set {path} into the CPU")
            with open(path, "rb") as file:
                val_bl_2 = torch.load(path, map_location="cpu")
                val_bl += val_bl_2
    
    # Shuffle both training and validation sets
    random.shuffle(train_bl)
    random.shuffle(val_bl)
    
    return train_bl, val_bl
@hydra.main(config_path="conf", config_name="config")
def main(conf):
    # Defining logger
    log_filename = [part for part in conf.training.log_file.split('/') if len(part) > 3][-1]
    log_folder_path = os.path.join(conf.experiment.base_path, "logs/")
    if not os.path.exists(log_folder_path):
        os.makedirs(log_folder_path)
    log_file = os.path.join(log_folder_path, log_filename)
    logging.basicConfig(filename = log_file,
                        filemode='a',
                        level = logging.DEBUG,
                        format = '%(asctime)s:%(levelname)s:  %(message)s')
    logging.info(f"Starting experiment {conf.experiment.name}")
    
    # We train on the device set by the user in the conf file
    train_device = torch.device(conf.training.training_gpu)
    
    # If a GPU is being used for validation, we use it otherwise we use the cpu
    validation_device = torch.device(conf.training.validation_gpu)
    # Defining the model
    logging.info("Defining the model")
    if conf.model.name == "recursive_lstm":
        model = Model_Recursive_LSTM_v2(
            input_size=conf.model.input_size,
            comp_embed_layer_sizes=list(conf.model.comp_embed_layer_sizes),
            drops=list(conf.model.drops),
            loops_tensor_size=8,
            device=train_device,
        )
    
    elif conf.model.name == "looper_transformer":
        model = LOOPerTransformer(
            d_model=conf.model.d_model,
            nhead=conf.model.nhead,
            num_layers=conf.model.num_layers,
            max_seq_len=conf.model.max_seq_len
        )

    elif conf.model.name == "fcnn":
        model = Model_FF_v1(
            input_size=conf.model.input_size,
            max_comps=conf.model.max_comps,
            comp_embed_layer_sizes=list(conf.model.comp_embed_layer_sizes),
            drops=list(conf.model.drops),
            mlp_num_layers=conf.model.mlp_num_layers,
            mlp_width=conf.model.mlp_width,
            #mlp_layer_sizes=list(conf.model.mlp_layer_sizes),
            mlp_dropout=conf.model.mlp_dropout,
            activation=conf.model.activation,
            norm_type=conf.model.norm_type,
            device=train_device,
        )

    elif conf.model.name == "residual-fcnn":
        model = Model_FF_v2(
            input_size=conf.model.input_size,
            max_comps=conf.model.max_comps,
            comp_embed_layer_sizes=list(conf.model.comp_embed_layer_sizes),
            drops=list(conf.model.drops),
            mlp_num_layers=conf.model.mlp_num_layers,
            mlp_width=conf.model.mlp_width,
            #mlp_layer_sizes=list(conf.model.mlp_layer_sizes),
            mlp_dropout=conf.model.mlp_dropout,
            activation=conf.model.activation,
            norm_type=conf.model.norm_type,
            device=train_device,
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
            cnn_num_blocks = conf.model.cnn_num_blocks,
            cnn_width = conf.model.cnn_width,
            cnn_use_bn = conf.model.cnn_use_bn,
            cnn_dropout = conf.model.cnn_dropout,
            cnn_activation = conf.model.cnn_activation,
            pooling = conf.model.pooling,
            device=train_device,
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
            cnn_num_blocks = conf.model.cnn_num_blocks,
            cnn_width = conf.model.cnn_width,
            cnn_use_bn = conf.model.cnn_use_bn,
            cnn_dropout = conf.model.cnn_dropout,
            cnn_activation = conf.model.cnn_activation,
            pooling = conf.model.pooling,
            mlp_use_residual = conf.model.mlp_use_residual,
            device=train_device,
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
            device=train_device,
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
            device=train_device,
        )

        # model = Model_RNN_v2(
        #     input_size=conf.model.input_size,
        #     comp_embed_layer_sizes=list(conf.model.comp_embed_layer_sizes),
        #     drops=list(conf.model.drops),
        #     rnn_hidden_size=conf.model.rnn_hidden_size,
        #     rnn_num_layers=conf.model.rnn_num_layers,
        #     rnn_bidirectional=conf.model.rnn_bidirectional,
        #     rnn_dropout=conf.model.rnn_dropout,
        #     device=train_device,
        # )
    
    # Load model weights and continue training if specified  
    if conf.training.continue_training:
        print(f"Continue training using model from {conf.training.model_weights_path}")
        model.load_state_dict(torch.load(conf.training.model_weights_path, map_location=train_device))
    
    # Enable gradient tracking for training
    for param in model.parameters():
        param.requires_grad = True
        
    # Reading data
    logging.info("Reading the dataset")
    train_bl_1 = []
    train_bl_2 = []
    val_bl_1 = []
    val_bl_2 = []
    
    # Training
    train_file_path = os.path.join( conf.experiment.base_path, "batched/train/", f"{Path(conf.data_generation.train_dataset_file).parts[-1][:-4]}_CPU.pt")
    if os.path.exists(train_file_path):
        print(f"Loading second part of the training set {train_file_path} into the CPU")
        with open(train_file_path, "rb") as file:
            train_bl_2 = torch.load(train_file_path, map_location="cpu")
            
    train_file_path = os.path.join(conf.experiment.base_path, "batched/train/", f"{Path(conf.data_generation.train_dataset_file).parts[-1][:-4]}_GPU.pt")
    
    print(f"Loading first part of the training set {train_file_path} into device number : {conf.training.training_gpu}")
    with open(train_file_path, "rb") as file:
        if conf.training.load_on_cpu == True:
            train_bl_1 = torch.load(train_file_path, map_location="cpu")
        else:
            train_bl_1 = torch.load(train_file_path, map_location=train_device)
    
    # Fuse loaded training batches
    train_bl = train_bl_1 + train_bl_2 if len(train_bl_2) > 0 else train_bl_1
    # Validation
    validation_file_path = os.path.join( conf.experiment.base_path, "batched/valid/", f"{Path(conf.data_generation.valid_dataset_file).parts[-1][:-4]}_CPU.pt")
    if os.path.exists(validation_file_path):
        print(f"Loading second part of the validation set {validation_file_path} into the CPU")
        with open(validation_file_path, "rb") as file:
            val_bl_2 = torch.load(validation_file_path, map_location="cpu")
            
    validation_file_path = os.path.join(conf.experiment.base_path, "batched/valid/", f"{Path(conf.data_generation.valid_dataset_file).parts[-1][:-4]}_GPU.pt")
    
    print(f"Loading first part of the validation set {validation_file_path} into device: {conf.training.validation_gpu}")
    with open(validation_file_path, "rb") as file:
        if conf.training.load_on_cpu == True:
            val_bl_1 = torch.load(validation_file_path, map_location="cpu")
        else:
            val_bl_1 = torch.load(validation_file_path, map_location=validation_device)
    
    # Fuse loaded training batches
    val_bl = val_bl_1 + val_bl_2 if len(val_bl_2) > 0 else val_bl_1
        
    # Defining training params
    criterion = mape_criterion
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=conf.training.lr, weight_decay=conf.training.weight_decay
    )
    logger = logging.getLogger()
    if conf.wandb.use_wandb:
        # Intializing wandb
        wandb.init(name = conf.experiment.name, project=conf.wandb.project)
        wandb.config = dict(conf)
        wandb.watch(model)
    
    # Training
    print("Training the model")
    bl_dict = {"train": train_bl, "val": val_bl}
    
    train_model(
        config=conf,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        max_lr=conf.training.lr,
        dataloader=bl_dict,
        num_epochs=conf.training.max_epochs,
        logger=logger,
        log_every=1,
        train_device=train_device,
        validation_device=conf.training.validation_gpu,
        max_batch_size=conf.data_generation.batch_size,
    )


if __name__ == "__main__":
    main()
