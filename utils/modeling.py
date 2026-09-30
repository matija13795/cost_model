from unicodedata import bidirectional
import torch
import math
from typing import Optional, Tuple
from torch import nn
from .data_utils import MAX_NUM_TRANSFORMATIONS, MAX_TAGS

def initialization_function_sparse(x):
    return nn.init.sparse_(x, sparsity=0.1)

def initialization_function_xavier(x):
    return nn.init.xavier_uniform_(x)

class DotAttentionPool(nn.Module):
    """
    Single-head Luong dot attention.
    - pooling mode: learned global query (per block), V = K = sequence outputs
    - cross-attn mode: external query (condition vector), V = K = sequence outputs

    Inputs:
      H: [B, T, D] sequence of states (Keys/Values)
      query: None -> pooling with learned q; else [B, 1, D] tensor used as Query
      scale: scaling factor; if None, use 1/sqrt(D)
      dropout: applied on attention weights

    Returns:
      context: [B, 1, D]
      weights: [B, 1, T]
    """
    def __init__(self, d_model: int, dropout: float = 0.1, use_scale: bool = True):
        super().__init__()
        self.d_model = d_model
        self.use_scale = use_scale
        self.scale = (d_model ** 0.5) if use_scale else 1.0
        self.dropout = nn.Dropout(dropout)
        # learned query for pooling mode
        self.q_param = nn.Parameter(torch.randn(1, 1, d_model))
        initialization_function_xavier(self.q_param)

    def forward(self, H: torch.Tensor, query: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # H: [B, T, D]
        B, T, D = H.shape
        if T == 0:
            # Caller should handle "no sequence" case and never call us with T==0.
            raise RuntimeError("DotAttentionPool received empty sequence (T==0).")

        if query is None:
            Q = self.q_param.expand(B, -1, -1)  # [B, 1, D]
        else:
            # expect [B, 1, D]
            assert query.dim() == 3 and query.size(0) == B and query.size(1) == 1 and query.size(2) == D, \
                f"query must be [B,1,{D}] but got {tuple(query.shape)}"
            Q = query

        # Luong "dot": score = Q @ K^T; here K=H
        # scores: [B, 1, T]
        scores = torch.bmm(Q, H.transpose(1, 2))
        if self.use_scale:
            scores = scores / self.scale

        attn = torch.softmax(scores, dim=-1)  # [B, 1, T]
        attn = self.dropout(attn)

        # context = attn @ V (V=H): [B, 1, D]
        context = torch.bmm(attn, H)
        return context, attn

# Define the architecture of the cost model
class Model_Recursive_LSTM_v2(nn.Module):
    def __init__(
        self,
        input_size,
        comp_embed_layer_sizes=[600, 350, 200, 180],
        drops=[0.225, 0.225, 0.225, 0.225],
        output_size=1,
        lstm_embedding_size=100,
        expr_embed_size=100,
        loops_tensor_size=8,
        device="cpu",
        num_layers=1,
        bidirectional=True,
    ):
        super().__init__()
        self.device = device
        embedding_size = comp_embed_layer_sizes[-1]
        
        regression_layer_sizes = [embedding_size] + comp_embed_layer_sizes[-2:]
        concat_layer_sizes = [
            embedding_size * 2 + loops_tensor_size
        ] + comp_embed_layer_sizes[-2:]
        
        comp_embed_layer_sizes = [
            input_size + lstm_embedding_size * (2 if bidirectional else 1) * num_layers + expr_embed_size
        ] + comp_embed_layer_sizes
        
        self.comp_embedding_layers = nn.ModuleList()
        self.comp_embedding_dropouts = nn.ModuleList()
        self.regression_layers = nn.ModuleList()
        self.regression_dropouts = nn.ModuleList()
        self.concat_layers = nn.ModuleList()
        self.concat_dropouts = nn.ModuleList()
        
        # Create the transformation encoding layers
        self.encode_vectors = nn.Linear(
            MAX_TAGS,
            MAX_TAGS,
            bias=True,
        )
        # Create the computation embedding layers
        for i in range(len(comp_embed_layer_sizes) - 1):
            self.comp_embedding_layers.append(
                nn.Linear(
                    comp_embed_layer_sizes[i], comp_embed_layer_sizes[i + 1], bias=True
                )
            )
            initialization_function_xavier(self.comp_embedding_layers[i].weight)
            self.comp_embedding_dropouts.append(nn.Dropout(drops[i]))
        # Create the final regression layers
        for i in range(len(regression_layer_sizes) - 1):
            self.regression_layers.append(
                nn.Linear(
                    regression_layer_sizes[i], regression_layer_sizes[i + 1], bias=True
                )
            )
            initialization_function_xavier(self.regression_layers[i].weight)
            self.regression_dropouts.append(nn.Dropout(drops[i]))
            
        # Create the feed forward netwrok responsible for embedding loop levels
        for i in range(len(concat_layer_sizes) - 1):
            self.concat_layers.append(
                nn.Linear(concat_layer_sizes[i], concat_layer_sizes[i + 1], bias=True)
            )
            initialization_function_xavier(self.concat_layers[i].weight)
            nn.init.zeros_(self.concat_layers[i].weight)
            self.concat_dropouts.append(nn.Dropout(drops[i]))
        # Output layer
        self.predict = nn.Linear(regression_layer_sizes[-1], output_size, bias=True)
        
        
        initialization_function_xavier(self.predict.weight)
        self.ELU = nn.ELU()
        self.LeakyReLU = nn.LeakyReLU(0.01)
        # Initialize a tensor to represent the absence of computations at a level in the program tree
        self.no_comps_tensor = nn.Parameter(
            initialization_function_xavier(torch.zeros(1, embedding_size))
        )
        # Initialize a tensor to represent the absence of child loops at a level in the program tree
        self.no_nodes_tensor = nn.Parameter(
            initialization_function_xavier(torch.zeros(1, embedding_size))
        )
        # LSTM to encode computations
        self.comps_lstm = nn.LSTM(
            comp_embed_layer_sizes[-1], embedding_size, batch_first=True
        )
        # LSTM to encode child loop levels
        self.nodes_lstm = nn.LSTM(
            comp_embed_layer_sizes[-1], embedding_size, batch_first=True
        )
        # LSTM to encode program roots
        self.roots_lstm = nn.LSTM(
            comp_embed_layer_sizes[-1], embedding_size, batch_first=True
        )
        # LSTM to encode computations
        self.transformation_vectors_embed = nn.LSTM(
            MAX_TAGS,
            lstm_embedding_size,
            batch_first=True,
            bidirectional=bidirectional,
            num_layers=num_layers,
        )
        # LSTM to encode computation expressions
        self.exprs_embed = nn.LSTM(
            11,
            expr_embed_size,
            batch_first=True,
        )
    # Recursive function to embed a root of the program
    def get_hidden_state(self, node, comps_embeddings, loops_tensor):
        nodes_list = []
        for n in node["child_list"]:
            # Recusrive call to embed all the children of the loop first if they exist
            nodes_list.append(self.get_hidden_state(
                n, comps_embeddings, loops_tensor))
        
        if nodes_list != []:
            # Pass the embedding of all the child loops through the nodes LSTM
            nodes_tensor = torch.cat(nodes_list, 1)
            lstm_out, (nodes_h_n, nodes_c_n) = self.nodes_lstm(nodes_tensor)
            nodes_h_n = nodes_h_n.permute(1, 0, 2)
        
        else: # If there are no child loops contained within this level
            # The nodes embedding is a random vector (no_nodes_tensor) that represents that there are no nodes underneath this level
            nodes_h_n = torch.unsqueeze(self.no_nodes_tensor, 0).expand(
                comps_embeddings.shape[0], -1, -1
            )
        if node["has_comps"]:
            # If there are computations contained in this loop, pass them through the computations LSTM
            selected_comps_tensor = torch.index_select(
                comps_embeddings, 
                1, 
                node["computations_indices"].to(self.device)
            )
            lstm_out, (comps_h_n, comps_c_n) = self.comps_lstm(
                selected_comps_tensor
            )
            comps_h_n = comps_h_n.permute(1, 0, 2)
        else: # If there are no child computations contained within this level
            # The computations embedding is a random vector (no_comps_tensor) that represents that there are no computations underneath this level
            comps_h_n = torch.unsqueeze(self.no_comps_tensor, 0).expand(
                comps_embeddings.shape[0], 
                -1, 
                -1
            )
        # Get the loop vector for this level
        selected_loop_tensor = torch.index_select(
            loops_tensor, 
            1, 
            node["loop_index"].to(self.device)
        )
        # Concatinate the loop vector, computations embedding and nodes (child loops) embedding
        x = torch.cat((nodes_h_n, comps_h_n, selected_loop_tensor), 2)
        # Pass the concatinated vector through a feed forward neural network
        for i in range(len(self.concat_layers)):
            x = self.concat_layers[i](x)
            x = self.concat_dropouts[i](self.ELU(x))
        return x

    def forward(self, tree_tensors):
        # Separate the input tensor
        tree, comps_tensor_first_part, comps_tensor_vectors, comps_tensor_third_part, loops_tensor, functions_comps_expr_tree = tree_tensors
        
        # Embed all the expressions in the tree 
        batch_size, num_comps, len_sequence, len_vector = functions_comps_expr_tree.shape
        
        # Expressions embedding layer
        x = functions_comps_expr_tree.view(batch_size* num_comps, len_sequence, len_vector)
        _, (expr_embedding, _) = self.exprs_embed(x)
        
        expr_embedding = expr_embedding.permute(1, 0, 2).reshape(
            batch_size * num_comps, -1
        )
        
        # Embed all the computations in the tree
        batch_size, num_comps, __dict__ = comps_tensor_first_part.shape
        
        first_part = comps_tensor_first_part.to(self.device).view(batch_size * num_comps, -1)
        vectors = comps_tensor_vectors.to(self.device) # No need to reshape this tensor since we transformed it when loading the data
        third_part = comps_tensor_third_part.to(self.device).view(batch_size * num_comps, -1)
        
        # Pass the transformation vectors through the vector encoding LSTM
        vectors = self.encode_vectors(vectors)
        _, (prog_embedding, _) = self.transformation_vectors_embed(vectors)
        prog_embedding = prog_embedding.permute(1, 0, 2).reshape(
            batch_size * num_comps, -1
        )
        
        # Concatinate the leftover parts from the computatuion, the vectors embedding, and the expression embedding
        x = torch.cat(
            (
                first_part,
                prog_embedding,
                third_part,
                expr_embedding,
            ),
            dim=1,
        ).view(batch_size, num_comps, -1)
        
        # Pass the concatinated vector through a feed forward neural network to extract the final computation embedding vector
        for i in range(len(self.comp_embedding_layers)):
            x = self.comp_embedding_layers[i](x)
            x = self.comp_embedding_dropouts[i](self.ELU(x))
        comps_embeddings = x
        
        # For each root in the program tree structure 
        roots_list = []
        for root in tree["roots"]:
            # We call the recusrive embedding function to extract the root's embedding
            roots_list.append(
                self.get_hidden_state(
                    root, 
                    comps_embeddings, 
                    loops_tensor
                )
            )
        
        # We concatinate the roots and pass them through an LSTM
        roots_tensor = torch.cat(roots_list, 1)
        lstm_out, (roots_h_n, roots_c_n) = self.roots_lstm(roots_tensor)
        roots_h_n = roots_h_n.permute(1, 0, 2)
        
        # Finally, we pass the output of the roots LSTM to the regression layers
        x = roots_h_n
        for i in range(len(self.regression_layers)):
            x = self.regression_layers[i](x)
            x = self.regression_dropouts[i](self.ELU(x))
        out = self.predict(x)
        # We know the speedups to be predicted need to be greater or  equal to zero
        # We use the LeakyRelu to assure this while avoiding the dying ReLu problem
        return self.LeakyReLU(out[:, 0, 0])


class MicroEncoders(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Encoders for raw features
        self.loop_enc = nn.Linear(8, d_model)        # sample[4]
        self.domain_enc = nn.Linear(33, d_model)     # sample[1]
        self.transf_enc = nn.Linear(16, d_model)     # sample[2]
        self.access_enc = nn.Linear(813, d_model)    # sample[3]
        self.op_enc = nn.Linear(11, d_model)         # sample[5]
        
        # Structural embeddings
        self.depth_emb = nn.Embedding(20, d_model)    # Nesting depth
        self.segment_emb = nn.Embedding(100, d_model) # Comp ID
        
        # Special Tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.loop_end_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.elu = nn.ELU()

    def forward(self, loop_feat, dom_feat, trans_feat, acc_feat, op_feat):
        # Batch-project all features into d_model
        return (self.elu(self.loop_enc(loop_feat)), 
                self.elu(self.domain_enc(dom_feat)),
                self.elu(self.transf_enc(trans_feat)), 
                self.elu(self.access_enc(acc_feat)),
                self.elu(self.op_enc(op_feat)))

class LOOPerTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=6, max_seq_len=1344):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.encoders = MicroEncoders(d_model)
        
        # Positional Encoding
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Regression Head
        self.regression = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ELU(),
            nn.Linear(d_model // 2, 1),
            nn.LeakyReLU(0.01) # Preserves non-negative speedup logic [cite: 422]
        )

    def assemble_sequence(self, l_e, d_e, t_e, a_e, o_e, types, indices, depths, segments, real_len):
        """
        Assembles a dynamically-sized sequence based on the actual program length.
        """
        batch_size = l_e.shape[0]
        device = l_e.device
        
        # 1. Initialize full_seq to the REAL length, not the MAX length
        full_seq = torch.zeros((batch_size, real_len, self.d_model), device=device)
        
        # 2. Set [CLS] token at pos 0
        full_seq[:, 0, :] = self.encoders.cls_token

        # 3. Flatten computation features
        num_comps = d_e.shape[1]
        t_e_flat = t_e.view(batch_size, num_comps * 4, -1)
        o_e_flat = o_e.view(batch_size, num_comps * 66, -1)

        # 4. Gather logic (Vectorized)
        for t_type in range(6):
            # types is already sliced to real_len in the forward pass
            type_mask = (types[1:] == t_type) 
            if not type_mask.any(): continue
            
            # Get sequence positions (offset by 1 because of [CLS])
            seq_indices = (type_mask.nonzero(as_tuple=True)[0] + 1)
            
            for s_idx in seq_indices:
                d_idx = indices[s_idx].item()
                
                if t_type == 0: # Loop
                    feat = l_e[:, d_idx, :]
                elif t_type == 1: # Domain
                    feat = d_e[:, d_idx, :]
                elif t_type == 2: # Transformation
                    feat = t_e_flat[:, d_idx, :]
                elif t_type == 3: # Access
                    feat = a_e[:, d_idx, :]
                elif t_type == 4: # Operations
                    c_val, op_sub = d_idx // 100, d_idx % 100
                    feat = o_e_flat[:, c_val * 66 + op_sub, :]
                elif t_type == 5: # Loop End
                    feat = self.encoders.loop_end_token.expand(batch_size, -1, -1).squeeze(1)

                full_seq[:, s_idx, :] = feat

        # 5. Add Structural Embeddings (sliced to real_len)
        depth_meta = self.encoders.depth_emb(depths).unsqueeze(0)
        seg_meta = self.encoders.segment_emb(segments).unsqueeze(0)
        
        return full_seq + depth_meta + seg_meta

    def forward(self, flattened_tensors):
        dom, trans, acc, loops, ops, types, indices, depths, segments, mask = flattened_tensors
        batch_size = dom.shape[0]
        device = dom.device

        # 1. Handle Metadata (1D vs 2D)
        if types.dim() == 2:
            types_bp = types[0]
            indices_bp = indices[0]
            depths_bp = depths[0]
            segments_bp = segments[0]
            mask_blueprint = mask[0] 
        else:
            types_bp = types
            indices_bp = indices
            depths_bp = depths
            segments_bp = segments
            mask_blueprint = mask

        # 2. Dynamic Slicing
        real_len = torch.where(~mask_blueprint)[0][-1].item() + 1
        
        # 3. Assemble Sequence
        l_e, d_e, t_e, a_e, o_e = self.encoders(loops, dom, trans, acc, ops)
        
        # Slice the blueprint inside the call
        full_seq = self.assemble_sequence(
            l_e, d_e, t_e, a_e, o_e, 
            types_bp[:real_len], 
            indices_bp[:real_len], 
            depths_bp[:real_len], 
            segments_bp[:real_len], 
            real_len
        )
        
        # 4. Positional Encoding
        full_seq = full_seq + self.pos_emb[:, :real_len, :]

        # 5. CRITICAL FIX: Contiguous Mask
        # .clone().contiguous() forces PyTorch to actually copy the data into a 
        # clean, packed memory block that the optimized kernels can read.
        final_mask = mask_blueprint[:real_len].unsqueeze(0).expand(batch_size, -1).clone().contiguous().to(device).bool()

        # 6. Safety Check: If it still fails, we'll want to see these shapes
        # print(f"Seq: {full_seq.shape}, Mask: {final_mask.shape}")

        # Ensure full_seq is also contiguous just in case
        full_seq = full_seq.contiguous()

        # 7. Transformer
        trans_out = self.transformer(full_seq, src_key_padding_mask=final_mask)
        
        return self.regression(trans_out[:, 0, :]).view(-1)



# -----------------------------------------------------------------------
# Feed‑forward alternative to LOOPer’s recursive model
# -----------------------------------------------------------------------
# This model keeps the original computation‑vector embedding pipeline
# (expressions → LSTM; transformation vectors → LSTM; three‑part static
# feature vector → FC layers) but drops all subsequent LSTM/recursive
# aggregation over the loop tree.  Instead, it flattens the resulting
# computation embeddings (post‑order traversal, up to max_comps)
# and feeds the concatenated vector through a 5‑layer MLP.
# -----------------------------------------------------------------------

class Model_FF_v1(nn.Module):
    """Feed-forward variant of the LOOPer cost model (no tree recursion).

    Parameters mirror those of the original model where they are still
    relevant.  Anything related to loop/node LSTMs is omitted because
    they are not used here.
    """
    def __init__(
        self,
        input_size,
        max_comps,
        comp_embed_layer_sizes=[600, 350, 200, 180],
        drops=[0.225, 0.225, 0.225, 0.225],
        *,  # <--- means "all following arguments must be specified as keyword arguments (i.e., name=value), not as positional arguments."
        mlp_num_layers=5,
        mlp_width=512,
        mlp_dropout=0.0,
        activation="elu",
        norm_type="none",  # "none" | "batchnorm" | "layernorm"
        output_size=1,
        lstm_embedding_size=100,
        expr_embed_size=100,
        device="cpu",
        num_layers=1,
        bidirectional=True,
    ):
        super().__init__()
        self.device = device
        self.max_comps = max_comps
        comp_emb_dim = comp_embed_layer_sizes[-1]

        comp_embed_layer_sizes = [
            input_size + lstm_embedding_size * (2 if bidirectional else 1) * num_layers + expr_embed_size
        ] + comp_embed_layer_sizes

        self.comp_embedding_layers = nn.ModuleList()
        self.comp_embedding_dropouts = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        self.mlp_norms  = nn.ModuleList()
        self.act_name = activation
        self.norm_type = norm_type

        # ------------------------------------------------------------------
        # Modules reused from the original pipeline — these produce the
        # per‑computation embedding vectors.
        # ------------------------------------------------------------------
        # Create the transformation encoding layers
        self.encode_vectors = nn.Linear(
            MAX_TAGS,
            MAX_TAGS,
            bias=True
        )
        # Create the computation embedding layers
        for i in range(len(comp_embed_layer_sizes) - 1):
            self.comp_embedding_layers.append(
                nn.Linear(
                    comp_embed_layer_sizes[i], comp_embed_layer_sizes[i + 1], bias=True
                )
            )
            initialization_function_xavier(self.comp_embedding_layers[i].weight)
            self.comp_embedding_dropouts.append(nn.Dropout(drops[i]))

        # ------------------------------------------------------------------
        # New feed‑forward regression head.
        # Input dim = max_comps * comp_embed_layer_sizes[-1]
        # ------------------------------------------------------------------
        #if isinstance(mlp_layer_sizes, str): # allow CLI/YAML to pass the list as a string, e.g. "[1024,512,256]"
        #    mlp_layer_sizes = [int(x) for x in mlp_layer_sizes.strip(" []").split(",") if x]

        flattened_dim = self.max_comps * comp_emb_dim
        #default_layers = [512, 256, 128, 64]           # fallback if nothing passed
        #hidden = mlp_layer_sizes or default_layers     # user-defined or default
        hidden = [mlp_width] * mlp_num_layers
        mlp_layer_sizes = [flattened_dim] + hidden + [output_size]

        for i in range(len(mlp_layer_sizes) - 1):
            self.mlp_layers.append(
                nn.Linear(
                    mlp_layer_sizes[i], mlp_layer_sizes[i+1], bias=True
                )
            )
            initialization_function_xavier(self.mlp_layers[i].weight)
            if i < len(mlp_layer_sizes) - 2:  # no norm on final linear
                if self.norm_type == "batchnorm":
                    self.mlp_norms.append(nn.BatchNorm1d(mlp_layer_sizes[i + 1]))
                elif self.norm_type == "layernorm":
                    self.mlp_norms.append(nn.LayerNorm(mlp_layer_sizes[i + 1]))
                else:
                    self.mlp_norms.append(nn.Identity())

        # one shared dropout module
        self.mlp_dropout = nn.Dropout(mlp_dropout)
        self.activation  = self._make_activation(self.act_name)
        self.ELU = nn.ELU() # static since it's used to create computation embeddings
        self.LeakyReLU = nn.LeakyReLU(0.01) # statis since it's used at the output layer

        # LSTM to encode computations
        self.transformation_vectors_embed = nn.LSTM(
            MAX_TAGS,
            lstm_embedding_size,
            batch_first=True,
            bidirectional=bidirectional,
            num_layers=num_layers,
        )
        # LSTM to encode computation expressions
        self.exprs_embed = nn.LSTM(
            11,  # length of operation‑one‑hot per token
            expr_embed_size,
            batch_first=True,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    # post‑order traversal to get computation indices
    # def post_order_indices(self, node):
    #     indices = []
    #     for child in node.get("child_list", []):
    #         indices.extend(self.post_order_indices(child))
    #     indices.extend(node.get("computations_indices", []))
    #     return indices

    # helper to build activation
    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        if name == "leaky_relu":
            return nn.LeakyReLU(0.01)
        if name == "gelu":
            return nn.GELU()
        if name in ("silu", "swish"):
            return nn.SiLU()
        if name == "mish":
            return nn.Mish()
        if name == "elu":
            return nn.ELU()
        raise ValueError(f"Unknown activation '{name}'")

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, tree_tensors):
        (
            indices,
            comps_tensor_first_part,
            comps_tensor_vectors,
            comps_tensor_third_part,
            functions_comps_expr_tree,
        ) = tree_tensors

        # -------------------------------------------------------------
        # 1. Expression embeddings
        # -------------------------------------------------------------
        batch_size, num_comps, len_sequence, len_vector = functions_comps_expr_tree.shape
        
        # Expressions embedding layer
        x = functions_comps_expr_tree.view(batch_size * num_comps, len_sequence, len_vector)
        _, (expr_embedding, _) = self.exprs_embed(x)

        expr_embedding = expr_embedding.permute(1, 0, 2).reshape(
            batch_size * num_comps, -1
        )

        # -------------------------------------------------------------
        # 2. Transformation‑vector embeddings
        # -------------------------------------------------------------
        vectors = self.encode_vectors(comps_tensor_vectors.to(self.device))
        _, (prog_embedding, _) = self.transformation_vectors_embed(vectors)
        prog_embedding = prog_embedding.permute(1, 0, 2).reshape(
            batch_size * num_comps, -1
        )

        # -------------------------------------------------------------
        # 3. Static first/third parts
        # -------------------------------------------------------------
        first_part = comps_tensor_first_part.to(self.device).view(batch_size * num_comps, -1)
        third_part = comps_tensor_third_part.to(self.device).view(batch_size * num_comps, -1)

        # -------------------------------------------------------------
        # 4. Build per‑computation embedding
        # -------------------------------------------------------------
        # Concatinate the leftover parts from the computatuion, the vectors embedding, and the expression embedding
        x = torch.cat(
            (
                first_part,
                prog_embedding,
                third_part,
                expr_embedding,
            ),
            dim=1,
        ).view(batch_size, num_comps, -1)

        # Pass the concatinated vector through a feed forward neural network to extract the final computation embedding vector
        for i in range(len(self.comp_embedding_layers)):
            x = self.comp_embedding_layers[i](x)
            x = self.comp_embedding_dropouts[i](self.ELU(x))
        comps_embeddings = x
        emb_dim = comps_embeddings.size(-1)

        # -------------------------------------------------------------
        # 5. Flatten computations in post‑order (pad to max_comps)
        # -------------------------------------------------------------
        # indices = []
        # for root in tree["roots"]: # get a post-order list of computation indices
        #     indices.extend(self.post_order_indices(root))

        # make sure number of computations < max_comps
        if len(indices) > self.max_comps:
            raise RuntimeError(
                f"Program has {len(indices)} computations, but MAX_COMPS is {self.max_comps}"
            )
        real_len = len(indices)

        # gather embeddings for those indices (shape B × real_len × emb_dim)
        flat = torch.index_select(comps_embeddings, 1, comps_embeddings.new_tensor(indices, dtype=torch.long))

        # if we need padding, append zeros so we reach (B × max_comps × emb_dim)
        if real_len < self.max_comps:
            pad_size = (batch_size, self.max_comps - real_len, emb_dim)
            flat = torch.cat([flat, flat.new_zeros(pad_size)], dim=1)

        # reshape to (B, max_comps * emb_dim)
        mlp_in = flat.reshape(batch_size, -1)

        # -------------------------------------------------------------
        # 6. MLP regression
        # -------------------------------------------------------------
        h = mlp_in
        for lin, norm in zip(self.mlp_layers[:-1], self.mlp_norms):
            h = self.mlp_dropout(self.activation(norm(lin(h))))
        out = self.LeakyReLU(self.mlp_layers[-1](h))
        return out.squeeze(-1)


class Model_FF_v2(nn.Module):
    """Feed-forward variant of the LOOPer cost model (no tree recursion)
       with residual connections.

    Parameters mirror those of the original model where they are still
    relevant.  Anything related to loop/node LSTMs is omitted because
    they are not used here.
    """
    def __init__(
        self,
        input_size,
        max_comps,
        comp_embed_layer_sizes=[600, 350, 200, 180],
        drops=[0.225, 0.225, 0.225, 0.225],
        *,  # <--- means "all following arguments must be specified as keyword arguments (i.e., name=value), not as positional arguments."
        mlp_num_layers=5,
        mlp_width=512,
        mlp_dropout=0.0,
        activation="elu",
        norm_type="none",  # "none" | "batchnorm" | "layernorm"
        output_size=1,
        lstm_embedding_size=100,
        expr_embed_size=100,
        device="cpu",
        num_layers=1,
        bidirectional=True,
    ):
        super().__init__()
        self.device = device
        self.max_comps = max_comps
        comp_emb_dim = comp_embed_layer_sizes[-1]

        comp_embed_layer_sizes = [
            input_size + lstm_embedding_size * (2 if bidirectional else 1) * num_layers + expr_embed_size
        ] + comp_embed_layer_sizes

        self.comp_embedding_layers = nn.ModuleList()
        self.comp_embedding_dropouts = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        self.mlp_norms  = nn.ModuleList()
        self.act_name = activation
        self.norm_type = norm_type

        # ------------------------------------------------------------------
        # Modules reused from the original pipeline — these produce the
        # per‑computation embedding vectors.
        # ------------------------------------------------------------------
        # Create the transformation encoding layers
        self.encode_vectors = nn.Linear(
            MAX_TAGS,
            MAX_TAGS,
            bias=True
        )
        # Create the computation embedding layers
        for i in range(len(comp_embed_layer_sizes) - 1):
            self.comp_embedding_layers.append(
                nn.Linear(
                    comp_embed_layer_sizes[i], comp_embed_layer_sizes[i + 1], bias=True
                )
            )
            initialization_function_xavier(self.comp_embedding_layers[i].weight)
            self.comp_embedding_dropouts.append(nn.Dropout(drops[i]))

        # ------------------------------------------------------------------
        # New feed‑forward regression head.
        # Input dim = max_comps * comp_embed_layer_sizes[-1]
        # ------------------------------------------------------------------
        #if isinstance(mlp_layer_sizes, str): # allow CLI/YAML to pass the list as a string, e.g. "[1024,512,256]"
        #    mlp_layer_sizes = [int(x) for x in mlp_layer_sizes.strip(" []").split(",") if x]

        flattened_dim = self.max_comps * comp_emb_dim
        hidden = [mlp_width] * mlp_num_layers
        mlp_layer_sizes = [flattened_dim] + hidden + [output_size]

        for i in range(len(mlp_layer_sizes) - 1):
            self.mlp_layers.append(
                nn.Linear(
                    mlp_layer_sizes[i], mlp_layer_sizes[i+1], bias=True
                )
            )
            initialization_function_xavier(self.mlp_layers[i].weight)
            if i < len(mlp_layer_sizes) - 2:  # no norm on final linear
                if self.norm_type == "batchnorm":
                    self.mlp_norms.append(nn.BatchNorm1d(mlp_layer_sizes[i + 1]))
                elif self.norm_type == "layernorm":
                    self.mlp_norms.append(nn.LayerNorm(mlp_layer_sizes[i + 1]))
                else:
                    self.mlp_norms.append(nn.Identity())

        self.mlp_skip_proj = nn.ModuleList()
        for i in range(len(mlp_layer_sizes) - 2):  # exclude final layer
            in_dim  = mlp_layer_sizes[i]
            out_dim = mlp_layer_sizes[i + 1]

            if in_dim == out_dim:
                # dimensions match → identity skip
                self.mlp_skip_proj.append(nn.Identity())
            else:
                # dimensions differ → learn a linear projection
                proj = nn.Linear(in_dim, out_dim, bias=False)
                initialization_function_xavier(proj.weight)
                self.mlp_skip_proj.append(proj)

        # one shared dropout module
        self.mlp_dropout = nn.Dropout(mlp_dropout)
        self.activation  = self._make_activation(self.act_name)
        self.ELU = nn.ELU() # static since it's used to create computation embeddings
        self.LeakyReLU = nn.LeakyReLU(0.01) # statis since it's used at the output layer

        # LSTM to encode computations
        self.transformation_vectors_embed = nn.LSTM(
            MAX_TAGS,
            lstm_embedding_size,
            batch_first=True,
            bidirectional=bidirectional,
            num_layers=num_layers,
        )
        # LSTM to encode computation expressions
        self.exprs_embed = nn.LSTM(
            11,  # length of operation‑one‑hot per token
            expr_embed_size,
            batch_first=True,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    # helper to build activation
    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        if name == "leaky_relu":
            return nn.LeakyReLU(0.01)
        if name == "gelu":
            return nn.GELU()
        if name in ("silu", "swish"):
            return nn.SiLU()
        if name == "mish":
            return nn.Mish()
        if name == "elu":
            return nn.ELU()
        raise ValueError(f"Unknown activation '{name}'")

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, tree_tensors):
        (
            indices,
            comps_tensor_first_part,
            comps_tensor_vectors,
            comps_tensor_third_part,
            functions_comps_expr_tree,
        ) = tree_tensors

        # -------------------------------------------------------------
        # 1. Expression embeddings
        # -------------------------------------------------------------
        batch_size, num_comps, len_sequence, len_vector = functions_comps_expr_tree.shape
        
        # Expressions embedding layer
        x = functions_comps_expr_tree.view(batch_size * num_comps, len_sequence, len_vector)
        _, (expr_embedding, _) = self.exprs_embed(x)

        expr_embedding = expr_embedding.permute(1, 0, 2).reshape(
            batch_size * num_comps, -1
        )

        # -------------------------------------------------------------
        # 2. Transformation‑vector embeddings
        # -------------------------------------------------------------
        vectors = self.encode_vectors(comps_tensor_vectors.to(self.device))
        _, (prog_embedding, _) = self.transformation_vectors_embed(vectors)
        prog_embedding = prog_embedding.permute(1, 0, 2).reshape(
            batch_size * num_comps, -1
        )

        # -------------------------------------------------------------
        # 3. Static first/third parts
        # -------------------------------------------------------------
        first_part = comps_tensor_first_part.to(self.device).view(batch_size * num_comps, -1)
        third_part = comps_tensor_third_part.to(self.device).view(batch_size * num_comps, -1)

        # -------------------------------------------------------------
        # 4. Build per‑computation embedding
        # -------------------------------------------------------------
        # Concatinate the leftover parts from the computatuion, the vectors embedding, and the expression embedding
        x = torch.cat(
            (
                first_part,
                prog_embedding,
                third_part,
                expr_embedding,
            ),
            dim=1,
        ).view(batch_size, num_comps, -1)

        # Pass the concatinated vector through a feed forward neural network to extract the final computation embedding vector
        for i in range(len(self.comp_embedding_layers)):
            x = self.comp_embedding_layers[i](x)
            x = self.comp_embedding_dropouts[i](self.ELU(x))
        comps_embeddings = x
        emb_dim = comps_embeddings.size(-1)

        # -------------------------------------------------------------
        # 5. Flatten computations in post‑order (pad to max_comps)
        # -------------------------------------------------------------

        # make sure number of computations < max_comps
        if len(indices) > self.max_comps:
            raise RuntimeError(
                f"Program has {len(indices)} computations, but MAX_COMPS is {self.max_comps}"
            )
        real_len = len(indices)

        # gather embeddings for those indices (shape B × real_len × emb_dim)
        flat = torch.index_select(comps_embeddings, 1, comps_embeddings.new_tensor(indices, dtype=torch.long))

        # if we need padding, append zeros so we reach (B × max_comps × emb_dim)
        if real_len < self.max_comps:
            pad_size = (batch_size, self.max_comps - real_len, emb_dim)
            flat = torch.cat([flat, flat.new_zeros(pad_size)], dim=1)

        # reshape to (B, max_comps * emb_dim)
        mlp_in = flat.reshape(batch_size, -1)

        # -------------------------------------------------------------
        # 6. MLP regression with residual (skip) connections
        # -------------------------------------------------------------
        h = mlp_in
        for lin, norm, skip in zip(self.mlp_layers[:-1],
                                   self.mlp_norms,
                                   self.mlp_skip_proj):
            residual = skip(h)                           # project if needed
            h = self.mlp_dropout(self.activation(norm(lin(h))))
            h = h + residual                            # ← add skip

        out = self.LeakyReLU(self.mlp_layers[-1](h))     # final layer (no skip)
        return out.squeeze(-1)


# -----------------------------------------------------------------------
# CNN alternative to LOOPer’s recursive model
# -----------------------------------------------------------------------
# This model keeps the original computation‑vector embedding pipeline
# (expressions → LSTM; transformation vectors → LSTM; three‑part static
# feature vector → FC layers) but drops all subsequent LSTM/recursive
# aggregation over the loop tree. Instead, it ...
# -----------------------------------------------------------------------

class Model_CNN_v1(nn.Module):
    """
    """
    def __init__(
        self,
        input_size: int,
        max_comps: int,
        comp_embed_layer_sizes=[600, 350, 200, 180],
        drops=[0.225, 0.225, 0.225, 0.225],
        *,
        # MLP head params (post‑pool), aligned with FF
        mlp_num_layers: int = 4,
        mlp_width: int = 128,
        mlp_dropout: float = 0.0,
        activation: str = "gelu",
        norm_type: str = "none",       # "none" | "batchnorm" | "layernorm"
        # CNN stack params
        cnn_channels=None,              # e.g. [256, 256, 256] or None to use cnn_width repeated
        cnn_num_blocks: int = 3,
        cnn_width: int = 256,
        cnn_kernel_sizes=None,          # e.g. [3,5,3]; if None -> all 3
        cnn_use_bn: bool = True,
        cnn_dropout: float = 0.0,       # Dropout1d between conv blocks
        cnn_activation: str = "gelu",
        pooling: str = "max",          # "max" or "avg"
        # shared / misc
        output_size: int = 1,
        lstm_embedding_size: int = 100,
        expr_embed_size: int = 100,
        device: str = "cpu",
        num_layers: int = 1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.device = device
        self.max_comps = max_comps
        comp_emb_dim = comp_embed_layer_sizes[-1]

        comp_embed_layer_sizes = [
            input_size + lstm_embedding_size * (2 if bidirectional else 1) * num_layers + expr_embed_size
        ] + comp_embed_layer_sizes

        self.comp_embedding_layers = nn.ModuleList()
        self.comp_embedding_dropouts = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        self.mlp_norms = nn.ModuleList()
        self.act_name = activation
        self.norm_type = norm_type
        self.pooling = pooling.lower()

        self.encode_vectors = nn.Linear(
            MAX_TAGS,
            MAX_TAGS,
            bias=True
        )

        # Create the computation embedding layers
        for i in range(len(comp_embed_layer_sizes) - 1):
            self.comp_embedding_layers.append(
                nn.Linear(
                    comp_embed_layer_sizes[i], comp_embed_layer_sizes[i + 1], bias=True
                )
            )
            initialization_function_xavier(self.comp_embedding_layers[i].weight)
            self.comp_embedding_dropouts.append(nn.Dropout(drops[i]))

        # -----------------------------
        # CNN stack over computations
        # -----------------------------
        if cnn_channels is None:
            cnn_channels = [cnn_width] * cnn_num_blocks
        if cnn_kernel_sizes is None:
            cnn_kernel_sizes = [3] * len(cnn_channels)
        assert len(cnn_kernel_sizes) == len(cnn_channels), "cnn_kernel_sizes must match cnn_channels length"

        self.conv_blocks = self._build_conv_stack(
            in_ch=comp_emb_dim,
            channels=cnn_channels,
            kernels=cnn_kernel_sizes,
            use_bn=cnn_use_bn,
            dropout=cnn_dropout,
            act_name=cnn_activation,
        )
        conv_out_ch = cnn_channels[-1] if len(cnn_channels) > 0 else comp_emb_dim

        # -----------------------------
        # MLP regression head (post pool)
        # -----------------------------
        hidden = [mlp_width] * mlp_num_layers
        mlp_layer_sizes = [conv_out_ch] + hidden + [output_size]
        for i in range(len(mlp_layer_sizes) - 1):
            self.mlp_layers.append(
                nn.Linear(
                    mlp_layer_sizes[i], mlp_layer_sizes[i+1], bias=True
                )
            )
            initialization_function_xavier(self.mlp_layers[i].weight)
            if i < len(mlp_layer_sizes) - 2:  # no norm on final linear
                if self.norm_type == "batchnorm":
                    self.mlp_norms.append(nn.BatchNorm1d(mlp_layer_sizes[i + 1]))
                elif self.norm_type == "layernorm":
                    self.mlp_norms.append(nn.LayerNorm(mlp_layer_sizes[i + 1]))
                else:
                    self.mlp_norms.append(nn.Identity())

        self.mlp_dropout = nn.Dropout(mlp_dropout)
        self.activation = self._make_activation(self.act_name)
        self.conv_activation = self._make_activation(cnn_activation)
        self.ELU = nn.ELU()            # used in comp embedding
        self.LeakyReLU = nn.LeakyReLU(0.01)  # output nonlinearity

        # -----------------
        # Sequence encoders
        # -----------------
        self.transformation_vectors_embed = nn.LSTM(
            MAX_TAGS,
            lstm_embedding_size,
            batch_first=True,
            bidirectional=bidirectional,
            num_layers=num_layers,
        )
        self.exprs_embed = nn.LSTM(
            11,
            expr_embed_size,
            batch_first=True,
        )

    # ---------- utils ----------
    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "relu": return nn.ReLU()
        if name == "leaky_relu": return nn.LeakyReLU(0.01)
        if name == "gelu": return nn.GELU()
        if name in ("silu", "swish"): return nn.SiLU()
        if name == "mish": return nn.Mish()
        if name == "elu": return nn.ELU()
        raise ValueError(f"Unknown activation '{name}'")

    def _build_conv_stack(self, in_ch, channels, kernels, use_bn, dropout, act_name):
        layers = []
        act = self._make_activation(act_name)
        ch_prev = in_ch
        for ch, k in zip(channels, kernels):
            padding = k // 2  # keep length
            conv = nn.Conv1d(ch_prev, ch, kernel_size=k, padding=padding, bias=not use_bn)
            nn.init.kaiming_uniform_(conv.weight, nonlinearity="relu")
            layers.append(conv)
            if use_bn:
                layers.append(nn.BatchNorm1d(ch))
            layers.append(act)
            if dropout and dropout > 0:
                layers.append(nn.Dropout1d(dropout))
            ch_prev = ch
        return nn.Sequential(*layers)

    # ---------- forward ----------
    def forward(self, tree_tensors):
        (
            indices,
            comps_tensor_first_part,
            comps_tensor_vectors,
            comps_tensor_third_part,
            functions_comps_expr_tree,
        ) = tree_tensors

        batch_size, num_comps, len_sequence, len_vector = functions_comps_expr_tree.shape

        # 1) Expression embeddings
        x = functions_comps_expr_tree.view(batch_size * num_comps, len_sequence, len_vector)
        _, (expr_embedding, _) = self.exprs_embed(x)
        expr_embedding = expr_embedding.permute(1, 0, 2).reshape(batch_size * num_comps, -1)

        # 2) Transformation‑vector embeddings
        vectors = self.encode_vectors(comps_tensor_vectors.to(self.device))
        _, (prog_embedding, _) = self.transformation_vectors_embed(vectors)
        prog_embedding = prog_embedding.permute(1, 0, 2).reshape(batch_size * num_comps, -1)

        # 3) Static parts
        first_part = comps_tensor_first_part.to(self.device).view(batch_size * num_comps, -1)
        third_part = comps_tensor_third_part.to(self.device).view(batch_size * num_comps, -1)

        # 4) Per‑computation embedding
        x = torch.cat((first_part, prog_embedding, third_part, expr_embedding), dim=1)
        x = x.view(batch_size, num_comps, -1)
        for i in range(len(self.comp_embedding_layers)):
            x = self.comp_embedding_layers[i](x)
            x = self.comp_embedding_dropouts[i](self.ELU(x))
        comps_embeddings = x  # (B, num_comps, comp_emb_dim)

        # 5) Gather in post‑order (pad to max_comps)
        if len(indices) > self.max_comps:
            raise RuntimeError(f"Program has {len(indices)} computations, but MAX_COMPS is {self.max_comps}")
        real_len = len(indices)
        flat = torch.index_select(comps_embeddings, 1, comps_embeddings.new_tensor(indices, dtype=torch.long))
        if real_len < self.max_comps:
            pad_size = (batch_size, self.max_comps - real_len, comps_embeddings.size(-1))
            flat = torch.cat([flat, flat.new_zeros(pad_size)], dim=1)

        # 6) CNN over sequence of comps -> pool
        h = flat.permute(0, 2, 1)    # (B, emb_dim, max_comps)
        h = self.conv_blocks(h)
        if self.pooling == "avg":
            h = torch.mean(h, dim=2)
        else:  # default max
            h = torch.max(h, dim=2).values

        # 7) MLP regression head
        z = h
        for lin, norm in zip(self.mlp_layers[:-1], self.mlp_norms):
            z = self.mlp_dropout(self.activation(norm(lin(z))))
        out = self.LeakyReLU(self.mlp_layers[-1](z))
        return out.squeeze(-1)


class Model_CNN_v2(nn.Module):
    """
    Same pipeline up to per-computation embeddings.
    After that: 1D residual conv blocks over the (post-order, padded) computation
    sequence, then pool -> MLP head. Optionally, the MLP head can also be residual.

    Differences vs Model_CNN_v1:
      - conv stack replaced by residual blocks (with 1x1 projection when channels change)
      - optional residual MLP (mlp_use_residual), off by default
    """
    def __init__(
        self,
        input_size: int,
        max_comps: int,
        comp_embed_layer_sizes=[600, 350, 200, 180],
        drops=[0.225, 0.225, 0.225, 0.225],
        *,
        # MLP head params (post-pool)
        mlp_num_layers: int = 4,
        mlp_width: int = 128,
        mlp_dropout: float = 0.0,
        activation: str = "gelu",
        norm_type: str = "none",  # "none" | "batchnorm" | "layernorm"
        mlp_use_residual: bool = False, # <- optional residual MLP
        # CNN residual stack params
        cnn_channels=None,              # e.g. [256, 256, 256] or None -> cnn_width repeated
        cnn_num_blocks: int = 3,
        cnn_width: int = 256,
        cnn_kernel_sizes=None,          # e.g. [3,5,3]; if None -> all 3
        cnn_use_bn: bool = True,
        cnn_dropout: float = 0.0,       # Dropout1d *inside* residual blocks
        cnn_activation: str = "gelu",
        pooling: str = "max",           # "max" or "avg"
        # shared / misc
        output_size: int = 1,
        lstm_embedding_size: int = 100,
        expr_embed_size: int = 100,
        device: str = "cpu",
        num_layers: int = 1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.device = device
        self.max_comps = max_comps
        self.pooling = pooling.lower()
        self.act_name = activation
        self.norm_type = norm_type
        self.mlp_use_residual = mlp_use_residual

        comp_emb_dim = comp_embed_layer_sizes[-1]
        comp_embed_layer_sizes = [
            input_size + lstm_embedding_size * (2 if bidirectional else 1) * num_layers + expr_embed_size
        ] + comp_embed_layer_sizes

        # ---------- computation embedding FF (unchanged) ----------
        self.comp_embedding_layers = nn.ModuleList()
        self.comp_embedding_dropouts = nn.ModuleList()
        for i in range(len(comp_embed_layer_sizes) - 1):
            lin = nn.Linear(comp_embed_layer_sizes[i], comp_embed_layer_sizes[i + 1], bias=True)
            initialization_function_xavier(lin.weight)
            self.comp_embedding_layers.append(lin)
            self.comp_embedding_dropouts.append(nn.Dropout(drops[i]))

        self.encode_vectors = nn.Linear(MAX_TAGS, MAX_TAGS, bias=True)

        # ---------- residual CNN stack ----------
        if cnn_channels is None:
            cnn_channels = [cnn_width] * cnn_num_blocks
        if cnn_kernel_sizes is None:
            cnn_kernel_sizes = [3] * len(cnn_channels)
        assert len(cnn_kernel_sizes) == len(cnn_channels), "cnn_kernel_sizes must match cnn_channels length"

        self.res_blocks = self._build_res_stack(
            in_ch=comp_emb_dim,
            channels=cnn_channels,
            kernels=cnn_kernel_sizes,
            use_bn=cnn_use_bn,
            dropout=cnn_dropout,
            act_name=cnn_activation,
        )
        conv_out_ch = cnn_channels[-1] if len(cnn_channels) > 0 else comp_emb_dim

        # ---------- MLP head ----------
        hidden = [mlp_width] * mlp_num_layers
        mlp_layer_sizes = [conv_out_ch] + hidden + [output_size]

        self.mlp_layers = nn.ModuleList()
        self.mlp_norms  = nn.ModuleList()
        for i in range(len(mlp_layer_sizes) - 1):
            lin = nn.Linear(mlp_layer_sizes[i], mlp_layer_sizes[i + 1], bias=True)
            initialization_function_xavier(lin.weight)
            self.mlp_layers.append(lin)
            if i < len(mlp_layer_sizes) - 2:
                if self.norm_type == "batchnorm":
                    self.mlp_norms.append(nn.BatchNorm1d(mlp_layer_sizes[i + 1]))
                elif self.norm_type == "layernorm":
                    self.mlp_norms.append(nn.LayerNorm(mlp_layer_sizes[i + 1]))
                else:
                    self.mlp_norms.append(nn.Identity())

        # Optional residual skips for MLP (match FF_v2 behavior)
        if self.mlp_use_residual:
            self.mlp_skip_proj = nn.ModuleList()
            for i in range(len(mlp_layer_sizes) - 2):  # exclude final layer
                in_dim  = mlp_layer_sizes[i]
                out_dim = mlp_layer_sizes[i + 1]
                if in_dim == out_dim:
                    self.mlp_skip_proj.append(nn.Identity())
                else:
                    proj = nn.Linear(in_dim, out_dim, bias=False)
                    initialization_function_xavier(proj.weight)
                    self.mlp_skip_proj.append(proj)
        else:
            self.mlp_skip_proj = None

        # ---------- activations / misc ----------
        self.mlp_dropout = nn.Dropout(mlp_dropout)
        self.activation = self._make_activation(self.act_name)
        self.conv_activation = self._make_activation(cnn_activation)
        self.ELU = nn.ELU()                 # comp embedding
        self.LeakyReLU = nn.LeakyReLU(0.01) # output nonlinearity

        # ---------- sequence encoders (unchanged) ----------
        self.transformation_vectors_embed = nn.LSTM(
            MAX_TAGS, lstm_embedding_size, batch_first=True,
            bidirectional=bidirectional, num_layers=num_layers
        )
        self.exprs_embed = nn.LSTM(11, expr_embed_size, batch_first=True)

    # ---------- utils ----------
    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "relu": return nn.ReLU()
        if name == "leaky_relu": return nn.LeakyReLU(0.01)
        if name == "gelu": return nn.GELU()
        if name in ("silu", "swish"): return nn.SiLU()
        if name == "mish": return nn.Mish()
        if name == "elu": return nn.ELU()
        raise ValueError(f"Unknown activation '{name}'")

    class _ResBlock1D(nn.Module):
        def __init__(self, in_ch, out_ch, k, use_bn, dropout, act: nn.Module):
            super().__init__()
            pad = k // 2
            bias1 = not use_bn
            bias2 = not use_bn

            self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, bias=bias1)
            nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity="relu")
            self.bn1 = nn.BatchNorm1d(out_ch) if use_bn else nn.Identity()
            self.act = act
            self.drop = nn.Dropout1d(dropout) if dropout and dropout > 0 else nn.Identity()

            self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, padding=pad, bias=bias2)
            nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity="relu")
            self.bn2 = nn.BatchNorm1d(out_ch) if use_bn else nn.Identity()

            # projection if channels differ
            if in_ch != out_ch:
                self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False if use_bn else True)
                if use_bn:
                    self.proj_bn = nn.BatchNorm1d(out_ch)
                else:
                    self.proj_bn = nn.Identity()
            else:
                self.proj = None
                self.proj_bn = None

        def forward(self, x):
            residual = x
            y = self.conv1(x)
            y = self.bn1(y)
            y = self.act(y)
            y = self.drop(y)
            y = self.conv2(y)
            y = self.bn2(y)

            if self.proj is not None:
                residual = self.proj(x)
                residual = self.proj_bn(residual)

            y = y + residual
            y = self.act(y)
            return y

    def _build_res_stack(self, in_ch, channels, kernels, use_bn, dropout, act_name):
        act = self._make_activation(act_name)
        layers = []
        ch_prev = in_ch
        for ch, k in zip(channels, kernels):
            layers.append(self._ResBlock1D(ch_prev, ch, k, use_bn, dropout, act))
            ch_prev = ch
        return nn.Sequential(*layers)

    # ---------- forward ----------
    def forward(self, tree_tensors):
        (
            indices,
            comps_tensor_first_part,
            comps_tensor_vectors,
            comps_tensor_third_part,
            functions_comps_expr_tree,
        ) = tree_tensors

        batch_size, num_comps, len_sequence, len_vector = functions_comps_expr_tree.shape

        # 1) Expression embeddings
        x = functions_comps_expr_tree.view(batch_size * num_comps, len_sequence, len_vector)
        _, (expr_embedding, _) = self.exprs_embed(x)
        expr_embedding = expr_embedding.permute(1, 0, 2).reshape(batch_size * num_comps, -1)

        # 2) Transformation-vector embeddings
        vectors = self.encode_vectors(comps_tensor_vectors.to(self.device))
        _, (prog_embedding, _) = self.transformation_vectors_embed(vectors)
        prog_embedding = prog_embedding.permute(1, 0, 2).reshape(batch_size * num_comps, -1)

        # 3) Static parts
        first_part = comps_tensor_first_part.to(self.device).view(batch_size * num_comps, -1)
        third_part = comps_tensor_third_part.to(self.device).view(batch_size * num_comps, -1)

        # 4) Per-computation embedding
        x = torch.cat((first_part, prog_embedding, third_part, expr_embedding), dim=1)
        x = x.view(batch_size, num_comps, -1)
        for i in range(len(self.comp_embedding_layers)):
            x = self.comp_embedding_layers[i](x)
            x = self.comp_embedding_dropouts[i](self.ELU(x))
        comps_embeddings = x  # (B, num_comps, comp_emb_dim)

        # 5) Post-order gather (pad to max_comps)
        if len(indices) > self.max_comps:
            raise RuntimeError(f"Program has {len(indices)} computations, but MAX_COMPS is {self.max_comps}")
        real_len = len(indices)
        flat = torch.index_select(comps_embeddings, 1, comps_embeddings.new_tensor(indices, dtype=torch.long))
        if real_len < self.max_comps:
            pad_size = (batch_size, self.max_comps - real_len, comps_embeddings.size(-1))
            flat = torch.cat([flat, flat.new_zeros(pad_size)], dim=1)

        # 6) Residual CNN over sequence of comps -> pool
        h = flat.permute(0, 2, 1)  # (B, emb_dim, max_comps)
        h = self.res_blocks(h)
        if self.pooling == "avg":
            h = torch.mean(h, dim=2)
        else:
            h = torch.max(h, dim=2).values

        # 7) MLP regression head (optionally residual)
        if self.mlp_use_residual:
            z = h
            for lin, norm, skip in zip(self.mlp_layers[:-1], self.mlp_norms, self.mlp_skip_proj):
                residual = skip(z)
                z = self.mlp_dropout(self.activation(norm(lin(z))))
                z = z + residual
            out = self.LeakyReLU(self.mlp_layers[-1](z))
        else:
            z = h
            for lin, norm in zip(self.mlp_layers[:-1], self.mlp_norms):
                z = self.mlp_dropout(self.activation(norm(lin(z))))
            out = self.LeakyReLU(self.mlp_layers[-1](z))

        return out.squeeze(-1)


class _DotAttention(nn.Module):
    """Luong-style dot-product attention over a sequence."""
    def __init__(self, scale: bool = True):
        super().__init__()
        self.scale = scale

    def forward(self, query, values, mask=None):
        """
        query:  (B, H)
        values: (B, T, H)
        mask:   (B, T) with True for valid positions, False for pad
        """
        B, T, H = values.shape
        q = query.unsqueeze(1)                    # (B, 1, H)
        scores = torch.bmm(values, q.transpose(1, 2)).squeeze(-1)  # (B, T)
        if self.scale:
            scores = scores / math.sqrt(H)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)   # (B, T)
        ctx = torch.bmm(weights.unsqueeze(1), values).squeeze(1)   # (B, H)
        return ctx, weights

class _AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention over a sequence."""
    def __init__(self, hidden_size):
        super().__init__()
        self.Wq = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wv = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v  = nn.Linear(hidden_size, 1, bias=False)
        initialization_function_xavier(self.Wq.weight)
        initialization_function_xavier(self.Wv.weight)
        initialization_function_xavier(self.v.weight)

    def forward(self, query, values, mask=None):
        """
        query:  (B, H)
        values: (B, T, H)
        mask:   (B, T) with True for valid positions, False for pad
        """
        # broadcast add: (B, T, H)
        s = torch.tanh(self.Wq(query).unsqueeze(1) + self.Wv(values))
        scores = self.v(s).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        ctx = torch.bmm(weights.unsqueeze(1), values).squeeze(1)  # (B, H)
        return ctx, weights


class Model_RNN_v1(nn.Module):
    """
    Same as the Model_FF_v1 up to the per-computation embeddings.
    The MLP head (flatten + MLP) is replaced with a configurable RNN head.
    """
    def __init__(
        self,
        input_size,
        comp_embed_layer_sizes=[600, 350, 200, 180],
        drops=[0.225, 0.225, 0.225, 0.225],
        *,
        activation="elu",
        norm_type="none",
        output_size=1,
        lstm_embedding_size=100,
        expr_embed_size=100,
        device="cpu",
        num_layers=1,
        bidirectional=True,

        # NEW: RNN head config
        rnn_type="lstm",  # "lstm" | "gru"
        rnn_hidden_size=256,
        rnn_num_layers=2,
        rnn_bidirectional=True,
        rnn_dropout=0.0,
        rnn_pooling="attn",  # "last" | "mean" | "attn"
        attn_type="dot",     # "dot" | "additive" | "none"
        attn_dropout=0.0,
    ):
        super().__init__()
        self.device = device
        self.act_name = activation

        # ------------------ embedding stack (unchanged) ------------------
        comp_emb_dim = comp_embed_layer_sizes[-1]
        comp_embed_layer_sizes = [
            input_size + lstm_embedding_size * (2 if bidirectional else 1) * num_layers + expr_embed_size
        ] + comp_embed_layer_sizes

        self.comp_embedding_layers = nn.ModuleList()
        self.comp_embedding_dropouts = nn.ModuleList()

        self.encode_vectors = nn.Linear(MAX_TAGS, MAX_TAGS, bias=True)

        for i in range(len(comp_embed_layer_sizes) - 1):
            lin = nn.Linear(comp_embed_layer_sizes[i], comp_embed_layer_sizes[i + 1], bias=True)
            initialization_function_xavier(lin.weight)
            self.comp_embedding_layers.append(lin)
            self.comp_embedding_dropouts.append(nn.Dropout(drops[i]))

        self.ELU = nn.ELU()  # used for embedding stack
        self.LeakyReLU = nn.LeakyReLU(0.01)
        # self.softplus = nn.Softplus()

        self.transformation_vectors_embed = nn.LSTM(
            MAX_TAGS,
            lstm_embedding_size,
            batch_first=True,
            bidirectional=bidirectional,
            num_layers=num_layers,
        )
        self.exprs_embed = nn.LSTM(11, expr_embed_size, batch_first=True)

        # ------------------ RNN head ------------------
        self.rnn_type = rnn_type.lower()
        self.rnn_bidirectional = rnn_bidirectional
        self.rnn_pooling = rnn_pooling.lower()
        self.attn_type = attn_type.lower()

        RNN = {"lstm": nn.LSTM, "gru": nn.GRU}.get(self.rnn_type)
        if RNN is None:
            raise ValueError(f"Unknown rnn_type '{rnn_type}'")

        self.rnn = RNN(
            input_size=comp_emb_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            dropout=(rnn_dropout if rnn_num_layers > 1 else 0.0),
            bidirectional=rnn_bidirectional,
            batch_first=True,
        )

        head_dim = rnn_hidden_size * (2 if rnn_bidirectional else 1)

        if self.rnn_pooling == "attn" and self.attn_type != "none":
            if self.attn_type == "dot":
                self.attn = _DotAttention(scale=True)
            elif self.attn_type == "additive":
                self.attn = _AdditiveAttention(head_dim)
            else:
                raise ValueError(f"Unknown attn_type '{attn_type}'")
            self.attn_drop = nn.Dropout(attn_dropout)
            rep_dim = head_dim
        elif self.rnn_pooling in ("last", "mean", "max") or self.attn_type == "none":
            self.attn = None
            self.attn_drop = nn.Identity()
            rep_dim = head_dim
        else:
            raise ValueError(f"Unknown rnn_pooling '{rnn_pooling}'")

        self.head = nn.Linear(rep_dim, output_size)
        initialization_function_xavier(self.head.weight)

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "relu": return nn.ReLU()
        if name == "leaky_relu": return nn.LeakyReLU(0.01)
        if name == "gelu": return nn.GELU()
        if name in ("silu", "swish"): return nn.SiLU()
        if name == "mish": return nn.Mish()
        if name == "elu": return nn.ELU()
        raise ValueError(f"Unknown activation '{name}'")

    def forward(self, tree_tensors):
        (
            indices,
            comps_tensor_first_part,
            comps_tensor_vectors,
            comps_tensor_third_part,
            functions_comps_expr_tree,
        ) = tree_tensors

        # ------------------ 1) expression embeddings ------------------
        B, num_comps, Lseq, Vdim = functions_comps_expr_tree.shape
        x = functions_comps_expr_tree.view(B * num_comps, Lseq, Vdim)
        _, (expr_embedding, _) = self.exprs_embed(x)   # (num_layers, B*num_comps, expr_embed)
        expr_embedding = expr_embedding.permute(1, 0, 2).reshape(B * num_comps, -1)

        # ------------------ 2) transformation vectors -----------------
        vectors = self.encode_vectors(comps_tensor_vectors.to(self.device))
        _, (prog_embedding, _) = self.transformation_vectors_embed(vectors)
        prog_embedding = prog_embedding.permute(1, 0, 2).reshape(B * num_comps, -1)

        # ------------------ 3) static parts ---------------------------
        first_part = comps_tensor_first_part.to(self.device).view(B * num_comps, -1)
        third_part = comps_tensor_third_part.to(self.device).view(B * num_comps, -1)

        # ------------------ 4) per-computation embedding --------------
        x = torch.cat((first_part, prog_embedding, third_part, expr_embedding), dim=1)
        x = x.view(B, num_comps, -1)
        for lin, drop in zip(self.comp_embedding_layers, self.comp_embedding_dropouts):
            x = drop(self.ELU(lin(x)))
        comps_embeddings = x  # (B, num_comps, comp_emb_dim)

        real_len = len(indices)
        idx = comps_embeddings.new_tensor(indices, dtype=torch.long)
        seq = torch.index_select(comps_embeddings, 1, idx)  # (B, real_len, D)

        # ------------------ 5) RNN head -------------------------------
        rnn_out, _ = self.rnn(seq)  # (B, real_len, H*dir)

        if self.rnn_pooling == "last":
            last_idx = real_len - 1
            rep = rnn_out[:, last_idx, :]  # (B, H*dir)

        elif self.rnn_pooling == "mean":
            rep = rnn_out.mean(dim=1)

        elif self.rnn_pooling == "max":
            rep = rnn_out.max(dim=1).values

        elif self.rnn_pooling == "attn" and self.attn is not None:
            query = rnn_out[:, real_len - 1, :]     # last valid step as query
            rep, _ = self.attn(query=query, values=rnn_out, mask=None)

        else:
            raise RuntimeError(f"Invalid pooling/attn combo: {self.rnn_pooling}, {self.attn_type}")

        out = self.LeakyReLU(self.head(rep))
        #out = self.softplus(self.head(rep))
        return out.squeeze(-1)

class Model_RNN_v2(nn.Module):
    def __init__(
        self,
        input_size,
        comp_embed_layer_sizes=[600, 350, 200, 180],
        drops=[0.225, 0.225, 0.225, 0.225],
        *,
        lstm_embedding_size=100,
        expr_embed_size=100,
        device,
        bidirectional=False,          # for the *embedding* LSTMs above (you already use this)
        num_layers=1,             # for the *embedding* LSTMs above (you already use this)
        # -------- LSTM head (new) --------
        rnn_hidden_size=384,
        rnn_num_layers=4,
        rnn_bidirectional=False,
        rnn_dropout=0.0,        # only applied between stacked LSTM layers
        output_size=1,          # scalar regression by default
    ):
        super().__init__()
        self.device = device

        # ---------- embedding stack ----------
        comp_emb_dim = comp_embed_layer_sizes[-1]
        comp_embed_layer_sizes = [
            input_size + lstm_embedding_size * (2 if bidirectional else 1) * num_layers + expr_embed_size
        ] + comp_embed_layer_sizes

        self.comp_embedding_layers = nn.ModuleList()
        self.comp_embedding_dropouts = nn.ModuleList()

        self.encode_vectors = nn.Linear(MAX_TAGS, MAX_TAGS, bias=True)

        for i in range(len(comp_embed_layer_sizes) - 1):
            lin = nn.Linear(comp_embed_layer_sizes[i], comp_embed_layer_sizes[i + 1], bias=True)
            initialization_function_xavier(lin.weight)
            self.comp_embedding_layers.append(lin)
            self.comp_embedding_dropouts.append(nn.Dropout(drops[i]))

        self.ELU = nn.ELU()  # used for embedding stack
        self.LeakyReLU = nn.LeakyReLU(0.01)

        self.transformation_vectors_embed = nn.LSTM(
            MAX_TAGS,
            lstm_embedding_size,
            batch_first=True,
            bidirectional=bidirectional,
            num_layers=num_layers,
        )
        self.exprs_embed = nn.LSTM(11, expr_embed_size, batch_first=True)

        # ---------- LSTM CODE (new head over comp embeddings) ----------
        self.seq_lstm = nn.LSTM(
            input_size=comp_emb_dim,  # each step is a comp embedding
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,         # (B, T, D)
            dropout=rnn_dropout if rnn_num_layers > 1 else 0.0,
            bidirectional=rnn_bidirectional,
        )

        seq_out_dim = rnn_hidden_size * (2 if rnn_bidirectional else 1)
        self.head_dropout = nn.Dropout(rnn_dropout)  # lightweight; ok if rnn_dropout=0.0
        self.regressor = nn.Linear(seq_out_dim, output_size)
        initialization_function_xavier(self.regressor.weight)

    def forward(self, tree_tensors):
        (
            indices,
            comps_tensor_first_part,
            comps_tensor_vectors,
            comps_tensor_third_part,
            functions_comps_expr_tree,
        ) = tree_tensors

        # ------------------ 1) expression embeddings ------------------
        B, num_comps, Lseq, Vdim = functions_comps_expr_tree.shape
        x = functions_comps_expr_tree.view(B * num_comps, Lseq, Vdim)
        _, (expr_embedding, _) = self.exprs_embed(x)   # (num_layers, B*num_comps, expr_embed)
        expr_embedding = expr_embedding.permute(1, 0, 2).reshape(B * num_comps, -1)

        # ------------------ 2) transformation vectors -----------------
        vectors = self.encode_vectors(comps_tensor_vectors.to(self.device))
        _, (prog_embedding, _) = self.transformation_vectors_embed(vectors)
        prog_embedding = prog_embedding.permute(1, 0, 2).reshape(B * num_comps, -1)

        # ------------------ 3) static parts ---------------------------
        first_part = comps_tensor_first_part.to(self.device).view(B * num_comps, -1)
        third_part = comps_tensor_third_part.to(self.device).view(B * num_comps, -1)

        # ------------------ 4) per-computation embedding --------------
        x = torch.cat((first_part, prog_embedding, third_part, expr_embedding), dim=1)
        x = x.view(B, num_comps, -1)
        for lin, drop in zip(self.comp_embedding_layers, self.comp_embedding_dropouts):
            x = drop(self.ELU(lin(x)))
        comps_embeddings = x  # (B, num_comps, comp_emb_dim)

        # ------------------ 5) order by indices (post-order already) --
        real_len = len(indices)                                    # same for all items in batch
        idx = comps_embeddings.new_tensor(indices, dtype=torch.long)
        seq = torch.index_select(comps_embeddings, 1, idx)         # (B, T=real_len, D)
        # (Optional) if your model/module moved to GPU, ensure data is there too
        seq = seq.to(self.device)

        # ------------------ 6) LSTM over the sequence -----------------
        # outputs: (B, T, H*dir), h_n: (L*dir, B, H)
        outputs, (h_n, c_n) = self.seq_lstm(seq)

        # Take the final layer's hidden state(s):
        # PyTorch stacks as [layer0_fwd, layer0_bwd, layer1_fwd, layer1_bwd, ...]
        num_directions = 2 if self.seq_lstm.bidirectional else 1
        last_layer_start = (self.seq_lstm.num_layers - 1) * num_directions

        if num_directions == 1:
            last = h_n[last_layer_start]                 # (B, H)
        else:
            h_fwd = h_n[last_layer_start + 0]            # (B, H)
            h_bwd = h_n[last_layer_start + 1]            # (B, H)
            last = torch.cat([h_fwd, h_bwd], dim=1)      # (B, 2H)

        # ------------------ 7) regression head -----------------------
        last = self.head_dropout(last)
        pred = self.regressor(last)                      # (B, output_size)

        return pred.squeeze(-1)

# Similar to the original LOOPer model, but with the addition of attention.
class Model_Recursive_LSTM_v3(nn.Module):
    def __init__(
        self,
        input_size: int,
        comp_embed_layer_sizes=[600, 350, 200, 180],
        drops=[0.225, 0.225, 0.225, 0.225],
        output_size: int = 1,
        lstm_embedding_size: int = 100,   # for transformation vectors (tags)
        expr_embed_size: int = 100,       # for expression tokens
        loops_tensor_size: int = 8,
        device: str = "cpu",
        num_layers: int = 1,              # for transformation_vectors_embed LSTM
        bidirectional: bool = True,       # for transformation_vectors_embed LSTM
        expr_token_size: int = 11,        # input size to exprs_embed LSTM

        # ---- Attention flags & hyperparams ----
        use_attn_comps: bool = False,         # Phase 1: attention-pooling over computations-in-loop
        use_attn_nodes: bool = False,         # Phase 1: attention-pooling over child loops
        use_attn_roots: bool = False,         # Phase 1: attention-pooling over roots

        use_cross_attn_children: bool = False,# Phase 2: loop-conditioned cross-attn over child loops
        use_cross_attn_comps: bool = False,   # Phase 2: loop-conditioned cross-attn over computations

        use_attn_tags: bool = False,          # Phase 3: attention-pooling over tag sequence
        use_attn_exprs: bool = False,         # Phase 3: attention-pooling over expr sequence

        attn_dropout: float = 0.1,
        attn_scale: Optional[bool] = True,    # True -> use 1/sqrt(d), False/None -> no scaling
    ):
        super().__init__()
        self.device = device
        self.output_size = output_size

        # ---- Effective dims for the two token LSTMs (tags & exprs) ----
        # If using attention on these encoders, the flattened dim drops the "* num_layers" factor.
        tags_dir = 2 if bidirectional else 1
        tags_out_dim = lstm_embedding_size * tags_dir
        eff_tags_dim = tags_out_dim if use_attn_tags else (tags_out_dim * num_layers)

        expr_out_dim = expr_embed_size  # exprs LSTM is unidirectional, num_layers fixed at 1
        eff_expr_dim = expr_out_dim if use_attn_exprs else expr_out_dim  # remains same (num_layers=1)

        # ---- The final comp embedding size is the last of comp_embed_layer_sizes ----
        embedding_size = comp_embed_layer_sizes[-1]

        # Build the concatenation size used in the recursive get_hidden_state()
        regression_layer_sizes = [embedding_size] + comp_embed_layer_sizes[-2:]
        concat_layer_sizes = [embedding_size * 2 + loops_tensor_size] + comp_embed_layer_sizes[-2:]

        # Compute the *input* feature size to the comp embedding MLP
        base_in = input_size + eff_tags_dim + eff_expr_dim
        comp_embed_layer_sizes = [base_in] + comp_embed_layer_sizes

        # ---- Save flags ----
        self.use_attn_comps = use_attn_comps
        self.use_attn_nodes = use_attn_nodes
        self.use_attn_roots = use_attn_roots
        self.use_cross_attn_children = use_cross_attn_children
        self.use_cross_attn_comps = use_cross_attn_comps
        self.use_attn_tags = use_attn_tags
        self.use_attn_exprs = use_attn_exprs
        self.attn_dropout = attn_dropout
        self.attn_scale = bool(attn_scale)

        # ---- Modules: encode transformation tags (projection + LSTM) ----
        self.encode_vectors = nn.Linear(MAX_TAGS, MAX_TAGS, bias=True)

        self.transformation_vectors_embed = nn.LSTM(
            input_size=MAX_TAGS,
            hidden_size=lstm_embedding_size,
            batch_first=True,
            bidirectional=bidirectional,
            num_layers=num_layers,
        )

        # ---- Modules: encode computation expressions ----
        self.exprs_embed = nn.LSTM(
            input_size=expr_token_size,
            hidden_size=expr_embed_size,
            batch_first=True,
            # keep single-layer, unidirectional as in your code
        )

        # Optional attention poolers for token LSTMs
        if self.use_attn_tags:
            self.tags_attn = DotAttentionPool(tags_out_dim, dropout=attn_dropout, use_scale=self.attn_scale)
        if self.use_attn_exprs:
            self.expr_attn = DotAttentionPool(expr_out_dim, dropout=attn_dropout, use_scale=self.attn_scale)

        # ---- Computation embedding MLP ----
        self.comp_embedding_layers = nn.ModuleList()
        self.comp_embedding_dropouts = nn.ModuleList()
        for i in range(len(comp_embed_layer_sizes) - 1):
            lin = nn.Linear(comp_embed_layer_sizes[i], comp_embed_layer_sizes[i + 1], bias=True)
            initialization_function_xavier(lin.weight)
            self.comp_embedding_layers.append(lin)
            self.comp_embedding_dropouts.append(nn.Dropout(drops[i]))

        # ---- LSTMs for recursive aggregation (fixed single-layer) ----
        self.comps_lstm = nn.LSTM(embedding_size, embedding_size, batch_first=True)
        self.nodes_lstm = nn.LSTM(embedding_size, embedding_size, batch_first=True)
        self.roots_lstm = nn.LSTM(embedding_size, embedding_size, batch_first=True)

        # Optional attention poolers for comps/nodes/roots
        if self.use_attn_comps or self.use_cross_attn_comps:
            self.comps_attn = DotAttentionPool(embedding_size, dropout=attn_dropout, use_scale=self.attn_scale)
        if self.use_attn_nodes or self.use_cross_attn_children:
            self.nodes_attn = DotAttentionPool(embedding_size, dropout=attn_dropout, use_scale=self.attn_scale)
        if self.use_attn_roots:
            self.roots_attn = DotAttentionPool(embedding_size, dropout=attn_dropout, use_scale=self.attn_scale)

        # Query projection for loop-conditioned cross-attn
        if self.use_cross_attn_children or self.use_cross_attn_comps:
            self.loop_q_proj = nn.Linear(loops_tensor_size, embedding_size, bias=False)
            initialization_function_xavier(self.loop_q_proj.weight)

        # ---- Recursive concat MLP (per-level fusion) ----
        self.concat_layers = nn.ModuleList()
        self.concat_dropouts = nn.ModuleList()
        for i in range(len(concat_layer_sizes) - 1):
            lin = nn.Linear(concat_layer_sizes[i], concat_layer_sizes[i + 1], bias=True)
            initialization_function_xavier(lin.weight)
            # the original code zeroed weights afterward; preserve behavior
            nn.init.zeros_(lin.weight)
            self.concat_layers.append(lin)
            self.concat_dropouts.append(nn.Dropout(drops[i]))

        # ---- Regression head ----
        self.regression_layers = nn.ModuleList()
        self.regression_dropouts = nn.ModuleList()
        for i in range(len(regression_layer_sizes) - 1):
            lin = nn.Linear(regression_layer_sizes[i], regression_layer_sizes[i + 1], bias=True)
            initialization_function_xavier(lin.weight)
            self.regression_layers.append(lin)
            self.regression_dropouts.append(nn.Dropout(drops[i]))

        self.predict = nn.Linear(regression_layer_sizes[-1], output_size, bias=True)
        initialization_function_xavier(self.predict.weight)

        # ---- Activations ----
        self.ELU = nn.ELU()
        self.LeakyReLU = nn.LeakyReLU(0.01)

        # ---- Learned "no content" sentinels ----
        self.no_comps_tensor = nn.Parameter(initialization_function_xavier(torch.zeros(1, embedding_size)))
        self.no_nodes_tensor = nn.Parameter(initialization_function_xavier(torch.zeros(1, embedding_size)))

    # --------------------------
    # Helpers for token encoders
    # --------------------------
    def _encode_tags(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        vectors: [B*N, T_tag, MAX_TAGS]
        returns pooled embedding:
          - if use_attn_tags: [B*N, tags_out_dim]
          - else (h_n flatten): [B*N, tags_out_dim * num_layers]
        """
        vectors = self.encode_vectors(vectors)
        outputs, (h_n, _) = self.transformation_vectors_embed(vectors)  # outputs [B*N,T, D], h_n [L*dir, B*N, H]
        if self.use_attn_tags:
            ctx, _ = self.tags_attn(outputs, query=None)                 # [B*N, 1, D]
            return ctx.squeeze(1)                                        # [B*N, D]
        else:
            # flatten all layers x directions
            return h_n.permute(1, 0, 2).reshape(vectors.size(0), -1)     # [B*N, H*dir*num_layers]

    def _encode_exprs(self, expr_seq: torch.Tensor) -> torch.Tensor:
        """
        expr_seq: [B*N, T_expr, expr_token_size]
        returns pooled embedding:
          - if use_attn_exprs: [B*N, expr_out_dim]
          - else (h_n): [B*N, expr_out_dim]
        """
        outputs, (h_n, _) = self.exprs_embed(expr_seq)                    # outputs [B*N,T, E], h_n [1, B*N, E]
        if self.use_attn_exprs:
            ctx, _ = self.expr_attn(outputs, query=None)                  # [B*N, 1, E]
            return ctx.squeeze(1)                                         # [B*N, E]
        else:
            return h_n.permute(1, 0, 2).reshape(expr_seq.size(0), -1)     # [B*N, E]

    # --------------------------
    # Recursive node embedder
    # --------------------------
    def get_hidden_state(self, node, comps_embeddings: torch.Tensor, loops_tensor: torch.Tensor):
        """
        Returns: x: [B, 1, embedding_size] for this node
        - Aggregates children and computations using LSTMs + (optional) attention.
        """
        B = comps_embeddings.shape[0]
        embedding_size = comps_embeddings.shape[-1]

        # ---- Recurse into children ----
        child_states = []
        for child in node["child_list"]:
            child_states.append(self.get_hidden_state(child, comps_embeddings, loops_tensor))  # each [B,1,D]

        # ---- Aggregate child loops ----
        if len(child_states) > 0:
            H_nodes = torch.cat(child_states, dim=1)                             # [B, T_child, D]
            _, (nodes_h_n, _) = self.nodes_lstm(H_nodes)                         # nodes_h_n: [1,B,D]
            nodes_h_n = nodes_h_n.permute(1, 0, 2)                               # [B,1,D]

            if self.use_cross_attn_children or self.use_attn_nodes:
                # Build loop-conditioned query if cross-attn enabled
                loop_q = None
                if self.use_cross_attn_children:
                    selected_loop_tensor = torch.index_select(
                        loops_tensor, 1, node["loop_index"].to(self.device)
                    )                                                            # [B,1,loops_tensor_size]
                    loop_q = self.loop_q_proj(selected_loop_tensor)              # [B,1,D]
                # Dot attention over child sequence outputs (use outputs from nodes_lstm?):
                # We want attention over the SEQUENCE representations of children.
                # Use H_nodes directly (pre-LSTM) or outputs from nodes_lstm?
                # Keep it consistent: attend over H_nodes_lstm outputs -> better contextualized
                nodes_outputs, _ = self.nodes_lstm(H_nodes)                      # [B,T_child,D]
                nodes_ctx, _ = self.nodes_attn(nodes_outputs, query=loop_q)     # [B,1,D]
                nodes_rep = nodes_ctx
            else:
                nodes_rep = nodes_h_n                                            # [B,1,D]
        else:
            # No children: use learned sentinel
            nodes_rep = torch.unsqueeze(self.no_nodes_tensor, 0).expand(B, -1, -1)  # [B,1,D]

        # ---- Aggregate computations under this loop ----
        if node["has_comps"]:
            # select computation embeddings for this node
            selected = torch.index_select(
                comps_embeddings, 1, node["computations_indices"].to(self.device)
            )                                                                    # [B, T_comp, D]
            comps_outputs, (comps_h_n, _) = self.comps_lstm(selected)            # [B,T_comp,D], [1,B,D]
            comps_h_n = comps_h_n.permute(1, 0, 2)                               # [B,1,D]

            if self.use_cross_attn_comps or self.use_attn_comps:
                # loop-conditioned query if cross-attn enabled
                loop_q = None
                if self.use_cross_attn_comps:
                    selected_loop_tensor = torch.index_select(
                        loops_tensor, 1, node["loop_index"].to(self.device)
                    )                                                            # [B,1,loops_tensor_size]
                    loop_q = self.loop_q_proj(selected_loop_tensor)              # [B,1,D]
                comps_ctx, _ = self.comps_attn(comps_outputs, query=loop_q)      # [B,1,D]
                comps_rep = comps_ctx
            else:
                comps_rep = comps_h_n
        else:
            comps_rep = torch.unsqueeze(self.no_comps_tensor, 0).expand(B, -1, -1)  # [B,1,D]

        # ---- Loop descriptor for this level ----
        selected_loop_tensor = torch.index_select(
            loops_tensor, 1, node["loop_index"].to(self.device)
        )                                                                        # [B,1,loops_tensor_size]

        # ---- Per-level fusion MLP ----
        x = torch.cat((nodes_rep, comps_rep, selected_loop_tensor), dim=2)       # [B,1, 2D + loops_tensor_size]
        for i in range(len(self.concat_layers)):
            x = self.concat_layers[i](x)
            x = self.concat_dropouts[i](self.ELU(x))
        return x                                                                 # [B,1,D]

    # --------------------------
    # Forward
    # --------------------------
    def forward(self, tree_tensors):
        """
        tree_tensors = (
          tree,
          comps_tensor_first_part,        # [B, Ncomp, ?]
          comps_tensor_vectors,           # [B*Ncomp, T_tag, MAX_TAGS] already shaped
          comps_tensor_third_part,        # [B, Ncomp, ?]
          loops_tensor,                   # [B, Nloop, loops_tensor_size]
          functions_comps_expr_tree,      # [B, Ncomp, T_expr, expr_token_size]
        )
        """
        (tree,
         comps_tensor_first_part,
         comps_tensor_vectors,
         comps_tensor_third_part,
         loops_tensor,
         functions_comps_expr_tree) = tree_tensors

        # ---- Embed expressions (per computation) ----
        B, Ncomp, T_expr, expr_dim = functions_comps_expr_tree.shape
        expr_seq = functions_comps_expr_tree.view(B * Ncomp, T_expr, expr_dim)
        expr_embedding = self._encode_exprs(expr_seq)                            # [B*Ncomp, E_eff]

        # ---- Embed transformation vectors (per computation) ----
        # comps_tensor_vectors is already [B*Ncomp, T_tag, MAX_TAGS]
        tags_embedding = self._encode_tags(comps_tensor_vectors)                 # [B*Ncomp, T_eff]

        # ---- Gather the other computation parts ----
        _, _, F_dim = comps_tensor_first_part.shape
        first_part = comps_tensor_first_part.to(self.device).view(B * Ncomp, F_dim)
        _, _, T_dim = comps_tensor_third_part.shape
        third_part = comps_tensor_third_part.to(self.device).view(B * Ncomp, T_dim)

        # ---- Concat to form per-computation feature ----
        comp_input = torch.cat((first_part, tags_embedding, third_part, expr_embedding), dim=1)  # [B*Ncomp, base_in]
        comp_input = comp_input.view(B, Ncomp, -1)                                               # [B, Ncomp, base_in]

        # ---- Feed-forward to get final computation embeddings ----
        x = comp_input
        for i in range(len(self.comp_embedding_layers)):
            x = self.comp_embedding_layers[i](x)
            x = self.comp_embedding_dropouts[i](self.ELU(x))
        comps_embeddings = x                                                                     # [B, Ncomp, D]

        # ---- For each root: recursively build representation ----
        roots_list = []
        for root in tree["roots"]:
            roots_list.append(self.get_hidden_state(root, comps_embeddings, loops_tensor))       # each [B,1,D]

        roots_tensor = torch.cat(roots_list, dim=1)                                              # [B, T_root, D]

        # ---- Aggregate roots ----
        roots_outputs, (roots_h_n, _) = self.roots_lstm(roots_tensor)                            # [B,T_root,D], [1,B,D]
        roots_h_n = roots_h_n.permute(1, 0, 2)                                                   # [B,1,D]

        if self.use_attn_roots:
            roots_ctx, _ = self.roots_attn(roots_outputs, query=None)                            # [B,1,D]
            x = roots_ctx
        else:
            x = roots_h_n

        # ---- Regression head ----
        for i in range(len(self.regression_layers)):
            x = self.regression_layers[i](x)
            x = self.regression_dropouts[i](self.ELU(x))

        out = self.predict(x)                                                                    # [B,1,output_size]
        # Force non-negative via LeakyReLU (as in original)
        return self.LeakyReLU(out[:, 0, 0])