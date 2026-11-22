import torch
import torch.nn as nn
import math
from .utils import attention, build_parameter_vector, build_positional_encoding, patchify, unpatchify_grid
from ..model_types import BatchModelInput, BatchModelOutput, ModelSpec


class ViT_V1(torch.nn.Module):
    def __init__(self, model_spec: ModelSpec):
        
        assert model_spec.model_name == "vit_v1"
        assert __file__.endswith(self.__class__.__name__[-2::] + ".py") # just check I actually made sure to align the versions (up to V9)
        assert model_spec.embedding_dims % model_spec.n_attention_heads == 0, "Embedding dimensions must be divisible by the number of attention heads"
        assert model_spec.query_dims % model_spec.n_attention_heads == 0, "Query dimensions must be divisible by the number of attention heads"
        
        super(ViT_V1, self).__init__()

        if model_spec.patch_size <= 0 or 64 % model_spec.patch_size != 0:
            raise ValueError("patch_size must be positive and divide 64")
        if model_spec.patch_overlap != 0:
            raise ValueError("ViT_V1 does not support patch overlap")
        training_mode_config = model_spec.training_mode_config
        if training_mode_config and any((
            training_mode_config.attention_output_dropout_rate,
            training_mode_config.mlp_output_dropout_rate,
            training_mode_config.attention_hidden_dropout_rate,
        )):
            raise ValueError("ViT_V1 does not support dropout configuration")

        # create seed for reproducibility
        torch.manual_seed(42)
        
        # hyperparameters
        self.embedding_dims = model_spec.embedding_dims
        self.query_dims = model_spec.query_dims
        self.n_attention_blocks = model_spec.n_attention_blocks
        self.n_attention_heads = model_spec.n_attention_heads

        self.n_patches_per_side = 64 // model_spec.patch_size
        self.n_patches = self.n_patches_per_side**2
        self.patch_size = model_spec.patch_size
        self.patch_overlap = model_spec.patch_overlap
        self.elements_per_patch = self.patch_size**2

        self.attention_output_dropout_rate = 0.0
        self.mlp_output_dropout_rate = 0.0
        self.attention_hidden_dropout_rate = 0.0

        self.embedding_dims_head = self.embedding_dims // self.n_attention_heads
        self.query_dims_head = self.query_dims // self.n_attention_heads
        
        # input and output
        self.linear_patch = nn.Linear(self.elements_per_patch, self.embedding_dims)
        self.linear_param = nn.Linear(1+4+4*64, self.embedding_dims) # dimension of the parameter vector
        self.readout = nn.Linear(self.n_patches*self.embedding_dims + self.embedding_dims, self.n_patches*self.elements_per_patch) # get the whole optimized topology at once
        
        # attention stuff
        linear_vs = []
        linear_qs = []
        linear_ks = []
        linears = []
        for block in range(self.n_attention_blocks): # create separate linear layers for each attention block
            linear_vs.append(nn.Linear(self.embedding_dims, self.embedding_dims))
            linear_qs.append(nn.Linear(self.embedding_dims, self.query_dims))
            linear_ks.append(nn.Linear(self.embedding_dims, self.query_dims))
            linears.append(nn.Linear(self.embedding_dims, self.embedding_dims))
        self.linear_vs = nn.ModuleList(linear_vs)
        self.linear_qs = nn.ModuleList(linear_qs)
        self.linear_ks = nn.ModuleList(linear_ks)
        self.linears = nn.ModuleList(linears)
        
        # positional encodings 
        self.register_buffer(
            "positional_encoding",
            build_positional_encoding(self.patch_size, self.embedding_dims),
            persistent=False,
        )
        
        
        self.n_weights = sum(p.numel() for p in self.parameters() if p.requires_grad) # total number of trainable parameters
                
        
    def forward(self, batch: BatchModelInput) -> BatchModelOutput:
        """Run ViT_V1 using the batched model input and return a density image."""
        n_samples = batch.stresses.shape[0]
        params = build_parameter_vector(
            load_positions=batch.load_positions,
            load_vectors=batch.load_vectors,
            boundary_conditions=batch.boundary_conditions,
            volume_fractions=batch.volume_fractions,
        )
        # Keep the original stress-only encoder so existing V1 checkpoint weights remain compatible.
        patches = patchify(
            batch.stresses.unsqueeze(-1),
            grid_shape=(self.n_patches_per_side, self.n_patches_per_side),
            overlap=0,
        )
        patches = patches.reshape(n_samples, self.n_patches, self.elements_per_patch)
        
    
        # embedding of parameters (load, boundary conditions, volume fraction)
        param_embeddings = self.linear_param(params).unsqueeze(1)
        
        # embedding of patches
        patch_embeddings = self.linear_patch(patches) + self.positional_encoding
        
        embeddings = torch.cat([param_embeddings, patch_embeddings], dim=1)
        
        for i in range(self.n_attention_blocks):
            
            k = self.linear_ks[i](embeddings)
            q = self.linear_qs[i](embeddings)
            v = self.linear_vs[i](embeddings)
            
            embeddings_head = []
            for head_idx in range(self.n_attention_heads): # multi-head attention

                embeddings_head.append(attention(
                    k[:, :, head_idx * self.query_dims_head:(head_idx + 1) * self.query_dims_head],
                    q[:, :, head_idx * self.query_dims_head:(head_idx + 1) * self.query_dims_head],
                    v[:, :, head_idx * self.embedding_dims_head:(head_idx + 1) * self.embedding_dims_head]
                ))


            embeddings = self.linears[i](torch.cat(embeddings_head, dim=-1))
        
        # readout
        res = self.readout(embeddings.flatten(start_dim=1)) # flatten again for the readout
        patch_grid = res.reshape(
            n_samples,
            self.n_patches_per_side,
            self.n_patches_per_side,
            self.patch_size,
            self.patch_size,
        )
        return BatchModelOutput(densities=torch.sigmoid(unpatchify_grid(patch_grid)))
    
    
    def name(self):
        """Defines a name based on the model architecture."""
        return f"{self.__class__.__name__}_e{self.embedding_dims}_q{self.query_dims}_b{self.n_attention_blocks}_h{self.n_attention_heads}_p{self.patch_size}"
    