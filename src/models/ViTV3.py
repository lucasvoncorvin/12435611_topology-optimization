import torch
import torch.nn as nn
import math
from .utils import add_positional_encoding, attention, build_positional_encoding


class ViT_V3(torch.nn.Module):
    """
    Changelog:
    
    Use pytorch implementation of multihead attention
    """
    def __init__(
            self,
            embedding_dims,
            query_dims,
            n_attention_blocks,
            n_attention_heads,
            patch_size,
        ):
        
        assert __file__.endswith(self.__class__.__name__[-2::] + ".py") # just check I actually made sure to align the versions (up to V9)
        assert embedding_dims % n_attention_heads == 0, "Embedding dimensions must be divisible by the number of attention heads"
        assert query_dims % n_attention_heads == 0, "Query dimensions must be divisible by the number of attention heads"
        
        super(ViT_V3, self).__init__()
        
        # create seed for reproducibility
        torch.manual_seed(42)
        
        # hyperparameters
        self.embedding_dims = embedding_dims
        self.query_dims = query_dims
        self.n_attention_blocks = n_attention_blocks
        self.n_attention_heads = n_attention_heads
        
        self.n_patches = (64 // patch_size) * (64 // patch_size) 
        self.patch_size = patch_size
        self.elements_per_patch = patch_size**2
        
        self.embedding_dims_head = self.embedding_dims // self.n_attention_heads
        self.query_dims_head = self.query_dims // self.n_attention_heads
        
        # input and output
        self.linear_patch = nn.Linear(self.elements_per_patch, self.embedding_dims)
        self.linear_param = nn.Linear(1+4+4*64, self.embedding_dims) # dimension of the parameter vector
        self.readout = nn.Linear(self.embedding_dims, self.elements_per_patch) # get the whole optimized topology at once
        
        # attention stuff
        
        
        linears = []
        attention_norms = []
        mlps = []
        mlp_norms = []
        multihead_attentions = []
        for block in range(self.n_attention_blocks): 
            
            
            linears.append(nn.Linear(self.embedding_dims, self.embedding_dims))
            
            
            multihead_attentions.append(nn.MultiheadAttention(
                embed_dim=self.embedding_dims,
                num_heads=self.n_attention_heads,
                batch_first=True,
            ))
            
            mlps.append(nn.Sequential(
                nn.Linear(self.embedding_dims, 4*self.embedding_dims), # researched a bit and that this is often 4 times the input of the mlp
                nn.ReLU(),
                nn.Linear(4*self.embedding_dims, self.embedding_dims)
            )) # non-linear projection for each attention block
            
            attention_norms.append(nn.LayerNorm(self.embedding_dims)) # create separate layer norms for each attention block
            mlp_norms.append(nn.LayerNorm(self.embedding_dims)) # create separate layer norms for each MLP block
            
        self.linears = nn.ModuleList(linears)
        self.mlps = nn.ModuleList(mlps)
        self.multihead_attentions = nn.ModuleList(multihead_attentions)
        self.attention_norms = nn.ModuleList(attention_norms)
        self.mlp_norms = nn.ModuleList(mlp_norms)
        self.final_norm = nn.LayerNorm(self.embedding_dims)
        
        
        # positional encodings 
        self.register_buffer(
            "positional_encoding",
            build_positional_encoding(patch_size, embedding_dims),
            persistent=False,
        )
        
        
        self.n_weights = sum(p.numel() for p in self.parameters() if p.requires_grad) # total number of trainable parameters
                
        
    def forward(self, x):
        """Forward pass of the Vision Transformer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1+4+4*64 + n_patches*elements_per_patch)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_patches*elements_per_patch)
            Each output corresponds to the concatenated flattened patches of densities in the optimized topology.
        """

        n_samples = x.shape[0]
        
        params = x[:, 0:1+4+4*64].unsqueeze(1) # add one dimension to concatenate along with the patch embeddings
        patches = x[:, 1+4+4*64::].reshape(n_samples, self.n_patches, self.elements_per_patch)
        
    
        # embedding of parameters (load, boundary conditions, volume fraction)
        param_embeddings = self.linear_param(params) 
        
        # embedding of patches
        patch_embeddings = self.linear_patch(patches) + self.positional_encoding
        
        embeddings = torch.cat([param_embeddings, patch_embeddings], dim=1)
        
        for i in range(self.n_attention_blocks):
            normalized = self.attention_norms[i](embeddings)

            attn_out, _ = self.multihead_attentions[i](normalized, normalized, normalized, need_weights=False)
            embeddings = embeddings + attn_out
            
            embeddings = embeddings + self.mlps[i](self.mlp_norms[i](embeddings)) # apply MLP with its own layer norm
        
        # readout
        embeddings = self.final_norm(embeddings)
        res = self.readout(embeddings[:, 1:, :]) # readout for each patch embedding
        
        return torch.sigmoid(res.flatten(start_dim=1)) # sigmoid to map result to the range [0, 1]
    
    
    def name(self):
        """Defines a name based on the model architecture."""
        # self.query_dims do not affect the model name anymore
        return f"{self.__class__.__name__}_e{self.embedding_dims}_b{self.n_attention_blocks}_h{self.n_attention_heads}_p{self.patch_size}"
    