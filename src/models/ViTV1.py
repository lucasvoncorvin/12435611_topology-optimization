import torch
import torch.nn as nn
import math
from .utils import add_positional_encoding, attention


class ViT_V1(torch.nn.Module):
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
        
        super(ViT_V1, self).__init__()
        
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
        patch_embeddings = self.linear_patch(patches)
        add_positional_encoding(patch_embeddings, self.patch_size, self.embedding_dims, device=patch_embeddings.device, dtype=patch_embeddings.dtype) # adds positional encoding to the patch embeddings
        
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
        
        return torch.sigmoid(res) # sigmoid to map result to the range [0, 1]
    
    
    def name(self):
        """Defines a name based on the model architecture."""
        return f"{self.__class__.__name__}_e{self.embedding_dims}_q{self.query_dims}_b{self.n_attention_blocks}_h{self.n_attention_heads}_p{self.patch_size}"
    