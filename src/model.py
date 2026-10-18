import torch
import torch.nn as nn
import math


def _idx_to_coordinates(idx, normalize=False):
    """translate a node index to its (normalized) coordinates within the grid"""
    row = idx // 64
    col = idx % 64
    if normalize:
        return row / 64, col / 64
    else:
        return row, col

def attention(keys, queries, values):
    """Computes the attention function in batches.
    
    Args:
        keys: Tensor (batch_size, n_patches, QUERY_DIMS)
        queries: Tensor (batch_size, n_patches, QUERY_DIMS)
        values: Tensor (batch_size, n_patches, EMBEDDING_DIMS)

    Returns:
        Tensor (batch_size, n_patches, EMBEDDING_DIMS)
    """
    scores = (queries @ keys.transpose(-2, -1)) / math.sqrt(queries.shape[-1]) # ! Used ai here to help me with the correct transposition for the matrix multiplication
    weights = torch.softmax(scores, dim=-1)
    return weights @ values




def _positional_encoding(position, dim_encoding):
    """Generates a frequency-based positional encoding for a given position and dimension
    
    Args:
        position (int): The position for which to generate the encoding.
        dim_encoding (int): The dimension of the positional encoding.

    Returns:
        torch.Tensor: Positional encoding vector of shape (dim_encoding,)
    """
    pe = torch.zeros(dim_encoding)
    N = 10000
    
    for k in range(0, dim_encoding, 2):
        pe[k] = math.sin(position / (N ** (k / dim_encoding)))
        pe[k+1] = math.cos(position / (N ** (k / dim_encoding)))
    return pe
    

def add_positional_encoding(patches, embedding_dims):
    """Adds frequency-based positional encoding to the patch embeddings in place.
    
    Args:
        patches: Tensor of shape (batch_size, n_patches, patch_size)
    """
    
    assert embedding_dims % 2 == 0, "Embedding dimensions must be even for positional encoding"
    
    for k in range(patches.shape[1]):
        row = k // 8
        col = k % 8
        patches[:, k, 0:embedding_dims//2] += _positional_encoding(row, embedding_dims//2)
        patches[:,k, embedding_dims//2:embedding_dims] += _positional_encoding(col, embedding_dims//2)
            


class ViT(torch.nn.Module):
    def __init__(
            self,
            embedding_dims,
            query_dims,
            n_attention_blocks,
            n_attention_heads,
        ):
        
        assert embedding_dims % n_attention_heads == 0, "Embedding dimensions must be divisible by the number of attention heads"
        assert query_dims % n_attention_heads == 0, "Query dimensions must be divisible by the number of attention heads"
        
        super(ViT, self).__init__()
        
        # create seed for reproducibility
        torch.manual_seed(42)
        
        # hyperparameters
        self.embedding_dims = embedding_dims
        self.query_dims = query_dims
        self.n_attention_blocks = n_attention_blocks
        self.n_attention_heads = n_attention_heads
        
        self.n_patches = 8*8 # 8x8 image patches -> 8*8 = 64 patch in 64x64 grid
        self.patch_size = 8*8
        
        self.embedding_dims_head = self.embedding_dims // self.n_attention_heads
        self.query_dims_head = self.query_dims // self.n_attention_heads
        
        # input and output
        self.linear_patch = nn.Linear(self.patch_size, self.embedding_dims)
        self.linear_param = nn.Linear(1+4+4*64, self.embedding_dims) # dimension of the parameter vector
        self.readout = nn.Linear(self.n_patches*self.embedding_dims + self.embedding_dims, self.n_patches*self.patch_size) # get the whole optimized topology at once
        
        # attention stuff
        self.linear_vs = []
        self.linear_qs = []
        self.linear_ks = []
        self.linears = []
        for block in range(self.n_attention_blocks): # create separate linear layers for each attention block
            self.linear_vs.append(nn.Linear(self.embedding_dims, self.embedding_dims))
            self.linear_qs.append(nn.Linear(self.embedding_dims, self.query_dims))
            self.linear_ks.append(nn.Linear(self.embedding_dims, self.query_dims))
            self.linears.append(nn.Linear(self.embedding_dims, self.embedding_dims))
        
        self.n_weights = sum(p.numel() for p in self.parameters() if p.requires_grad) # total number of trainable parameters
                
        
    def forward(self, x):
        """Forward pass of the Vision Transformer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1+4+4*64 + n_patches*patch_size)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_patches*patch_size)
            Each output corresponds to the concatenated flattened patches of densities in the optimized topology.
        """

        n_samples = x.shape[0]
        
        params = x[:, 0:1+4+4*64].unsqueeze(1) # add one dimension to concatenate along with the patch embeddings
        patches = x[:, 1+4+4*64::].reshape(n_samples, self.n_patches, self.patch_size)
        
    
        # embedding of parameters (load, boundary conditions, volume fraction)
        param_embeddings = self.linear_param(params) 
        
        # embedding of patches
        patch_embeddings = self.linear_patch(patches)
        add_positional_encoding(patch_embeddings, self.embedding_dims) # adds positional encoding to the patch embeddings
        
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
        
        return res
    
    
    def name(self):
        """Defines a name based on the hyperparameters of the model."""
        return f"ViT_{self.n_weights}_{self.embedding_dims}_{self.query_dims}_{self.n_attention_blocks}_{self.n_attention_heads}"
    