import torch
import torch.nn as nn
import math

EMBEDDING_DIMS = 16
QUERY_DIMS = 8
N_ATTENTION_BLOCKS = 4

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




def _positional_encoding(position, d_model=EMBEDDING_DIMS//2):
    """Generates a frequency-based positional encoding for a given position and dimension"""
    pe = torch.zeros(d_model)
    N = 10000
    
    for k in range(0, d_model, 2):
        pe[k] = math.sin(position / (N ** (k / d_model)))
        pe[k+1] = math.cos(position / (N ** (k / d_model)))
    return pe
    

def add_positional_encoding(patches):
    """Adds frequency-based positional encoding to the patch embeddings in place.
    
    Args:
        patches: Tensor of shape (batch_size, n_patches, patch_size)
    """
    
    assert EMBEDDING_DIMS % 2 == 0, "Embedding dimensions must be even for positional encoding"
    
    for k in range(patches.shape[1]):
        row = k // 8
        col = k % 8
        patches[:, k, 0:EMBEDDING_DIMS//2] += _positional_encoding(row)
        patches[:,k, EMBEDDING_DIMS//2:EMBEDDING_DIMS] += _positional_encoding(col)
            


class ViT(torch.nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
        
        self.n_patches = 8*8 # 8x8 image patches -> 8*8 = 64 patch in 64x64 grid
        self.patch_size = 8*8
        
        # input and output
        self.linear_patch = nn.Linear(self.patch_size, EMBEDDING_DIMS)
        self.linear_param = nn.Linear(1+4+4*64, EMBEDDING_DIMS) # dimension of the parameter vector
        self.readout = nn.Linear(self.n_patches*EMBEDDING_DIMS + EMBEDDING_DIMS, self.n_patches*self.patch_size) # get the whole optimized topology at once
        
        # attention stuff
        self.linear_v = nn.Linear(EMBEDDING_DIMS, EMBEDDING_DIMS)
        self.linear_q = nn.Linear(EMBEDDING_DIMS, QUERY_DIMS)
        self.linear_k = nn.Linear(EMBEDDING_DIMS, QUERY_DIMS)
        self.linear = nn.Linear(EMBEDDING_DIMS, EMBEDDING_DIMS)
        
        
        
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
        add_positional_encoding(patch_embeddings) # adds positional encoding to the patch embeddings
        
        embeddings = torch.cat([param_embeddings, patch_embeddings], dim=1)
        
        for i in range(N_ATTENTION_BLOCKS):
            
            # attention block
            v = self.linear_v(embeddings)
            q = self.linear_q(embeddings)
            k = self.linear_k(embeddings)
            embeddings = attention(k, q, v)
            
            # linear layer after attention
            embeddings = self.linear(embeddings)
        
        # readout
        res = self.readout(embeddings.flatten(start_dim=1)) # flatten again for the readout
        
        return res