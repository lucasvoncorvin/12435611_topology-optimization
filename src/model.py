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
    scores = (queries @ keys) / math.sqrt(EMBEDDING_DIMS)
    weights = torch.softmax(scores, dim=-1)
    return weights @ values

def params_vectors(load, fixed_dofs, volume_fraction):
    """Returns a representation of the load, the boundary conditions and the volume fraction
    with length 1 + 4 + 4*64 = 256 (volume fraction + load representation + boundary conditions)
    """
    
    assert len(fixed_dofs) == 65*65*2, "Number of fixed DOFs must be EDGE_LENGTH squared times 2 (number of nodes * DOFs per node)"
    # ensure that dofs of the same node are the same value, since we fix both dimensions
    for i in range(0, len(fixed_dofs), 2):
        assert fixed_dofs[i] == fixed_dofs[i+1], "DOFs of the same node must be the same"
    assert 0 <= volume_fraction <= 1, "Volume fraction must be between 0 and 1"
    
    # first two elements of the load representation are the normalized coordinates of the load application point
    loads = torch.zeros(4)
    loads[0], loads[1] = _idx_to_coordinates(load[0], normalize=True)
    
    # second two elements are the load vector
    loads[2] = load[1]
    loads[3] = load[2]
    
    # remaining elements give the fixed nodes by one-hot encoding counter clockwise starting from the origin (upper-left corner)
    # 1: fixed in both dimensions, 0: free
    
    bc = torch.zeros(4*64) # number of boundary nodes is 4*64
    counter = 0
    for i in range(len(fixed_dofs) // 2):
        row, col = _idx_to_coordinates(i, normalize=False)
        if row in [0, 64] and col in [0, 64]:
            bc[counter] = fixed_dofs[i*2]
            counter += 1
    

    return torch.cat([torch.tensor([volume_fraction]), loads, bc])


def _positional_encoding(position, d_model=EMBEDDING_DIMS//2):
    """Generates a frequency-based positional encoding for a given position and dimension"""
    pe = torch.zeros(d_model)
    N = 10000
    
    for k in range(0, d_model, 2):
        pe[k] = math.sin(position / (N ** (k / d_model)))
        pe[k+1] = math.cos(position / (N ** (k / d_model)))
    return pe
    

def add_positional_encoding(patches):
    """Adds frequency-based positional encoding to the patch embeddings in place."""
    
    assert EMBEDDING_DIMS % 2 == 0, "Embedding dimensions must be even for positional encoding"
    
    for k in range(patches.shape[0]):
        row = k // 8
        col = k % 8
        patches[k, 0:EMBEDDING_DIMS//2] += _positional_encoding(row)
        patches[k, EMBEDDING_DIMS//2:EMBEDDING_DIMS] += _positional_encoding(col)
            


class ViT(torch.nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
        
        self.n_patches = 8*8 # 8x8 image patches -> 8*8 = 64 patch in 64x64 grid
        self.patch_size = 8*8
        
        # input and output
        self.linear_embed = nn.Linear(self.patch_size, EMBEDDING_DIMS)
        self.linear_param = nn.Linear(1+4+4*64, EMBEDDING_DIMS) # dimension of the parameter vector
        self.readout = nn.Linear(EMBEDDING_DIMS*self.n_patches, self.n_patches*self.patch_size) # get the whole optimized topology at once
        
        # attention stuff
        self.linear_v = nn.Linear(EMBEDDING_DIMS, EMBEDDING_DIMS)
        self.linear_q = nn.Linear(EMBEDDING_DIMS, QUERY_DIMS)
        self.linear_k = nn.Linear(EMBEDDING_DIMS, QUERY_DIMS)
        self.linear = nn.Linear(EMBEDDING_DIMS, EMBEDDING_DIMS)
        
        
        
    def forward(self, stress_distribution, load, boundary_dofs, volume_fraction):

        # flatten all patches and append to x
        patches = torch.zeros(self.n_patches, self.patch_size)
        counter = 0
        for row in range(8):
            for col in range(8):
                patches[counter, :] = stress_distribution[8*row:8*(row+1), 8*col:8*(col+1)].flatten()
        
        
        x = torch.zeros(1 + self.n_patches, EMBEDDING_DIMS, dtype=torch.float32, requires_grad=True)
        
        # embedding of parameters (load, boundary conditions, volume fraction)
        x[0, :] = self.linear_param(params_vectors(load, boundary_dofs, volume_fraction))
        
        # embedding of patches
        x[1::, :] = self.linear_embed(patches)
        add_positional_encoding(x[1::, :]) # adds positional encoding to the patch embeddings
        
        for i in range(N_ATTENTION_BLOCKS):
            
            # attention block
            v = self.linear_v(x)
            q = self.linear_q(x)
            k = self.linear_k(x)
            x = attention(k, q, v)
            
            # linear layer after attention
            x = self.linear(x)
        
        # readout
        x = self.readout(x)
        
        # reorder the output to match the original patch layout
        density = torch.zeros(8*8, 8*8)
        for i in range(self.n_patches):
            row = i // 8
            col = i % 8
            # todo: fill in the density matrix with the corresponding patch
        
        return density