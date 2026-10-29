"""Useful functions for Vision Transformer (ViT) models"""

import torch
import torch.nn as nn
import math
from enum import StrEnum

class OrientationMode(StrEnum):
    CENTER_OF_MASS = "center_of_mass"
    LOAD = "load"
    BOUNDARY_CONDITIONS = "boundary_conditions"
    


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




def _positional_encoding(position, dim_encoding, device, dtype):
    """Generates a frequency-based positional encoding for a given position and dimension
    
    Args:
        position (int): The position for which to generate the encoding.
        dim_encoding (int): The dimension of the positional encoding.

    Returns:
        torch.Tensor: Positional encoding vector of shape (dim_encoding,)
    """
    pe = torch.zeros(dim_encoding, device=device, dtype=dtype)
    N = 10000
    
    for k in range(0, dim_encoding, 2):
        pe[k] = math.sin(position / (N ** (k / dim_encoding)))
        pe[k+1] = math.cos(position / (N ** (k / dim_encoding)))
    return pe
    
def build_positional_encoding(patch_size, embedding_dims) -> torch.Tensor:
    """Builds the positional encoding for all patches in a 64x64 grid given the patch size and embedding dimensions.
    
    Returns positional encodings for all patches as a tensor of shape (n_patches, embedding_dims)
    """
    if embedding_dims % 4 != 0:
        raise ValueError("embedding_dims must be divisible by 4")

    side = 64 // patch_size
    positions = torch.arange(side * side, dtype=torch.float64)
    frequencies = torch.arange(0, embedding_dims // 2, 2, dtype=torch.float64)

    def encode(position):
        angles = position[:, None] / (10000 ** (frequencies / (embedding_dims // 2)))
        return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(start_dim=1)

    rows = (positions // side)
    cols = (positions % side)
    return torch.cat((encode(rows), encode(cols)), dim=1).float()

def add_positional_encoding(patches, patch_size, embedding_dims, device, dtype):
    """Adds frequency-based positional encoding to the patch embeddings in place.
    
    Args:
        patches: Tensor of shape (batch_size, n_patches, patch_size)
        patch_size (int): The size of each image patch.
        embedding_dims (int): The dimension of the patch embeddings.
        device: The device on which to create the positional encodings.
        dtype: The data type of the positional encodings.
    """
    
    assert embedding_dims % 4 == 0, "Embedding dimensions must be divisible by 4 for positional encoding"
    
    for k in range(patches.shape[1]):
        row = k // (64 // patch_size)
        col = k % (64 // patch_size)
        patches[:, k, 0:embedding_dims//2] += _positional_encoding(row, embedding_dims//2, device, dtype)
        patches[:,k, embedding_dims//2:embedding_dims] += _positional_encoding(col, embedding_dims//2, device, dtype)
            


