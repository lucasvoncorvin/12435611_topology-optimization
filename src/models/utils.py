"""Useful functions for Vision Transformer (ViT) models"""

import torch
import torch.nn as nn
import math
from enum import StrEnum

class OrientationMode(StrEnum):
    CENTER_OF_MASS = "center_of_mass"
    LOAD = "load"
    BOUNDARY_CONDITIONS = "boundary_conditions"
    

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

def build_parameter_vector(load_positions, load_vectors, boundary_conditions, volume_fractions):
    # todo: write tests for this function
    """Returns a representation of the load, the boundary conditions and the volume fraction
    with length 1 + 4 + 4*64 = 261 (volume fraction + load representation + boundary conditions)
    
    Args:
        load_positions (torch.Tensor): batch of load positions.
        load_directions (torch.Tensor): batch of load directions.
        boundary_conditions (torch.Tensor): batch of fixed degrees of freedom.
        volume_fractions (torch.Tensor): batch of volume fractions of the material.
    """
    
    return torch.cat([volume_fractions, load_positions, load_vectors, boundary_conditions], dim=1)

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
            
def patchify(images, grid_shape: tuple[int, int], overlap: int):
    """Divides images into overlapping patches.

    Args:
        images (torch.Tensor): Tensor of shape (batch_size, height, width, channels)
        grid_shape (tuple[int, int]): Number of patches along image height and width
        overlap (int): Number of extra pixels per patch dimension, centered as closely
            as integer pixels allow. The image is zero-padded at its edges so boundary
            patches have the same dimensions.

    Returns:
        torch.Tensor: Tensor of shape (batch_size, n_patches_x, n_patches_y, patch_height, patch_width, channels)
    """
    if images.ndim != 4:
        raise ValueError("images must have shape (batch_size, height, width, channels)")
    if len(grid_shape) != 2 or grid_shape[0] <= 0 or grid_shape[1] <= 0:
        raise ValueError("grid_shape must contain two positive dimensions")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    batch_size, height, width, channels = images.shape
    if height % grid_shape[0] != 0 or width % grid_shape[1] != 0:
        raise ValueError("Image dimensions must be divisible by grid shape")

    patch_base_height = height // grid_shape[0]
    patch_base_width = width // grid_shape[1]

    if overlap == 0:
        return (
            images.reshape(
                batch_size,
                grid_shape[0],
                patch_base_height,
                grid_shape[1],
                patch_base_width,
                channels,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
        )

    patch_height = patch_base_height + overlap
    patch_width = patch_base_width + overlap
    before = overlap // 2
    after = overlap - before
    padded_images = torch.nn.functional.pad(
        images,
        (0, 0, before, after, before, after),
    )
    return ( # ! AI helped here a bit with all the unfold, permute stuff
        padded_images.unfold(1, patch_height, patch_base_height)
        .unfold(2, patch_width, patch_base_width)
        .permute(0, 1, 2, 4, 5, 3)
        .contiguous()
    )


def unpatchify_grid(patches: torch.Tensor) -> torch.Tensor:
    """Reconstruct batched 64x64 images from non-overlapping patch grids."""
    if patches.ndim != 5:
        raise ValueError("patches must have shape (batch, grid_height, grid_width, patch_height, patch_width)")

    batch_size, grid_height, grid_width, patch_height, patch_width = patches.shape
    if grid_height * patch_height != 64 or grid_width * patch_width != 64:
        raise ValueError("Patch grid dimensions must reconstruct a 64x64 image")

    return (
        patches.permute(0, 1, 3, 2, 4)
        .contiguous()
        .reshape(batch_size, 64, 64)
    )
    
