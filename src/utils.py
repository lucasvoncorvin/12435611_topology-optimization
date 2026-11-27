
from unicodedata import mirrored
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from .models.ViTV1 import ViT_V1
from .models.ViTV2 import ViT_V2
from .models.ViTV3 import ViT_V3
from .models.ViTV4 import ViT_V4
from .model_types import (
    BatchModelInput,
    BatchModelOutput,
    Checkpoint,
    ModelInput,
    ModelOutput,
    ModelSpec,
)

MODEL_REGISTRY = {
    "vit_v1": ViT_V1,
    "vit_v2": ViT_V2,
    "vit_v3": ViT_V3,
    "vit_v4": ViT_V4
}
    

def _rotate_boundary_conditions(boundary_conditions: torch.Tensor, n_rotations: int) -> torch.Tensor:
    """Rotate the 1 hot encoded boundary conditions tensor by 90 degrees k times."""
    
    assert boundary_conditions.numel() == 256, "Boundary conditions tensor must have 256 elements"
    
    
    nodes = torch.tensor(boundary_node_indices(), device=boundary_conditions.device)
    grid = boundary_conditions.new_zeros((65, 65))
    grid[nodes % 65, nodes // 65] = boundary_conditions
    rotated_grid = torch.rot90(grid, k=n_rotations, dims=(0, 1))
    return rotated_grid[nodes % 65, nodes // 65]
    
def _rotate_load_position(load_position: torch.Tensor, n_rotations: int) -> torch.Tensor:
    """Rotate the load position by 90 degrees k times."""
    assert load_position.numel() == 2, "Load position tensor must have 2 elements"
    x, y = load_position
    for _ in range(n_rotations % 4):
        x, y = y, 1.0 - x
    return torch.stack((x, y))

def _rotate_load_vector(load_vector: torch.Tensor, n_rotations: int) -> torch.Tensor:
    """Rotate the load vector by 90 degrees k times, with positive y down."""
    assert load_vector.numel() == 2, "Load vector tensor must have 2 elements"
    fx, fy = load_vector
    for _ in range(n_rotations % 4):
        fx, fy = fy, -fx
    return torch.stack((fx, fy))

# covered by normalize_sample (normalize_rotation) test
def rotate_sample(model_input: ModelInput, model_output: ModelOutput | None = None, n_rotations: int = 0) -> tuple[ModelInput, ModelOutput | None]:
    
    if n_rotations == 0:
        return model_input, model_output
    rotated_input = ModelInput(
        stress=torch.rot90(model_input.stress, k=n_rotations, dims=(0, 1)),
        strain_energy=torch.rot90(model_input.strain_energy, k=n_rotations, dims=(0, 1)),
        volume_fraction=model_input.volume_fraction,
        load_position=_rotate_load_position(model_input.load_position, n_rotations),
        load_vector=_rotate_load_vector(model_input.load_vector, n_rotations),
        boundary_conditions=_rotate_boundary_conditions(model_input.boundary_conditions, n_rotations)
    )
    rotated_output = None
    if model_output is not None:
        rotated_output = ModelOutput(
            density=torch.rot90(model_output.density, k=n_rotations, dims=(0, 1))
        )
    return rotated_input, rotated_output


def _mirror_boundary_conditions(boundary_conditions: torch.Tensor) -> torch.Tensor:
    # todo: CHECK THIS
    nodes = torch.tensor(boundary_node_indices(), device=boundary_conditions.device)
    grid = boundary_conditions.new_zeros((65, 65))
    grid[nodes % 65, nodes // 65] = boundary_conditions
    return grid.flip(1)[nodes % 65, nodes // 65]

def _mirror_load_position(load_position: torch.Tensor) -> torch.Tensor:
    """Mirror the load position along the vertical axis."""
    assert load_position.numel() == 2, "Load position tensor must have 2 elements"
    x, y = load_position
    return torch.stack((1.0 - x, y))

def _mirror_load_vector(load_vector: torch.Tensor) -> torch.Tensor:
    """Mirror the load vector along the vertical axis."""
    assert load_vector.numel() == 2, "Load vector must have 2 elements"
    fx, fy = load_vector
    return torch.stack((-fx, fy))

# covered by normalize_sample (normalize chirality) test
def mirror_sample(model_input: ModelInput, model_output: ModelOutput | None = None) -> tuple[ModelInput, ModelOutput | None]:
    mirrored_input = ModelInput(
        stress=torch.flip(model_input.stress, dims=(1,)),
        strain_energy=torch.flip(model_input.strain_energy, dims=(1,)),
        volume_fraction=model_input.volume_fraction,
        load_position=_mirror_load_position(model_input.load_position),
        load_vector=_mirror_load_vector(model_input.load_vector),
        boundary_conditions=_mirror_boundary_conditions(model_input.boundary_conditions)
    )
    mirrored_output = None
    if model_output is not None:
        mirrored_output = ModelOutput(
            density=torch.flip(model_output.density, dims=(1,))
        )
    return mirrored_input, mirrored_output


# covered by normalize_sample test
def normalize_sample_rotation(model_input: ModelInput, model_output: ModelOutput | None = None) -> tuple[int, ModelInput, ModelOutput | None]:
    """Rotate a boundary load onto the top edge (y = 0)."""
    
    epsilon = 0.5 / 64 
    
    n_rotations = 3
    if abs(model_input.load_position[1] - 0.0) < epsilon:
        n_rotations = 0
    elif abs(model_input.load_position[0] - 1.0) < epsilon:
        n_rotations = 1
    elif abs(model_input.load_position[1] - 1.0) < epsilon:
        n_rotations = 2
    
    return n_rotations, *rotate_sample(model_input=model_input, model_output=model_output, n_rotations=n_rotations)

# covered by normalize_sample test
def normalize_sample_chirality(model_input: ModelInput, model_output: ModelOutput | None = None) -> tuple[bool, ModelInput, ModelOutput | None]:
    """Normalize the chirality (mirror symmetry) of the sample's input tensors."""
    mirrored = False
    
    if model_input.load_position[0] > 0.5: 
        return True, *mirror_sample(model_input, model_output)
    else:
        return False, model_input, model_output

def normalize_sample(model_input: ModelInput, model_output: ModelOutput | None = None) -> tuple[int, bool, ModelInput, ModelOutput | None]:
    """Normalize both the rotation and chirality of the sample."""
    n_rotations, model_input, model_output = normalize_sample_rotation(model_input, model_output)
    mirrored, model_input, model_output = normalize_sample_chirality(model_input, model_output)
    return n_rotations, mirrored, model_input, model_output

def encode_boundary_conditions(fixed_dofs):
    """Encode paired fixed degrees of freedom as a 256-entry boundary-node mask."""
    fixed_dofs_set = set(fixed_dofs)
    for dof in fixed_dofs_set:
        partner = dof + 1 if dof % 2 == 0 else dof - 1 # make dof belonging to the same node have the same value
        if partner not in fixed_dofs_set:
            raise ValueError("DOF pairs must be consecutive.")

    boundary_nodes = boundary_node_indices()
    boundary_positions = {node: index for index, node in enumerate(boundary_nodes)}
    boundary_conditions = torch.zeros(len(boundary_nodes))
    for dof in fixed_dofs_set:
        node = dof // 2
        if node not in boundary_positions:
            raise ValueError("Fixed DOFs must belong to boundary nodes.")
        boundary_conditions[boundary_positions[node]] = 1

    return boundary_conditions

def collate_samples(
    batch: list[tuple[ModelInput, ModelOutput]],
) -> tuple[BatchModelInput, BatchModelOutput]:
    """Combine a batch of individual samples into batched model input and output."""
    
    inputs, outputs = zip(*batch)

    batched_input = BatchModelInput(
        stresses=torch.stack([sample.stress for sample in inputs]),
        strain_energies=torch.stack([sample.strain_energy for sample in inputs]),
        volume_fractions=torch.stack([sample.volume_fraction for sample in inputs]),
        load_positions=torch.stack([sample.load_position for sample in inputs]),
        load_vectors=torch.stack([sample.load_vector for sample in inputs]),
        boundary_conditions=torch.stack(
            [sample.boundary_conditions for sample in inputs]
        ),
    )
    batched_output = BatchModelOutput(
        densities=torch.stack([sample.density for sample in outputs])
    )

    return batched_input, batched_output

def idx_to_coordinates(idx, normalize=False):
    """translate a node index to its (normalized) coordinates within the grid
    
    Mental model:
    (0,0)    (1, 0)
    --------> x
    |
    |
    v y
    (0, 1)
    """
    
    x = idx // 65 # 65 nodes per column
    y = idx % 65
    if normalize:
        return x / 64, y / 64
    else:
        return x, y

def coordinates_to_idx(x, y, normalized=False) -> int:
    """translate (normalized) coordinates within the grid to a node index"""
    if normalized:
        x = int(round(x * 64))
        y = int(round(y * 64))
    return x * 65 + y

def boundary_node_indices() -> list[int]:
    """Return boundary nodes in the same index order used by the parameter vector."""
    return [
        node
        for node in range(65 * 65)
        if node // 65 in (0, 64) or node % 65 in (0, 64)
    ]
    
def build_model(model_spec: ModelSpec) -> nn.Module:
    """Construct a registered model from the current model specification."""
    model_class = MODEL_REGISTRY.get(model_spec.model_name)
    if model_class is None:
        raise ValueError(f"Model {model_spec.model_name} not found in registry")
    return model_class(model_spec)

def load_model_from_checkpoint_path(checkpoint_path: Path, device: torch.device = torch.device("cpu")) -> nn.Module:
    """
    Load a model from a checkpoint.

    checkpoint: the checkpoint containing the model specification and state dict
    device: the device to load the model onto (default: CPU)

    Returns the loaded model as an nn.Module.
    """
    checkpoint = Checkpoint.load(checkpoint_path)
    return checkpoint.get_model(device=device)