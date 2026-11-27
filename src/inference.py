"""Create model inputs and run inference on one sample."""

from dataclasses import replace

import torch
import numpy as np

from data.generate_data import _generate_stress_and_strain_distribution, _get_bcs
from .model_types import BatchModelInput, ModelInput, ModelOutput
from .utils import boundary_node_indices, coordinates_to_idx, normalize_sample


def build_model_input(load_position, load_vector, boundary_conditions, volume_fraction) -> ModelInput:
    """Compute the input stress and strain energy distributions and create a raw ModelInput.
    
    Args:
        load_position: The position where the load is applied. [x, y] normalized coordinates.
        load_vector: The vector representing the applied load. [fx, fy] components.
        boundary_conditions: A 256-entry mask ordered by boundary_node_indices(); true fixes both displacement components at a boundary node.
        volume_fraction: The volume fraction of the material as a scalar value.

    Returns:
        A ModelInput instance containing the computed stress and strain energy distributions.
    
    """

    load_position = torch.as_tensor(load_position, dtype=torch.float32)
    load_vector = torch.as_tensor(load_vector, dtype=torch.float32)
    boundary_conditions = torch.as_tensor(boundary_conditions, dtype=torch.float32)
    volume_fraction = torch.as_tensor(volume_fraction, dtype=torch.float32).reshape(1)
    
    # compute distributions
    node = coordinates_to_idx(*load_position.tolist(), normalized=True)
    force = np.zeros((2 * 65 * 65, 1))
    force[2 * node:2 * node + 2, 0] = (load_vector[0].item(), -load_vector[1].item())
    fixed_nodes = np.asarray(boundary_node_indices())[boundary_conditions.bool().numpy()]
    fixed_dofs = np.column_stack((2 * fixed_nodes, 2 * fixed_nodes + 1)).reshape(-1)
    stress, strain_energy = _generate_stress_and_strain_distribution(_get_bcs(force, fixed_dofs))
    
    # create ModelInput instance
    return ModelInput(
        load_position=load_position,
        load_vector=load_vector,
        boundary_conditions=boundary_conditions,
        volume_fraction=volume_fraction,
        stress=torch.from_numpy(stress.T.copy()).float(),
        strain_energy=torch.from_numpy(strain_energy.T.copy()).float()
    )

def preprocess_sample(sample: ModelInput) -> tuple[BatchModelInput, int, bool]:
    """Scale input and normalize orientation."""
    
    # Scale stress and strain energy fields using log1p
    sample = replace(sample, stress=torch.log1p(sample.stress), strain_energy=torch.log1p(sample.strain_energy))
    
    # Normalize Orientation
    rotation, mirrored, sample, _ = normalize_sample(sample)

    # Create a batch with a single sample for model input
    batch = BatchModelInput(
        stresses=sample.stress.unsqueeze(0),
        strain_energies=sample.strain_energy.unsqueeze(0),
        volume_fractions=sample.volume_fraction.unsqueeze(0),
        load_positions=sample.load_position.unsqueeze(0),
        load_vectors=sample.load_vector.unsqueeze(0),
        boundary_conditions=sample.boundary_conditions.unsqueeze(0),
    )
    return batch, rotation, mirrored


def postprocess(density: torch.Tensor, rotation: int, mirrored: bool) -> ModelOutput:
    """Restore the predicted density field to the input orientation."""
    if mirrored:
        density = density.flip(1)
    if rotation:
        density = torch.rot90(density, -rotation, dims=(0, 1))
    return ModelOutput(density=density.detach().cpu())


def infer(model: torch.nn.Module, sample: ModelInput, device: torch.device | str = "cpu") -> ModelOutput:
    """Predict a density field from raw stress and strain-energy fields."""
    batch, rotation, mirrored = preprocess_sample(sample)
    device = torch.device(device)
    model = model.to(device).eval()
    with torch.inference_mode():
        density = model(batch.to(device)).densities[0]
    return postprocess(density, rotation, mirrored)
