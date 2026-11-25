# Functions to test
# normalize_sample (rotation and chirality)
# encode_boundary_conditions

from math import sqrt

from src.utils import normalize_sample, encode_boundary_conditions
from src.model_types import ModelInput, ModelOutput

import torch

from pytest import fixture

@fixture
def model_input():
    # boundary conditions (top right corner fixed, bottom right half fixed)
    
    boundary_conditions = torch.zeros(256)
    boundary_conditions[191] = 1  # top right corner fixed (x=64, y=0)
    # bottom right half fixed (y=64, x=32 through 64, including the midpoint)
    boundary_conditions[128:191:2] = 1  # bottom nodes alternate with top nodes for x=32 through 63
    boundary_conditions[255] = 1  # bottom right corner (x=64, y=64)
    
    
    # load position (right bottom quarter) -> we need one rotation and one reflection to normalize it
    load_position = torch.tensor([1.0, 48 / 64])  # right edge, 48 of 64 intervals down
    
    # load vector (60 degrees)
    load_vector = torch.tensor([0.5, sqrt(3) / 2])  # 60 degrees below the positive x-axis
    
    # stress and strain fields (64x64 grid)
    stress = torch.arange(64*64).reshape(64, 64)
    strain = torch.arange(64*64).reshape(64, 64) + 1  # just to differentiate from stress
    
    # dummy volume fraction
    volume_fraction = torch.tensor(0.5)  # just a placeholder value

    return ModelInput(
        boundary_conditions=boundary_conditions,
        load_position=load_position,
        load_vector=load_vector,
        stress=stress,
        strain_energy=strain,
        volume_fraction=volume_fraction
    )
    
@fixture
def model_output():
    return ModelOutput(density=torch.arange(64*64).reshape(64, 64))

@fixture
def normalized_input(model_input):
    # boundary conditions (top right corner and upper half of the left edge fixed)
    boundary_conditions = torch.zeros(256)
    boundary_conditions[191] = 1  # top right corner (x=64, y=0)
    boundary_conditions[:33] = 1  # left edge (x=0, y=0 through 32)

    # Rotation moves the load to (0.75, 0), then reflection moves it to (0.25, 0).
    load_position = torch.tensor([0.25, 0.0])
    load_vector = torch.tensor([-sqrt(3) / 2, -0.5])  # (fx, fy) maps to (-fy, -fx)

    # The two operations together reflect the fields across the opposite diagonal.
    stress = model_input.stress.T.flip((0, 1))
    strain = model_input.strain_energy.T.flip((0, 1))

    return ModelInput(
        boundary_conditions=boundary_conditions,
        load_position=load_position,
        load_vector=load_vector,
        stress=stress,
        strain_energy=strain,
        volume_fraction=model_input.volume_fraction,
    )
    
@fixture
def normalized_output(model_output):
    # Rotation followed by reflection maps the density across the opposite diagonal.
    return ModelOutput(density=model_output.density.T.flip((0, 1)))




def test_normalize_sample(model_input, model_output, normalized_input, normalized_output):
    _, _, normalized, actual_output = normalize_sample(model_input, model_output)
    assert torch.allclose(normalized.boundary_conditions, normalized_input.boundary_conditions)
    assert torch.allclose(normalized.load_position, normalized_input.load_position)
    assert torch.allclose(normalized.load_vector, normalized_input.load_vector)
    assert torch.allclose(normalized.stress, normalized_input.stress)
    assert torch.allclose(normalized.strain_energy, normalized_input.strain_energy)
    assert torch.allclose(normalized.volume_fraction, normalized_input.volume_fraction)
    assert actual_output is not None
    assert torch.allclose(actual_output.density, normalized_output.density)


def test_encode_boundary_conditions():
    boundary_nodes = [
        *range(65),  # left edge
        *(65 * x + y for x in range(1, 64) for y in (0, 64)),  # top and bottom nodes in each interior column
        *range(64 * 65, 65 * 65),  # right edge
    ]
    # select every second boundary node to be fixed
    fixed_dofs = [2 * node + component for node in boundary_nodes[::2] for component in (0, 1)]

    expected = torch.zeros(256)
    expected[::2] = 1
    torch.testing.assert_close(encode_boundary_conditions(fixed_dofs), expected)
