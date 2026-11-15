
from dataclasses import dataclass
from unicodedata import mirrored

import matplotlib.pyplot as plt
import numpy as np
import torch



    

@dataclass
class ModelInput:
    stress: torch.Tensor # (64,64)
    strain_energy: torch.Tensor # (64,64)
    volume_fraction: torch.Tensor # (1,)
    load_position: torch.Tensor # (2,) normalized coordinates [x, y] within the grid
    load_vector: torch.Tensor # (2,) normalized vector [fx, fy] within the grid
    boundary_conditions: torch.Tensor # (256,) one-hot encoded boundary mask, 1 fixes both displacement components
    
@dataclass
class ModelOutput:
    density: torch.Tensor # (64,64)
    
@dataclass
class Sample:
    input: ModelInput
    output: ModelOutput

@dataclass
class BatchModelInput:
    stresses: torch.Tensor # (batch_size, 64, 64)
    strain_energies: torch.Tensor # (batch_size, 64, 64)
    volume_fractions: torch.Tensor # (batch_size, 1)
    load_positions: torch.Tensor # (batch_size, 2)
    load_vectors: torch.Tensor # (batch_size, 2)
    boundary_conditions: torch.Tensor # (batch_size, 256)
    
    def to(self, device: torch.device): # Move all tensors to the specified device
        self.stresses = self.stresses.to(device)
        self.strain_energies = self.strain_energies.to(device)
        self.volume_fractions = self.volume_fractions.to(device)
        self.load_positions = self.load_positions.to(device)
        self.load_vectors = self.load_vectors.to(device)
        self.boundary_conditions = self.boundary_conditions.to(device)
        return self
    
@dataclass
class BatchModelOutput:
    densities: torch.Tensor # (batch_size, 64, 64)

    def to(self, device: torch.device): # Move all tensors to the specified device
        self.densities = self.densities.to(device)
        return self

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

def normalize_sample_chirality(model_input: ModelInput, model_output: ModelOutput | None = None) -> tuple[bool, ModelInput, ModelOutput | None]:
    """Normalize the chirality (mirror symmetry) of the sample's input tensors."""
    mirrored = False
    
    if model_input.load_position[0] > 0.5: 
        return True, *mirror_sample(model_input, model_output)
    else:
        return False, model_input, model_output

# TODO: review this
def normalize_sample(model_input: ModelInput, model_output: ModelOutput | None = None) -> tuple[int, bool, ModelInput, ModelOutput | None]:
    # todo: write tests for this function
    """Normalize both the rotation and chirality of the sample."""
    n_rotations, model_input, model_output = normalize_sample_rotation(model_input, model_output)
    mirrored, model_input, model_output = normalize_sample_chirality(model_input, model_output)
    return n_rotations, mirrored, model_input, model_output

def encode_boundary_conditions(fixed_dofs):
    # todo: write tests for this function
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
    

def patchify(image: torch.Tensor, patch_size: int, overlap: int = 0) -> torch.Tensor:
    # todo: write test for this function
    """Converts image of shape (..., 64, 64) into consecutive patches of shape (..., n_patches, patch_size, patch_size)
    
    if overlap != 0: the patches will have the specified number of overlapping pixels along each dimension.
    """
    
    if image.shape[-2:] != (64, 64):
        raise ValueError("Expected image dimensions (64, 64)")
    if patch_size <= 0 or (64 % patch_size != 0 and overlap == 0):
        raise ValueError("Patch size must be positive and divide 64")
    if overlap != 0:
        raise NotImplementedError("Overlapping patches are not yet implemented")

    n = 64 // patch_size
    leading_shape = image.shape[:-2]
    return image.reshape(*leading_shape, n, patch_size, n, patch_size).transpose(-3, -2).reshape(*leading_shape, n*n, patch_size, patch_size) # ! ai helped me with the reshape stuff, since I didnt seem to get it right


def unpatchify(flattened_image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Reconstruct 64x64 image(s) from flattened consecutive patches."""
    if patch_size <= 0 or 64 % patch_size != 0:
        raise ValueError("Patch size must be positive and divide 64")
    if flattened_image.ndim == 0 or flattened_image.shape[-1] != 64 * 64:
        raise ValueError("Expected flattened image(s) with 4096 values each")

    n = 64 // patch_size
    leading_shape = flattened_image.shape[:-1]

    return (
        flattened_image.reshape(*leading_shape, n, n, patch_size, patch_size)
        .transpose(-3, -2)
        .reshape(*leading_shape, 64, 64)
    )

def _validate_sample(sample: torch.Tensor) -> None:
    expected_size = 1 + 4 + 4 * 64 + 64 * 64
    if sample.ndim != 1 or sample.numel() != expected_size:
        raise ValueError(f"sample must be a flat tensor with {expected_size} values")
    

    

def preprocess(volume_fraction, load, fixed_dofs, stress, patch_size):
    """returns a sample for the model"""
    params = params_vectors(load, fixed_dofs, volume_fraction)
    angle = orientation_angle(params[1:3])
    sample = torch.cat([params, stress.flatten()])
    for _ in range(angle):
        sample = rotate_sample(sample)
    sample[5+4*64:] = patchify(sample[5+4*64:].reshape(64, 64), patch_size).flatten()

    return sample

def postprocess(prediction: torch.Tensor, angle: int, patch_size: int, load=None, fixed_dofs_one_hot=None, binarify_densities: bool = False):
    """
    Postprocess a sample from the model.

    prediction: the output sample tensor from the model
    patch_size: the size of the patches used during preprocessing
    load: the load vector from rotated sample (optional)
    fixed_dofs_one_hot: one-hot encoding of the fixed degrees of freedom (optional)
    angle: the angle of rotation applied during preprocessing
    binarify_densities: if true, the density values will be rounded to 0 or 1 (as int)

    Returns the postprocessed sample as a tuple containing:
    - the prediction tensor
    - the load vector (if provided, otherwise None)
    - the fixed degrees of freedoms un-one-hot encoded (if provided, otherwise None)
    """
    
    
    # unpatchify
    prediction = unpatchify(prediction, patch_size)
    
    # unrotate    
    if load is None and fixed_dofs_one_hot is None:
        for _ in range(4 - angle):
            prediction = prediction.rot90(1, [0, 1])
    
    if load is not None or fixed_dofs_one_hot is not None:
        sample = torch.cat([torch.tensor([0.0]), load, fixed_dofs_one_hot, prediction.flatten()])
        for _ in range(4 - angle):
            sample = rotate_sample(sample)
        
        prediction = sample[5+4*64:].reshape(64, 64)
        
        if load is not None:
            load = sample[1:5]
        if fixed_dofs_one_hot is not None:
            fixed_dofs_one_hot = sample[5:5+4*64]
            # un-one-hot-encode the fixed degrees of freedom to return them to their original form
            fixed_dofs = un_one_hot_encode(fixed_dofs_one_hot).detach().cpu().numpy()
            
        

        
    # binarify if requested
    if binarify_densities:
        prediction = (prediction > 0.5).int()
        
    res = (prediction, load, fixed_dofs if fixed_dofs_one_hot is not None else None)
    
    return res
