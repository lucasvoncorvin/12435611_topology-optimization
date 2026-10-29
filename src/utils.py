
import matplotlib.pyplot as plt
import numpy as np
import torch

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
    
    x = idx // 65 # 65 nodes per row
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

def reshape_distribution(distribution):
    """Reshape a flattened (64*64,) distribution into a (64, 64) image indexed by [y, x]."""
    distribution = np.asarray(distribution)
    if distribution.size == 64 * 64 and distribution.shape == (64 * 64,):
        return distribution.reshape(64, 64)
    if distribution.shape == (64, 64):
        return distribution.flatten()
    raise ValueError("distribution must be either a flattened (64*64,) array or a (64, 64) array.")

#! this function was created by ai to help me visualize the results
def visualize(true_density, predicted_density, load=None, bcs=None, volume_fraction=None):
    """Compare true and predicted densities and return (figure, axes).

    Both densities must be patch-ordered vectors of shape (4096,) or (1, 4096).
    The 8-by-8 patches and the values within each patch are in row-major order,
    matching the dataset and model outputs. axes contains the true and
    predicted panels, respectively.
    load is [x_normalized, y_normalized, fx, fy], with coordinates in [0, 1].
    bcs is the dataset's 256-entry binary boundary mask, ordered by increasing
    node index along the perimeter; 1 fixes both displacement components.
    volume_fraction is an optional scalar target in [0, 1]. Panel titles show
    the actual mean density; the figure title shows the target when supplied.
    Lists, NumPy arrays, and tensors are accepted.
    """
    def to_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    densities = []
    for name, value in (("true_density", true_density), ("predicted_density", predicted_density)):
        density = to_numpy(value)
        if density.shape not in ((4096,), (1, 4096)) or not np.isfinite(density).all():
            raise ValueError(f"{name} must be a finite patch-ordered vector of shape (4096,) or (1, 4096).")
        density = density.reshape(4096)
        densities.append(unpatchify(torch.as_tensor(density), patch_size=8).numpy())

    fixed_nodes = np.empty(0, dtype=np.int64)
    if bcs is not None:
        boundary_mask = to_numpy(bcs)
        if (
            boundary_mask.shape != (256,)
            or not np.isfinite(boundary_mask).all()
            or not np.isin(boundary_mask, (0, 1)).all()
        ):
            raise ValueError("bcs must be the dataset's binary boundary mask of shape (256,).")
        boundary_nodes = np.asarray(boundary_node_indices())
        fixed_nodes = boundary_nodes[boundary_mask == 1]

    if load is not None:
        load = to_numpy(load)
        if (
            load.shape != (4,)
            or not np.isfinite(load).all()
            or np.any((load[:2] < 0) | (load[:2] > 1))
        ):
            raise ValueError("load must be [x_normalized, y_normalized, fx, fy] with coordinates in [0, 1].")

    if volume_fraction is not None:
        volume_fraction = to_numpy(volume_fraction)
        if (
            volume_fraction.shape != ()
            or not np.isfinite(volume_fraction)
            or not 0 <= volume_fraction <= 1
        ):
            raise ValueError("volume_fraction must be a finite scalar in [0, 1].")
        volume_fraction = volume_fraction.item()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True, layout="constrained")
    if volume_fraction is not None:
        fig.suptitle(f"Target volume fraction: {volume_fraction:.3f}")
    for ax, density, title in zip(axes, densities, ("True density", "Predicted density")):
        image = ax.imshow(
            density, cmap="gray_r", vmin=0, vmax=1,
            origin="upper", extent=(0, 64, 64, 0),
        )

        if fixed_nodes.size:
            ax.scatter(
                fixed_nodes // 65, fixed_nodes % 65, marker="s", s=35,
                color="tab:blue", label="Fixed nodes (x and y)", zorder=3, clip_on=False,
            )

        margin = 2
        if load is not None:
            x_normalized, y_normalized, fx, fy = load
            x, y = 64 * x_normalized, 64 * y_normalized
            ax.scatter([x], [y], color="tab:red", label="Load", zorder=4)
            magnitude = np.hypot(fx, fy)
            if magnitude > 0:
                arrow_length = 12.8
                ax.annotate(
                    "", xy=(x, y),
                    xytext=(x - arrow_length * fx / magnitude, y - arrow_length * fy / magnitude),
                    arrowprops=dict(arrowstyle="-|>", color="tab:red", lw=2),
                    zorder=4,
                )
                margin += arrow_length

        ax.set_xlim(-margin, 64 + margin)
        ax.set_ylim(64 + margin, -margin)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{title}\nVolume fraction: {density.mean():.3f}")
        if fixed_nodes.size or load is not None:
            ax.legend()
    fig.colorbar(image, ax=axes.tolist(), label="Density")
    plt.show()
    return fig, axes


def patchify(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Flatten images into consecutive patches, preserving leading dimensions."""
    if image.shape[-2:] != (64, 64):
        raise ValueError("Expected image dimensions (64, 64)")
    if patch_size <= 0 or 64 % patch_size != 0:
        raise ValueError("Patch size must be positive and divide 64")

    n = 64 // patch_size
    leading_shape = image.shape[:-2]

    return ( #! ai helped me a bit here with the transpose
        image.reshape(*leading_shape, n, patch_size, n, patch_size)
        .transpose(-3, -2)
        .flatten(start_dim=-4)
    )
                    

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


def rotate_sample(sample: torch.Tensor) -> torch.Tensor:
    """Rotate a flat sample 90 degrees counterclockwise.

    The sample contains volume fraction, load position and direction, 256
    boundary-condition flags, and a flattened 64x64 spatial distribution.
    """
    _validate_sample(sample)

    rotated_sample = sample.clone()

    load_coordinates = sample[1:3] # normalized coordinates of the load application point
    load_direction = sample[3:5] # components of the load vector
    rotated_sample[1] = load_coordinates[1]
    rotated_sample[2] = 1 - load_coordinates[0]
    rotated_sample[3] = -load_direction[1]
    rotated_sample[4] = load_direction[0]


    # Boundary flags follow the parameter vector's node-index order, not perimeter order.
    fixed_nodes = sample[5:5 + 4 * 64]
    boundary_nodes = boundary_node_indices()
    boundary_positions = {node: index for index, node in enumerate(boundary_nodes)}
    destination_indices = []
    for node in boundary_nodes:
        x, y = divmod(node, 65)
        rotated_node = 65 * y + (64 - x)
        destination_indices.append(boundary_positions[rotated_node])
    rotated_fixed_nodes = torch.empty_like(fixed_nodes)
    destination_indices = torch.tensor(destination_indices, device=sample.device)
    rotated_fixed_nodes[destination_indices] = fixed_nodes
    rotated_sample[5:5 + 4 * 64] = rotated_fixed_nodes

    
    # rotate the image
    image = sample[5+4*64:].reshape(64, 64) # unflatten so we can use the rot90 function
    image = torch.rot90(image, k=1, dims=(0, 1))
    rotated_sample[5+4*64:] = image.flatten()
    
    return rotated_sample
    
def params_vectors(load, fixed_dofs, volume_fraction):
    """Returns a representation of the load, the boundary conditions and the volume fraction
    with length 1 + 4 + 4*64 = 261 (volume fraction + load representation + boundary conditions)
    """
    fixed_dofs_set = set(fixed_dofs)
    for dof in fixed_dofs_set:
        # just rechecking the data
        partner = dof + 1 if dof % 2 == 0 else dof - 1
        assert partner in fixed_dofs_set, "DOF pairs must be consecutive"
    assert 0 <= volume_fraction <= 1, "Volume fraction must be between 0 and 1"
    
    # first two elements of the load representation are the normalized coordinates of the load application point
    loads = torch.zeros(4)
    loads[0], loads[1] = idx_to_coordinates(int(load[0]), normalize=True)
    
    # second two elements are the load vector
    loads[2] = load[1]
    loads[3] = load[2]
    
    # Remaining elements encode fixed nodes in ascending node-index order.
    # 1 means both displacement components are fixed; 0 means free.
    
    bc = torch.zeros(4*64) # number of boundary nodes is 4*64
    boundary_nodes = boundary_node_indices()
    assert len(boundary_nodes) == len(bc)
    fixed_dofs_set = set(fixed_dofs)
    for i, node in enumerate(boundary_nodes):
        if node*2 in fixed_dofs_set:
            assert node*2+1 in fixed_dofs_set, "DOF pairs must be consecutive"
            bc[i] = 1
    

    return torch.cat([torch.tensor([volume_fraction]), loads, bc])


def orientation_angle(normalized_load_coordinates) -> int:
    """Returns the angle of rotation needed to orient the image based on the load position.
    (This can be generalized for multiple loads, by taking the average weighted position of the loads)

    Returns the angle of rotation applied (in steps of pi/2).
    """
    # _validate_sample(sample)
    # determine the angle based on the load position
    if normalized_load_coordinates[0] == 1.0:  # right side
        angle = 1
    elif normalized_load_coordinates[1] == 1.0:  # bottom side
        angle = 2
    elif normalized_load_coordinates[0] == 0.0:  # left side
        angle = 3
    else:
        angle = 0
    
    return angle        

def preprocess(volume_fraction, load, fixed_dofs, stress, patch_size):
    """returns a sample for the model"""
    params = params_vectors(load, fixed_dofs, volume_fraction)
    angle = orientation_angle(params[1:3])
    sample = torch.cat([params, stress.flatten()])
    for _ in range(angle):
        sample = rotate_sample(sample)
    sample[5+4*64:] = patchify(sample[5+4*64:].reshape(64, 64), patch_size).flatten()

    return sample
