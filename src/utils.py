import matplotlib.pyplot as plt
import numpy as np
import torch

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
        densities.append(density.reshape(8, 8, 8, 8).transpose(0, 2, 1, 3).reshape(64, 64))

    fixed_nodes = np.empty(0, dtype=np.int64)
    if bcs is not None:
        boundary_mask = to_numpy(bcs)
        if (
            boundary_mask.shape != (256,)
            or not np.isfinite(boundary_mask).all()
            or not np.isin(boundary_mask, (0, 1)).all()
        ):
            raise ValueError("bcs must be the dataset's binary boundary mask of shape (256,).")
        nodes = np.arange(65 * 65)
        boundary_nodes = nodes[
            np.isin(nodes // 65, (0, 64)) | np.isin(nodes % 65, (0, 64))
        ]
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
