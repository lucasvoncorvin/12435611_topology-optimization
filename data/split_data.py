

from pathlib import Path
import numpy as np
from .utils import Samples, load_samples, save_samples, LABELED_DATA_DIR

def split_data(samples: Samples, train_ratio: float = 0.8):
    """Split the data into training and validation sets"""
    num_samples = samples.fixed_dofs.shape[0]
    
    train_size = int(num_samples * train_ratio)
    val_size = int((num_samples - train_size) / 2)
    
    np.random.seed(42)  # for reproducibility
    
    # scramble the indices
    indices = np.random.permutation(num_samples)
    samples = Samples(
        fixed_dofs=samples.fixed_dofs[indices],
        loads=samples.loads[indices],
        stresses=samples.stresses[indices],
        strain_energies=samples.strain_energies[indices],
        densities=samples.densities[indices],
        volume_fractions=samples.volume_fractions[indices]
    )
    
    # split the data into train, val, and test sets
    
    train_samples = Samples(
        fixed_dofs=samples.fixed_dofs[:train_size],
        loads=samples.loads[:train_size],
        stresses=samples.stresses[:train_size],
        strain_energies=samples.strain_energies[:train_size],
        densities=samples.densities[:train_size],
        volume_fractions=samples.volume_fractions[:train_size]
    )
    
    val_samples = Samples(
        fixed_dofs=samples.fixed_dofs[train_size:train_size+val_size],
        loads=samples.loads[train_size:train_size+val_size],
        stresses=samples.stresses[train_size:train_size+val_size],
        strain_energies=samples.strain_energies[train_size:train_size+val_size],
        densities=samples.densities[train_size:train_size+val_size],
        volume_fractions=samples.volume_fractions[train_size:train_size+val_size]
    )

    test_samples = Samples(
        fixed_dofs=samples.fixed_dofs[train_size+val_size:],
        loads=samples.loads[train_size+val_size:],
        stresses=samples.stresses[train_size+val_size:],
        strain_energies=samples.strain_energies[train_size+val_size:],
        densities=samples.densities[train_size+val_size:],
        volume_fractions=samples.volume_fractions[train_size+val_size:]
    )
    
    return train_samples, val_samples, test_samples


def main():
    data_dir = LABELED_DATA_DIR
    samples = load_samples(data_dir)
    train_samples, val_samples, test_samples = split_data(samples)
    save_samples(train_samples, Path(data_dir / "train"))
    save_samples(val_samples, Path(data_dir / "val"))
    save_samples(test_samples, Path(data_dir / "test"))
    
if __name__ == "__main__":
    main()
    
