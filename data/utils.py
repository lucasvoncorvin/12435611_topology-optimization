import numpy as np
from dataclasses import dataclass
from pathlib import Path

LABELED_DATA_DIR = Path(__file__).resolve().parent / "labeled_data"

@dataclass
class Samples:
    fixed_dofs: np.ndarray
    loads: np.ndarray
    stresses: np.ndarray
    strain_energies: np.ndarray
    densities: np.ndarray
    volume_fractions: np.ndarray
    
    
    # define addition for combining two Samples objects for convenience in augment_data function
    def __add__(self, other: "Samples") -> "Samples":
        return Samples(
            fixed_dofs=np.concatenate([self.fixed_dofs, other.fixed_dofs], axis=0),
            loads=np.concatenate([self.loads, other.loads], axis=0),
            stresses=np.concatenate([self.stresses, other.stresses], axis=0),
            strain_energies=np.concatenate([self.strain_energies, other.strain_energies], axis=0),
            densities=np.concatenate([self.densities, other.densities], axis=0),
            volume_fractions=np.concatenate([self.volume_fractions, other.volume_fractions], axis=0)
        )


def load_samples(directory: Path) -> Samples:
    """expects naming convention like 'fixed_dof.npy', 'load.npy', 'stress.npy', 'strain_energy_density.npy', 'density.npy', 'volume_fraction.npy'"""
    fixed_dofs = np.load(directory / "fixed_dof.npy", allow_pickle=True)
    loads = np.load(directory / "load.npy")
    stresses = np.load(directory / "stress.npy")
    strain_energies = np.load(directory / "strain_energy_density.npy")
    densities = np.load(directory / "density.npy")
    volume_fractions = np.load(directory / "volume_fraction.npy")

    return Samples(
        fixed_dofs=fixed_dofs,
        loads=loads,
        stresses=stresses,
        strain_energies=strain_energies,
        densities=densities,
        volume_fractions=volume_fractions
    )
    
def save_samples(samples: Samples, path: Path):
    """Save the samples to the specified path"""
    
    assert path.resolve() != LABELED_DATA_DIR # prevent overwriting the original labeled data
    
    # safe in separate npy files for each attribute
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "fixed_dof.npy", samples.fixed_dofs)
    np.save(path / "load.npy", samples.loads)
    np.save(path / "stress.npy", samples.stresses)
    np.save(path / "strain_energy_density.npy", samples.strain_energies)
    np.save(path / "density.npy", samples.densities)
    np.save(path / "volume_fraction.npy", samples.volume_fractions)