import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from enum import StrEnum
import math
from .utils import boundary_node_indices, idx_to_coordinates, patchify, orientation_angle, rotate_sample, params_vectors
import logging

logger = logging.getLogger(__name__)

LABELED_DATA_DIR_PATH = Path(__file__).parent.parent / "data/labeled_data/"



class Mode(StrEnum):
    TRAIN = "train"
    VALIDATE = "val"
    TEST = "test"

class TODataset(Dataset):
    def __init__(self, dir: Path, patch_size: int):
        
        # make sure patch size divides 64 evenly
        assert 64 % patch_size == 0, "Patch size must divide 64 evenly"
        
        logger.info(f"Initializing Dataset with path {dir}")
        
        densities = torch.as_tensor(np.load(dir / f"density.npy"))
        fixed_dofs = np.load(dir / f"fixed_dof.npy", allow_pickle=True)
        loads = torch.as_tensor(np.load(dir / f"load.npy"))
        # strain_energy_densities = torch.as_tensor(np.load(training_dir / f"strain_energy_density.npy"))
        stresses = torch.as_tensor(np.load(dir / f"stress.npy"))
        volume_fractions = torch.as_tensor(np.load(dir / f"volume_fraction.npy"))

        
        self.n_samples = densities.shape[0]
        self.angles = torch.zeros(self.n_samples, dtype=torch.int)
        self.patch_size = patch_size
        n_elements = 64*64
        n_patches = (64//self.patch_size) * (64//self.patch_size)
        n_elements_per_patch = self.patch_size**2
        
        logger.info(f"Dataset: n_samples {self.n_samples}")
        
        
        # organize the data into samples
        params = torch.stack([
            params_vectors(loads[i], fixed_dofs[i], volume_fractions[i]) 
            for i in range(self.n_samples)
        ]) 
        
        # params are now
        # [volume_fraction, normalized_load_coordinates, load_direction, one-hot encoded fixed DOFs]
        
        # reorient the samples to a standard orientation based on load position
        oriented_samples = torch.zeros(self.n_samples, params.shape[1] + stresses.shape[1]*stresses.shape[2])
        for i in range(self.n_samples):
            angle = orientation_angle(params[i][1:3])
            self.angles[i] = angle
            sample = torch.cat([params[i], stresses[i].flatten()])
            for _ in range(angle):
                sample = rotate_sample(sample)
            oriented_samples[i] = sample

        
        # below we restructure the data, s.t. each sample is represented as a flattened vector of patches
        # of the stress, strain and density distributions
        self.y = torch.zeros(self.n_samples, n_patches*n_elements_per_patch) # these are the densities
        for sample_idx in range(self.n_samples):
            n_params = params.shape[1]
            oriented_samples[sample_idx, n_params:] = patchify(
                oriented_samples[sample_idx, n_params:].reshape(64, 64), self.patch_size
            )
            # strain_energy_patches[sample_idx] = patchify(strain_energy_densities[sample_idx], self.patch_size)
            self.y[sample_idx] = patchify(
                densities[sample_idx]
                .reshape(64, 64)
                .rot90(k=self.angles[sample_idx], dims=(0, 1)), 
                self.patch_size
            ).flatten()
        
        self.x = oriented_samples
        
        logger.info("Dataset initialized succesfully")
        

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]