import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from enum import StrEnum

import logging

logger = logging.getLogger(__name__)

LABELED_DATA_DIR_PATH = Path(__file__).parent.parent / "data/labeled_data/"

def _idx_to_coordinates(idx, normalize=False):
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
    loads[0], loads[1] = _idx_to_coordinates(int(load[0]), normalize=True)
    
    # second two elements are the load vector
    loads[2] = load[1]
    loads[3] = load[2]
    
    # remaining elements give the fixed nodes by one-hot encoding counter clockwise starting from the origin (upper-left corner)
    # 1: fixed in both dimensions, 0: free
    
    bc = torch.zeros(4*64) # number of boundary nodes is 4*64
    boundary_nodes = [
        node
        for node in range(65*65)
        if _idx_to_coordinates(node, normalize=False)[0] in [0, 64] or _idx_to_coordinates(node, normalize=False)[1] in [0, 64]
    ]
    assert len(boundary_nodes) == len(bc)
    fixed_dofs_set = set(fixed_dofs)
    for i, node in enumerate(boundary_nodes):
        if node*2 in fixed_dofs_set:
            assert node*2+1 in fixed_dofs_set, "DOF pairs must be consecutive"
            bc[i] = 1
    

    return torch.cat([torch.tensor([volume_fraction]), loads, bc])


class Mode(StrEnum):
    TRAIN = "train"
    VALIDATE = "val"
    TEST = "test"

class TODataset(Dataset):
    def __init__(self, mode: Mode):
        
        logger.info(f"Initializing Dataset with mode {mode}")
        
        
        dir = Path(LABELED_DATA_DIR_PATH / mode)
        
        densities = torch.as_tensor(np.load(dir / f"density_{mode.value}.npy"))
        fixed_dofs = np.load(dir / f"fixed_dof_{mode.value}.npy", allow_pickle=True)
        loads = torch.as_tensor(np.load(dir / f"load_{mode.value}.npy"))
        # strain_energy_densities = torch.as_tensor(np.load(training_dir / f"strain_energy_density_{mode.value}.npy"))
        stresses = torch.as_tensor(np.load(dir / f"stress_{mode.value}.npy"))
        volume_fractions = torch.as_tensor(np.load(dir / f"volume_fraction_{mode.value}.npy"))

        
        self.n_samples = densities.shape[0]
        n_elements = 64*64
        n_patches = 8*8
        n_elements_per_patch = 8*8
        
        logger.info(f"Dataset: n_samples {self.n_samples}")
        
        
        # organize the data into samples
        params = torch.stack([
            params_vectors(loads[i], fixed_dofs[i], volume_fractions[i]) 
            for i in range(self.n_samples)
        ]) 
        
        
        
        # below we restructure the data, s.t. each sample is represented as a flattened vector of patches
        # of the stress, strain and density distributions
        self.y = torch.zeros(self.n_samples, n_patches*n_elements_per_patch) # these are the densities
        stress_samples = torch.zeros(self.n_samples, n_patches*n_elements_per_patch) # all samples flattened and concatenated
        for sample_idx in range(self.n_samples):
            stress_sample = stresses[sample_idx]
            # strain_energy_sample = strain_energy_densities[sample_idx]
            
            # reorganize the distributions into patches for each sample
            stress_patches = torch.zeros(n_patches*n_elements_per_patch) # all patches flattened and concatenated
            # strain_energy_patches = torch.zeros(n_patches*n_elements_per_patch) # all patches flattened and concatenated           
            counter = 0
            for row in range(8):
                for col in range(8):
                    # create patch, flatten and store it as the corresponding stress samples
                    stress_samples[sample_idx, counter*n_elements_per_patch:(counter+1)*n_elements_per_patch] = stress_sample[8*row:8*(row+1), 8*col:8*(col+1)].flatten()
                    # strain_energy_patches[sample_idx, counter*n_elements_per_patch:(counter+1)*n_elements_per_patch] = strain_energy_densities[8*row:8*(row+1), 8*col:8*(col+1)].flatten()
                    
                    # densities
                    self.y[sample_idx, counter*n_elements_per_patch:(counter+1)*n_elements_per_patch] = densities[sample_idx, 8*row:8*(row+1), 8*col:8*(col+1)].flatten()
                    
                    counter += 1
                    
        
        # concatenate params with distributions, s.t. each feature vector contains both
        # the first 1+4+4*64 = 261 elements correspond to the params, the rest to the stress samples of size n_patches*n_elements_per_patch
        self.x = torch.cat([params, stress_samples], dim=1)
        
        logger.info("Dataset initialized succesfully")
        

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]