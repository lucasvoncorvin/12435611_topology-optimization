import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from enum import StrEnum
import math
from .utils import (
    encode_boundary_conditions,
    idx_to_coordinates,
    normalize_sample
)
from .model_types import ModelInput, ModelOutput, Sample
import logging

logger = logging.getLogger(__name__)

DATA_SPLIT = (0.8, 0.1, 0.1)  # train, val, test split ratios

class Mode(StrEnum):
    TRAIN = "train"
    VALIDATE = "val"
    TEST = "test"

class TODataset(Dataset):
    def __init__(self, npz_path: Path,  mode: Mode):
        """
        Args:
            npz_path (Path): Path to the NPZ file containing the dataset.
            mode (Mode): Mode of the dataset (train, validate, test).
        """
        self.normalization_angle = 0
        self.normalization_chirality = 0
        
        logger.info(f"Initializing Dataset with path {npz_path} in mode {mode}")
        
        logger.info("Reading NPZ data")
        with np.load(npz_path) as data:
            logger.info(f"Data loaded:\n{data.files}")
            
            # since data in the csv is randomly ordered we just can split it by counting from the top
            loads = data["loads"]
            n_total = len(loads)
            n_train = int(n_total * DATA_SPLIT[0])
            n_val = int(n_total * DATA_SPLIT[1])
            n_test = n_total - n_train - n_val

            if mode == Mode.TRAIN:
                logger.info(f"Loading {n_train} samples for training")
                start, stop = 0, n_train
            elif mode == Mode.VALIDATE:
                logger.info(f"Loading {n_val} samples for validation")
                start, stop = n_train, n_train + n_val
            elif mode == Mode.TEST:
                logger.info(f"Loading {n_test} samples for testing")
                start, stop = n_train + n_val, n_total
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # stress distribution has a very high peak and lots of low values, whereas the result will have only values between 0 and 1
            # To avoid floating point errors, the stress values are normalized before being used as input
            stresses = torch.log1p(torch.from_numpy(data["stresses"][start:stop]))
            
            # same for strain energy density
            strain_energies = torch.log1p(torch.from_numpy(data["strain_energy_densities"][start:stop]))
            densities = torch.from_numpy(data["densities"][start:stop].copy())
            volume_fractions = data["volume_fractions"][start:stop]
            loads = loads[start:stop]
            fixed_dofs_flat = data["fixed_dofs_flat"]
            offsets = data["fixed_dofs_offsets"][start:stop + 1]

        self.samples = []
        for i, load in enumerate(loads):
            
            load_position = torch.tensor(idx_to_coordinates(int(load[0]), normalize=True)) # divide by 64 to normalize coordinates to [0, 1] range
            load_direction = torch.tensor([load[1], -load[2]]) # topoptlab fy points up, flip to image convention (y down)
            fixed_dofs = fixed_dofs_flat[offsets[i]:offsets[i + 1]]
            
            boundary_conditions = encode_boundary_conditions(fixed_dofs)

            
            input = ModelInput(
                stress=stresses[i],
                strain_energy=strain_energies[i],
                volume_fraction=torch.tensor([float(volume_fractions[i])]),
                load_position=load_position,
                load_vector=load_direction,
                boundary_conditions=boundary_conditions
            )
            output = ModelOutput(
                density=densities[i]
            )
            
            self.normalization_angle, self.normalization_chirality, input, output = normalize_sample(input, output)
            assert output is not None
            
            self.samples.append(Sample(input=input, output=output))
            
        logger.info("Dataset initialized successfully")
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx].input, self.samples[idx].output