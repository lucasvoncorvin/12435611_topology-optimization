import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from enum import StrEnum
import math
from .utils import (
    encode_boundary_conditions,
    idx_to_coordinates,
    mirror_sample,
    normalize_sample,
    rotate_sample
)
from .model_types import ModelInput, ModelOutput, Sample
import logging

logger = logging.getLogger(__name__)

DATA_SPLIT = (0.8, 0.1, 0.1)  # train, val, test split ratios


def orientation_for_evaluation(orientation: str) -> str:
    if orientation not in ("normalized", "augment", "none"):
        raise ValueError(f"Unknown orientation mode: {orientation}")
    return "none" if orientation == "augment" else orientation


class Mode(StrEnum):
    TRAIN = "train"
    VALIDATE = "val"
    TEST = "test"

class TODataset(Dataset):
    def __init__(
        self,
        npz_path: Path,
        mode: Mode,
        input_scaling: str = "log1p",
        orientation: str = "normalized",
    ):
        """
        Args:
            npz_path (Path): Path to the NPZ file containing the dataset.
            mode (Mode): Mode of the dataset (train, validate, test).
            input_scaling: Apply log1p to stress and strain energy, or keep them raw.
            orientation: Normalize orientations, augment training samples, or leave them unchanged.
        """
        if input_scaling not in ("log1p", "raw"):
            raise ValueError(f"Unknown input scaling: {input_scaling}")
        if orientation not in ("normalized", "augment", "none"):
            raise ValueError(f"Unknown orientation mode: {orientation}")

        self.input_scaling = input_scaling
        self.orientation = (
            "none" if mode != Mode.TRAIN and orientation == "augment" else orientation
        )
        self.normalization_angles: list[int] = []
        self.normalization_chiralities: list[bool] = []
        
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

            stresses = torch.from_numpy(data["stresses"][start:stop])
            strain_energies = torch.from_numpy(data["strain_energy_densities"][start:stop])
            if input_scaling == "log1p":
                # Compress the highly skewed distributions before passing them to the model.
                stresses = torch.log1p(stresses)
                strain_energies = torch.log1p(strain_energies)
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
            
            if self.orientation == "normalized":
                rotation, mirrored, input, output = normalize_sample(input, output)
            else:
                rotation, mirrored = 0, False
            assert output is not None
            
            self.normalization_angles.append(rotation)
            self.normalization_chiralities.append(mirrored)
            self.samples.append(Sample(input=input, output=output))
            
        logger.info("Dataset initialized successfully")
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.orientation != "augment":
            return sample.input, sample.output

        rotations = int(torch.randint(4, ()).item())
        mirrored = bool(torch.randint(2, ()).item())
        model_input, model_output = rotate_sample(
            sample.input, sample.output, n_rotations=rotations
        )
        if mirrored:
            model_input, model_output = mirror_sample(model_input, model_output)
        assert model_output is not None
        return model_input, model_output