from io import BytesIO
from unittest.mock import patch

import numpy as np
import torch

from src.dataset import TODataset, Mode

from pytest import fixture

@fixture
def mode():
    return Mode.TEST

@fixture
def data_bytes():
    n_samples = 10  # Ten samples leave one sample in the validation split.
    shape = (n_samples, 64, 64)
    stresses = np.arange(n_samples * 64 * 64, dtype=np.float32).reshape(shape)
    strain_energies = 2 * stresses + 1
    density_grid = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
    densities = np.broadcast_to(density_grid, shape)

    buffer = BytesIO()
    np.savez(
        buffer,
        stresses=stresses,
        strain_energy_densities=strain_energies,
        densities=densities,
        volume_fractions=np.full(n_samples, 0.5, dtype=np.float32),
        loads=np.tile([64 * 65 + 48, 0.0, -1.0], (n_samples, 1)),  # Right edge at y=48.
        fixed_dofs_flat=np.tile([0, 1], n_samples),  # Both displacements at the top left node.
        fixed_dofs_offsets=2 * np.arange(n_samples + 1),
    )
    return buffer.getvalue()


# test input gets scaled on log1p
def test_input_scaled_on_log1p(data_bytes, mode):
    dataset = TODataset(
        npz_path=BytesIO(data_bytes),
        mode=mode,
        input_scaling="log1p",
        orientation="none",
    )
    assert len(dataset) == 1
    input_data, _ = dataset[0]

    with np.load(BytesIO(data_bytes)) as source:
        # ! AI helped me here with the np.testing assertion
        np.testing.assert_allclose(input_data.stress.numpy(), np.log1p(source["stresses"][-1]))
        np.testing.assert_allclose(
            input_data.strain_energy.numpy(), np.log1p(source["strain_energy_densities"][-1])
        )

# test input does not get scaled when not log1p
def test_input_not_scaled_when_not_log1p(data_bytes, mode):
    dataset = TODataset(
        npz_path=BytesIO(data_bytes),
        mode=mode,
        input_scaling="raw",
        orientation="none",
    )
    assert len(dataset) == 1
    input_data, _ = dataset[0]

    with np.load(BytesIO(data_bytes)) as source:
        # ! AI helped me here with the np.testing assertion
        np.testing.assert_array_equal(input_data.stress.numpy(), source["stresses"][-1])
        np.testing.assert_array_equal(input_data.strain_energy.numpy(), source["strain_energy_densities"][-1])

def test_input_gets_normalized_for_normalized_orientation(data_bytes, mode):
    dataset = TODataset(BytesIO(data_bytes), mode, input_scaling="raw", orientation="normalized")
    sample, _ = dataset[0]

    torch.testing.assert_close(sample.load_position, torch.tensor([0.25, 0.0]))
    with np.load(BytesIO(data_bytes)) as source:
        expected = source["stresses"][-1].T[::-1, ::-1]
        np.testing.assert_array_equal(sample.stress.numpy(), expected)

def test_input_gets_augmented_in_training(data_bytes):
    dataset = TODataset(BytesIO(data_bytes), Mode.TRAIN, input_scaling="raw", orientation="augment")

    # Two quarter turns followed by a left-right reflection give a vertical flip.
    with patch("src.dataset.torch.randint", side_effect=[torch.tensor(2), torch.tensor(1)]):
        sample, _ = dataset[0]

    torch.testing.assert_close(sample.load_position, torch.tensor([1.0, 0.25]))
    with np.load(BytesIO(data_bytes)) as source:
        expected = source["stresses"][0][::-1, :]
        np.testing.assert_array_equal(sample.stress.numpy(), expected)
