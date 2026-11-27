import torch

from src.inference import infer
from src.model_types import BatchModelOutput, ModelInput


def test_infer_applies_scaling():
    stress = torch.linspace(0, 1, 64).repeat(64, 1) # initial stress field with some direction
    sample = ModelInput(
        stress=stress,
        strain_energy=torch.zeros(64, 64),
        volume_fraction=torch.tensor([0.4]),
        load_position=torch.tensor([1.0, 0.75]), # right side lower 3/4, so infer() will actually rotate AND mirror
        load_vector=torch.tensor([1.0, 0.0]),
        boundary_conditions=torch.zeros(256),
    )

    class DummyModel(torch.nn.Module):
        def forward(self, batch):
            return BatchModelOutput(densities=batch.stresses)

    result = infer(DummyModel(), sample)

    torch.testing.assert_close(result.density, torch.log1p(stress)) # scaling was applied
    torch.testing.assert_close(sample.stress, stress) # orientation and mirroring restored
    
