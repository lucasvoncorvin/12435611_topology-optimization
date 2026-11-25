import pytest
import torch

import src.eval as evaluation
from src.model_types import BatchModelOutput, ModelInput, ModelOutput


def test_compliance():
    # Only the corner node's x displacement is free, so the FEM system is one spring.
    fixed_dofs = range(1, 2 * 65 * 65)
    expected = 6 * (1 - 0.3**2) / (3 - 0.3) # analytical compliance for a single spring with given material properties
    actual = evaluation.compliance(torch.ones(64, 64), [0, 0, 1, 0], fixed_dofs)
    assert actual == pytest.approx(expected)

def test_shape_overlap():
    
    
    predicted = torch.zeros(64, 64)
    target = torch.zeros(64, 64)

    # set just a few pixels to 1 to create partial overlap
    predicted[0, 0] = predicted[0, 1] = 1.0
    target[0, 0] = target[1, 0] = 1.0
    
    assert evaluation.shape_overlap(predicted, target) == pytest.approx(1 / 3)

    # no overlap
    predicted.zero_()
    assert evaluation.shape_overlap(predicted, target) == 0.0

def test_evaluate_model(monkeypatch):
    predicted = torch.zeros(64, 64)
    predicted[0, 0] = predicted[0, 1] = 1.0
    target = torch.zeros(64, 64)
    target[0, 0] = 1.0
    boundary_conditions = torch.zeros(256)
    boundary_conditions[0] = 1.0

    # create dummy model input and dataset
    model_input = ModelInput(
        stress=predicted,
        strain_energy=torch.zeros(64, 64),
        volume_fraction=torch.tensor([0.5]),
        load_position=torch.tensor([1.0, 0.5]),
        load_vector=torch.tensor([0.5, -1.0]),
        boundary_conditions=boundary_conditions,
    )
    dataset = [(model_input, ModelOutput(density=target))]

    class DummyModel(torch.nn.Module):
        def forward(self, batch):
            return BatchModelOutput(densities=batch.stresses)

    loads = []
    # replace compliance with fake function to avoid actual FEM computation
    def fake_compliance(densities, load, fixed_dofs):
        loads.append((tuple(load), tuple(fixed_dofs)))
        return 1.0
    monkeypatch.setattr(evaluation, "compliance", fake_compliance) 
    
    result = evaluation.evaluate_model(DummyModel(), dataset)

    # there should be 4 calls, 2 (binary, continuous) and 2 (predicted, target)
    assert len(loads) == 4
    # all calls should have the same load and fixed dofs as specified in fake_compliance
    assert all(load == (1.0, 0.5, 0.5, 1.0) and fixed == (0, 1) for load, fixed in loads)

    # construct expected result
    assert result == evaluation.ModelEvaluation(SampleEvaluations=[
        evaluation.SampleEvaluation(
            sample_id=0,
            rel_compliance_error_binary=0.0,
            rel_compliance_error_continuous=0.0,
            rel_volume_error=1023 / 1024,
            density_error=0.0,
            shape_overlap=0.5,
        )
    ])
