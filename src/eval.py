"""Create evaluations for the results of the vision transformer model"""

from dataclasses import asdict, dataclass, fields
import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse.linalg import spsolve
from topoptlab.elements.bilinear_quadrilateral import create_edofMat
from topoptlab.elements.linear_elasticity_2d import lk_linear_elast_2d
from topoptlab.fem import assemble_matrix, create_matrixinds
from matplotlib import pyplot as plt

from src.dataset import TODataset, Mode, orientation_for_evaluation
from src.model_types import Checkpoint
from src.utils import boundary_node_indices, collate_samples


logger = logging.getLogger(__name__)

@dataclass
class SampleEvaluation:
    sample_id: int
    rel_compliance_error_binary: float # all values either 1 or 0
    rel_compliance_error_continuous: float # values in interval [0, 1] using penalization for intermediate values (compliance function in SIMP)
    rel_volume_error: float # error in the volume fraction of the topology
    density_error: float # measure for non-binary densities (values not equal to 0 or 1)
    
@dataclass
class ModelEvaluation:
    SampleEvaluations: list[SampleEvaluation]

    def __post_init__(self):
        if not self.SampleEvaluations:
            raise ValueError("An evaluation needs at least one sample")

        metric_names = [field.name for field in fields(SampleEvaluation) if field.name != "sample_id"]
        metric_values = [
            getattr(sample, metric_name)
            for sample in self.SampleEvaluations
            for metric_name in metric_names
        ]
        if not np.isfinite(metric_values).all():
            raise ValueError("Sample metrics must all be finite")

    def to_dataframe(self) -> pd.DataFrame:
        """Return per-sample metrics indexed by sample ID."""
        return pd.DataFrame(
            asdict(sample) for sample in self.SampleEvaluations
        ).set_index("sample_id")

    def distribution(self, metric: str) -> pd.Series:
        """Return every observed value for a metric, indexed by sample ID."""
        if metric not in {field.name for field in fields(SampleEvaluation)} - {"sample_id"}:
            raise ValueError(f"Unknown evaluation metric: {metric}")
        return self.to_dataframe()[metric]

    def summary(self) -> pd.DataFrame:
        """Summarize each metric, including selected empirical percentiles."""
        return self.to_dataframe().describe(
            percentiles=[0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        )

    def fraction_above(self, metric: str, threshold: float) -> float:
        """Return the fraction of samples whose metric exceeds a threshold."""
        values = self.distribution(metric)
        if not np.isfinite(threshold):
            raise ValueError("threshold must be finite")
        return float((values > threshold).mean())
    
    def save(self, path: Path):
        """Save the model evaluation to a file"""
        with open(path, "w") as f:
            json.dump(asdict(self), f)
            
    @staticmethod
    def load(path: Path):
        """Load the model evaluation from a file"""
        with open(path, "r") as f:
            data = json.load(f)
        sample_evaluations = [SampleEvaluation(**se) for se in data["SampleEvaluations"]]
        return ModelEvaluation(SampleEvaluations=sample_evaluations)
    
    


# todo: create a test for this
def compliance(densities, load, fixed_dofs):
    """Compute the compliance of a given topology"""

    #! this function was made in part by ai, to help me with all the non-deep-learning related maths / FEM
    
    # convert to numpy arrays and check inputs first
    
    if hasattr(densities, "detach"):
        densities = densities.detach().cpu().numpy()
    densities = np.asarray(densities, dtype=float)
    if densities.shape != (64, 64):
        raise ValueError("densities must have shape (64, 64)")

    if hasattr(load, "detach"):
        load = load.detach().cpu().numpy()
    load = np.asarray(load, dtype=float).reshape(-1)
    if load.shape != (4,) or not np.isfinite(load).all():
        raise ValueError("load must contain finite [x_normalized, y_normalized, fx, fy] values")
    if np.any((load[:2] < 0) | (load[:2] > 1)):
        raise ValueError("normalized load coordinates must be between 0 and 1")
    load_x, load_y = (int(round(coordinate * 64)) for coordinate in load[:2])
    load_node = load_x * 65 + load_y


    fixed_dofs = np.asarray(fixed_dofs, dtype=float).reshape(-1)
    if not np.isfinite(fixed_dofs).all() or np.any(fixed_dofs != np.floor(fixed_dofs)):
        raise ValueError("fixed_dofs must contain integer degree-of-freedom indices")
    fixed_dofs = fixed_dofs.astype(int)
    
    # check that the load node and fixed degrees of freedom are within the mesh boundaries
    nelx = nely = 64
    ndof = 2 * (nelx + 1) * (nely + 1)
    if load_node < 0 or load_node >= (nelx + 1) * (nely + 1):
        raise ValueError("load node index is outside the mesh")
    if np.any((fixed_dofs < 0) | (fixed_dofs >= ndof)):
        raise ValueError("fixed_dofs contains an index outside the mesh")

    force = np.zeros(ndof, dtype=float)
    force[2 * load_node:2 * load_node + 2] = load[2:]
    free_dofs = np.setdiff1d(np.arange(ndof), fixed_dofs)
    if free_dofs.size == 0 or not np.any(force[free_dofs]):
        raise ValueError("the load must act on at least one free degree of freedom")


    # assemble the global stiffness matrix for the finite element modelq
    
    element_matrix = lk_linear_elast_2d(
        E=1.0, nu=0.3, plane_stress=True, l=np.array([1.0, 1.0])
    )
    edof_mat, _, _, _, _ = create_edofMat(nelx=nelx, nely=nely, nnode_dof=2)
    i_k, j_k = create_matrixinds(edofMat=edof_mat, mode="full")

    element_densities = densities.T.reshape(-1)
    stiffness_scale = 1e-9 + (1.0 - 1e-9) * element_densities**3
    element_stiffness = element_matrix[None, :, :] * stiffness_scale[:, None, None]
    global_stiffness = assemble_matrix(
        sK=element_stiffness.reshape(-1),
        iK=i_k,
        jK=j_k,
        ndof=ndof,
        solver="scipy-direct",
        springs=None,
    )

    # solve the linear system for the free degrees of freedom to obtain displacements
    reduced_stiffness = global_stiffness[free_dofs, :][:, free_dofs]
    displacement = spsolve(reduced_stiffness.tocsc(), force[free_dofs])
    if not np.isfinite(displacement).all():
        raise np.linalg.LinAlgError("FEM solve produced non-finite displacements")

    return float(force[free_dofs] @ displacement) # compliance  = force^T * displacement

# define evaluation functions for the vision transformer model
def rel_compliance_error(predicted, target, load, fixed_dofs, binarify_densities=False):
    """Compute the error in compliance between the predicted and target topologies
    
    predicted: predicted topology as a 64x64 tensor of material densities of 0 or 1
    target: target topology as a 64x64 tensor of material densities of 0 or 1
    
    Returns the absolut relative compliance error as a scalar value (abs(c_predicted-c_target) / c_target)
    """

    if binarify_densities:
        predicted = predicted.round()
        target = target.round()
    
    c_predicted = compliance(densities=predicted, load=load, fixed_dofs=fixed_dofs)
    c_target = compliance(densities=target, load=load, fixed_dofs=fixed_dofs)
    
    return min(abs(c_predicted - c_target) / c_target, 1) # max relative compliance error capped at 1

def density_error(predicted):
    """
    predicted: predicted topology as a 64x64 tensor of (non-binarified) material densities between 0 or 1

    Returns an error indicating how far the predicted densities deviate from valid binary values (0 or 1).
    by comparing the predicted densities to their rounded (binary) values.
    """
    
    return torch.abs(predicted - predicted.round()).mean()


def floating_material_error(predicted, target):
    # todo: how do we define floating material error?
    pass

def rel_volume_error(predicted, target_volume_fraction):
    """Compute the absolute error in material volume fraction between the predicted and target topologies"""
    
    predicted_volume = (predicted.sum() / predicted.numel()).item()
    
    return abs(predicted_volume - target_volume_fraction) / target_volume_fraction # relative volume error

def evaluate_prediction(predicted, target, load, fixed_dofs, volume_fraction):
    """Evaluate a predicted topology against the target topology using multiple error metrics.
    
    
    Args:
        predicted: predicted topology as a 64x64 tensor of material densities between 0 and 1
        target: target topology as a 64x64 tensor of material densities between 0 and 1
        

    Returns:
        A dictionary containing various error metrics for the predicted topology.
    """
    
    # the purpose of this model is to accelerat topology optimization
    # to evaluate the results we have to evaluate
    # - physicality: densities other then 0 or 1 should be minimized
    #   -> we measure density and floating material
    # - function: (as in inverse of stiffness). Since the goal of this type of topology optimization is to 
    #   minimize compliance for a given material volume, the predicted topology should have a compliance close to the target compliance.
    #   -> we measure compliance error
    # - volume: the predicted topology should respect the target material volume.
    #   -> we measure volume error
    
    return {
        "rel_compliance_error_binary": rel_compliance_error(predicted, target, load, fixed_dofs, binarify_densities=True),
        "rel_compliance_error_continuous": rel_compliance_error(predicted, target, load, fixed_dofs, binarify_densities=False),
        "density_error": float(density_error(predicted)),
        "rel_volume_error": float(rel_volume_error(predicted, volume_fraction)),
    }
    
def evaluate_model(model: torch.nn.Module, dataset: TODataset, sample_ids: list[int] | None = None):
        
    
    model.eval()
    model_tensor = next(model.parameters(), None)
    if model_tensor is None:
        model_tensor = next(model.buffers(), None)
    device = model_tensor.device if model_tensor is not None else torch.device("cpu")

    evaluations = []
    ids_to_evaluate = sample_ids if sample_ids is not None else range(len(dataset))
    with torch.no_grad():
        for i in ids_to_evaluate:
            logger.info(f"Evaluating sample {i}/{len(ids_to_evaluate)}")
            model_input, target_output = dataset[i]
            model_batch, _ = collate_samples([(model_input, target_output)])
            model_batch = model_batch.to(device)
            predicted = model(model_batch).densities.squeeze(0)
            target = target_output.density
            # The dataset transforms the fields and physical parameters together, so evaluate in that frame.
            load = torch.cat((model_input.load_position, model_input.load_vector)).numpy()
            boundary_nodes = np.asarray(boundary_node_indices())
            fixed_nodes = boundary_nodes[model_input.boundary_conditions.bool().numpy()]
            fixed_dofs = np.column_stack((2 * fixed_nodes, 2 * fixed_nodes + 1)).reshape(-1)

            evaluation_tuple = evaluate_prediction(
                predicted,
                target,
                load,
                fixed_dofs,
                model_input.volume_fraction.item(),
            )
            evaluations.append(
                SampleEvaluation(
                    sample_id=i,
                    **evaluation_tuple
                )
            )
            

    res = ModelEvaluation(SampleEvaluations=evaluations)
        
    
    return res


def same_weights(a: dict | None, b: dict | None) -> bool:
    """Check whether two model state dicts contain identical weights."""
    if a is None or b is None:
        return a is b
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def main():
    """Main function for evaluating the model."""
    project_root = Path(__file__).parent.parent
    n_samples = None
    
    
    #*#####################################
    # n_samples = 250  # comment this line to use all samples
    mode = Mode.TEST # TEST, VALIDATE
    checkpoint_path = (
        project_root
        / "trainings"
        / "Vit_V4_e128_b6_h4_p16_c2_o2"
        / "adam_0.0003_mse_binary_mismatch_03_datt0.1_dmlp0.1"
        / "best.pth"
    )
    #*#####################################
    
    
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = Checkpoint.load(checkpoint_path)
    
    # store a snapshot of the evaluated weights, so the evaluation stays tied to them even if training later overwrites the original
    eval_dir = checkpoint_path.parent / f"eval_epoch{checkpoint.epoch}"
    logger.info(f"Using evaluation directory {eval_dir}")
    eval_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = eval_dir / "checkpoint.pth"
    if snapshot_path.is_file():
        # the same epoch can belong to different weights (e.g. after resuming from an older checkpoint)
        snapshot = Checkpoint.load(snapshot_path)
        if not same_weights(snapshot.model_state_dict, checkpoint.model_state_dict):
            raise ValueError(
                f"{snapshot_path} contains different weights for epoch {checkpoint.epoch}; "
                "move or rename the existing evaluation directory first"
            )
    else:
        logger.info(f"Saving checkpoint snapshot to {snapshot_path}")
        checkpoint.save(snapshot_path)
    
    model = checkpoint.get_model()
    # load dataset
    npz_path = project_root / "data" / "processed" / "topopt_data.npz"
    logger.info(f"Loading dataset from {npz_path}")
    dataset = TODataset(
        npz_path,
        mode=mode,
        input_scaling=checkpoint.training_config.input_scaling,
        orientation=orientation_for_evaluation(checkpoint.training_config.orientation),
    )
    
    
    np.random.seed(42)
    sample_ids = None if n_samples is None else list(range(n_samples))
    
    
    # define the path to save the evaluation results
    sample_limit = "all" if sample_ids is None else len(sample_ids)
    save_path = eval_dir / f"eval_epoch{checkpoint.epoch}_{sample_limit}.json"
    
    # evaluate the model on the selected samples and save the evaluation results
    res: ModelEvaluation = evaluate_model(
        model,
        dataset,
        sample_ids=sample_ids,
    )
    res.save(save_path)
    
    print(f"Evaluation results saved to: {save_path}")
    print(f"results:\n{res.summary()}")
    
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()