"""Create evaluations for the results of the vision transformer model"""

from dataclasses import dataclass
import logging

import numpy as np
import torch
from scipy.sparse.linalg import spsolve
from topoptlab.elements.bilinear_quadrilateral import create_edofMat
from topoptlab.elements.linear_elasticity_2d import lk_linear_elast_2d
from topoptlab.fem import assemble_matrix, create_matrixinds
from src.utils import un_one_hot_encode, postprocess

logger = logging.getLogger(__name__)

@dataclass
class SampleEvaluation:
    rel_compliance_error_binary: float # all values either 1 or 0
    rel_compliance_error_continuous: float # values in interval [0, 1] using penalization for intermediate values (compliance function in SIMP)
    rel_volume_error: float # error in the volume fraction of the topology
    density_error: float # measure for non-binary densities (values not equal to 0 or 1)
    
@dataclass
class ModelEvaluation:
    mean_rel_compliance_error_binary: float
    median_rel_compliance_error_binary: float
    rel_compliance_error_above_30_percent: float
    mean_rel_compliance_error_continuous: float
    median_rel_compliance_error_continuous: float
    rel_compliance_error_continuous_above_30_percent: float
    mean_density_error: float
    median_density_error: float
    density_error_above_0_1: float
    mean_rel_volume_error: float
    median_rel_volume_error: float
    rel_volume_error_above_0_1: float
    SampleEvaluations: list[SampleEvaluation]


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
    
    Returns the signed relative compliance error as a scalar value ((c_target - c_predicted) / c_target)
    """

    if binarify_densities:
        predicted = predicted.round()
        target = target.round()
    
    # todo: how do we actually compute the compliance????
    c_predicted = compliance(densities=predicted.numpy(), load=load, fixed_dofs=fixed_dofs)
    c_target = compliance(densities=target.numpy(), load=load, fixed_dofs=fixed_dofs)
    
    return (c_target - c_predicted) / c_target

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
    
    predicted_volume = predicted.sum() / predicted.numel()
    
    return (predicted_volume - target_volume_fraction) / target_volume_fraction # relative volume error

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
    
    return SampleEvaluation(
        rel_compliance_error_binary=rel_compliance_error(predicted, target, load, fixed_dofs, binarify_densities=True),
        rel_compliance_error_continuous=rel_compliance_error(predicted, target, load, fixed_dofs, binarify_densities=False),
        density_error=float(density_error(predicted)),
        rel_volume_error=float(rel_volume_error(predicted, volume_fraction)),
    )
    
def evaluate_model(model, dataset):
    results = []
    
    model.eval()
    evaluations = []
    with torch.no_grad():
        for i in range(len(dataset)):
            logger.info(f"Evaluating sample {i}/{len(dataset)}")
            x, y = dataset[i]
            angle = dataset.angles[i]
            predicted_patched = model(x.unsqueeze(0)).squeeze(0)

            predicted, load, fixed_dofs = postprocess(
                predicted_patched, angle, dataset.patch_size,
                load=x[1:5], fixed_dofs_one_hot=x[5:5 + 4 * 64],
            )
            target, _, _ = postprocess(y, angle, dataset.patch_size)

            evaluations.append(
                evaluate_prediction(predicted, target, load, fixed_dofs, float(x[0]))
            )

    if not evaluations:
        raise ValueError("dataset must contain at least one sample")

    # statistics over all evaluations
    mean_rel_compliance_error_binary = sum(e.rel_compliance_error_binary for e in evaluations) / len(evaluations)
    median_rel_compliance_error_binary = sorted(e.rel_compliance_error_binary for e in evaluations)[len(evaluations) // 2]
    rel_compliance_error_above_30_percent = sum(e.rel_compliance_error_binary > 0.3 for e in evaluations) / len(evaluations)

    mean_rel_compliance_error_continuous = sum(e.rel_compliance_error_continuous for e in evaluations) / len(evaluations)
    median_rel_compliance_error_continuous = sorted(e.rel_compliance_error_continuous for e in evaluations)[len(evaluations) // 2]
    rel_compliance_error_continuous_above_30_percent = sum(e.rel_compliance_error_continuous > 0.3 for e in evaluations) / len(evaluations)

    mean_density_error = sum(e.density_error for e in evaluations) / len(evaluations)
    median_density_error = sorted(e.density_error for e in evaluations)[len(evaluations) // 2]
    density_error_above_0_1 = sum(e.density_error > 0.1 for e in evaluations) / len(evaluations)

    mean_rel_volume_error = sum(e.rel_volume_error for e in evaluations) / len(evaluations)
    median_rel_volume_error = sorted(e.rel_volume_error for e in evaluations)[len(evaluations) // 2]
    rel_volume_error_above_0_1 = sum(e.rel_volume_error > 0.1 for e in evaluations) / len(evaluations)
        
    res = ModelEvaluation(
        mean_rel_compliance_error_binary=mean_rel_compliance_error_binary,
        median_rel_compliance_error_binary=median_rel_compliance_error_binary,
        rel_compliance_error_above_30_percent=rel_compliance_error_above_30_percent,
        mean_rel_compliance_error_continuous=mean_rel_compliance_error_continuous,
        median_rel_compliance_error_continuous=median_rel_compliance_error_continuous,
        rel_compliance_error_continuous_above_30_percent=rel_compliance_error_continuous_above_30_percent,
        mean_density_error=mean_density_error,
        median_density_error=median_density_error,
        density_error_above_0_1=density_error_above_0_1,
        mean_rel_volume_error=mean_rel_volume_error,
        median_rel_volume_error=median_rel_volume_error,
        rel_volume_error_above_0_1=rel_volume_error_above_0_1,
        SampleEvaluations=evaluations
    )
        
    return res