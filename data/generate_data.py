import csv
import os
from functools import partial
from itertools import combinations
from multiprocessing import Pool
from typing import Any, Callable, Tuple
from time import time

#! AI code snippet
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
#!#####

import numpy as np
from scipy.sparse.linalg import spsolve

from topoptlab import topology_optimization
from topoptlab.elements.bilinear_quadrilateral import bmatrix, create_edofMat
from topoptlab.elements.linear_elasticity_2d import lk_linear_elast_2d
from topoptlab.fem import assemble_matrix, create_matrixinds

N_ELEMENTS_PER_EDGE = 64 # Number of elements along one edge of the square domain
EDGE_LENGTH = N_ELEMENTS_PER_EDGE + 1
N_PROCESSES = 6 

def generate_data():
    
    n_samples = 30000
    
    # build catalog of possible boundary conditions once in the beginning
    bc_catalog = _build_bc_catalog()
    
    # set output directory and file for storing generated data
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_data")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "topopt_data.csv")
    
    
    # check if output file already exists, so we don't accidentally overwrite existing data
    if os.path.exists(output_file):
        print(f"File {output_file} already exists.")
        print(f"Data will be appended to the existing file.")
        input("Press Enter to continue and append data to the existing file...")  # wait for user input before appending to the existing file
    else:
        # create csv with header row
        print(f"Creating new output file {output_file}.")
        with open(output_file, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=";")
            csvwriter.writerow(["stresses", "strain_energy_densities", "fixed_dofs", "f", "volfrac", "densities"])

    
    #! Parallel computationstuff was AI aided  
    
    # one distinct random seed per sample, so the parallel workers never produce duplicate samples
    seeds = np.random.SeedSequence().generate_state(n_samples)
    
    # samples are computed in parallel worker processes; only this (main) process writes to the csv
    with Pool(processes=N_PROCESSES) as pool:
        for i, (row, optimization_time, fem_time) in enumerate(pool.imap_unordered(partial(_generate_sample_safe, output_file=output_file, bc_catalog=bc_catalog), seeds)):
            # a failed sample (e.g. MemoryError) is skipped and logged instead of aborting the whole run
            if row is None:
                print(f"Sample {i+1}/{n_samples} FAILED and was skipped (seed {optimization_time}): {fem_time}")
                continue
            
            # append data to csv
            with open(output_file, "a", newline="") as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=";")
                csvwriter.writerow(row)
            
            print(f"Sample {i+1}/{n_samples} generated in {optimization_time + fem_time:.2f} seconds. (Optimization: {optimization_time:.2f}, FEM: {fem_time:.2f})")
    #!####

def _generate_sample_safe(seed, output_file, bc_catalog):
    # I ran out of RAM some times, so this wrapper ensures that the whole process doesn't crash and logs the error instead
    try:
        return _generate_sample(seed, output_file, bc_catalog)
    except Exception as e:
        return None, seed, f"{type(e).__name__}: {e}"

def _generate_sample(seed, output_file, bc_catalog):
    np.random.seed(seed)
    start_time = time()
    
    # get random parameters
    fixed_dofs, fixed_nodes = _sample_fixed_dofs(bc_catalog)
    f, load_node = _sample_f(fixed_nodes)
    bcs = _get_bcs(f, fixed_dofs)
    volfrac = _sample_volfrac()
    
    densities = _generate_topology(bcs, volfrac, output_file)
    
    optimization_finished_time = time()
    
    stresses, strain_energy_density = _generate_stress_and_strain_distribution(bcs)
    
    end_time = time()
    
    # store the load  as [node index, fx, fy] for the CSV file
    load = [int(load_node), float(f[2 * load_node, 0]), float(f[2 * load_node + 1, 0])]
    
    row = [stresses.tolist(), strain_energy_density.tolist(), fixed_dofs, load, volfrac, densities.tolist()]
    return row, optimization_finished_time - start_time, end_time - optimization_finished_time
        
def _sample_f(fixed_nodes):
    """
    Return a randomly sampled load corresponding to the degrees of freedom not fixed by the boundary conditions.
    """
    # sample from angle
    angle = np.random.randint(0, 6) * (np.pi / 3) # 6 evenly distributed directions
    force_vector = np.array([np.cos(angle), np.sin(angle)]) # always unit magnitude
    
    
    #! Code snippet created by AI resolving a bug I had
    # select a random boundary node that is not fixed by the boundary conditions
    # node index = EDGE_LENGTH * x + y
    boundary_nodes = {
        EDGE_LENGTH * x + y
        for x in range(EDGE_LENGTH)
        for y in range(EDGE_LENGTH)
        if x in (0, EDGE_LENGTH - 1) or y in (0, EDGE_LENGTH - 1)
    }
    #!######
    
    
    available_nodes = sorted(boundary_nodes - set(fixed_nodes))
    selected_node = np.random.choice(available_nodes)
    
    # assign the force vector to the selected node / dofs 
    f = np.zeros(((EDGE_LENGTH * EDGE_LENGTH * 2),1))
    f[2 * selected_node] = force_vector[0] 
    f[2 * selected_node + 1] = force_vector[1]
    
    return f, selected_node
    
def _build_bc_catalog() -> list:   
    
    last = EDGE_LENGTH - 1
    mid = EDGE_LENGTH // 2
    first_half = range(mid)  # 0 .. mid - 1
    second_half = range(mid, EDGE_LENGTH)  # mid .. last


    # topoptlab numbers nodes column-wise: node = EDGE_LENGTH * x + y,
    # with x from left (0) to right and y from top (0) to bottom
    def node(x: int, y: int) -> int:
            return EDGE_LENGTH * x + y
    
    
    # possible fixed nodes
    top_left_point = [node(0, 0), node(1, 0)] # two nodes, to avoid rotational dof
    top_middle_point = [node(mid, 0), node(mid - 1, 0)] # use both middle point
    top_right_point = [node(last, 0), node(last - 1, 0)] 
    left_middle_point = [node(0, mid), node(0, mid - 1)] 
    right_middle_point = [node(last, mid), node(last, mid - 1)] 
    bottom_left_point = [node(0, last), node(1, last)] 
    bottom_middle_point = [node(mid, last), node(mid - 1, last)]
    bottom_right_point = [node(last, last), node(last - 1, last)]
    top_left_edge = [node(x, 0) for x in first_half]
    top_right_edge = [node(x, 0) for x in second_half]
    left_top_edge = [node(0, y) for y in first_half]
    left_bottom_edge = [node(0, y) for y in second_half]
    right_top_edge = [node(last, y) for y in first_half]
    right_bottom_edge = [node(last, y) for y in second_half]
    bottom_left_edge = [node(x, last) for x in first_half]
    bottom_right_edge = [node(x, last) for x in second_half]
    top_edge = top_left_edge + top_right_edge
    bottom_edge = bottom_left_edge + bottom_right_edge
    left_edge = left_top_edge + left_bottom_edge
    right_edge = right_top_edge + right_bottom_edge
    
    
    
    # create mapping for sampling
    bc_mapping = {
        1: top_left_point,
        2: top_middle_point,
        3: top_right_point,
        4: left_middle_point,
        5: right_middle_point,
        6: bottom_left_point,
        7: bottom_middle_point,
        8: bottom_right_point,
        9: top_left_edge,
        10: top_right_edge,
        11: left_top_edge,
        12: left_bottom_edge,
        13: right_top_edge,
        14: right_bottom_edge,
        15: bottom_left_edge,
        16: bottom_right_edge,
        17: top_edge,
        18: bottom_edge,
        19: left_edge,
        20: right_edge
    }
    
    # create a catalog of all possible combinations of boundary conditions
    bc_catalog = set()
    for n_boundary_conditions in range(1, 5):
        for combination in combinations(bc_mapping.keys(), n_boundary_conditions):
            fixed_nodes = set()
            for bc_idx in combination:
                fixed_nodes.update(bc_mapping[bc_idx])
            bc_catalog.add(tuple(sorted(fixed_nodes)))

    
    # exclude combination where all sides are fully fixed
    bc_catalog = bc_catalog - {tuple(sorted(top_edge + bottom_edge + left_edge + right_edge))}
    
    return sorted(bc_catalog)

def _sample_fixed_dofs(bc_catalog: list) -> tuple[list[int], set[int]]:
    """
    Return a list with the indices of the fixed degrees of freedom based on randomly sampled boundary conditions.
    """
    
    # note: dof index = 2 * node index for x displacement, 2 * node index + 1 for y displacement 

    # sample from the boundary condition catalog
    fixed_dofs = set()
    fixed_nodes = set()
    bc_idx = np.random.randint(0, len(bc_catalog))
    bc = bc_catalog[bc_idx]
    fixed_dofs.update([2 * node for node in bc]) # x
    fixed_dofs.update([2 * node + 1 for node in bc]) # y
    fixed_nodes.update(bc)
    
    fixed_dofs = list(fixed_dofs)  # convert set to list for further processing
    
    return fixed_dofs, fixed_nodes
    
def _get_bcs(f, fixed_dofs) -> Callable:
    u = np.zeros(((EDGE_LENGTH * EDGE_LENGTH * 2), 1))
    dofs = np.arange(EDGE_LENGTH * EDGE_LENGTH * 2)
    
    # define the boundary conditions function called in topology_optimization.main()
    def bcs_fun(
        nelx: int, 
        nely: int,
        ndof: int, 
        **kwargs: Any
    ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,None]:
        return u,f,fixed_dofs,np.setdiff1d(dofs,fixed_dofs),None
    
    return bcs_fun

def _sample_volfrac():
    return np.random.uniform(0.3, 0.5)  # range as in paper

def _generate_topology(
    bcs: Callable,
    volfrac: float,
    output_file: str,
):  
    rmin = 1.5 # Minimum radius for the filter, i.e. the size of the neighborhood used for sensitivity filtering
    
    # +------------------+-------+-----------+---------------------------------------------+
    # | Key              | Type  | Default   | Description                                 |
    # +------------------+-------+-----------+---------------------------------------------+
    # | "file"           | str   | "topopt"  | Base name used for output files.            |
    # | "display"        | bool  | True      | Display intermediary design.                |
    # | "export"         | bool  | True      | Export final results to disk as VTK.        |
    # | "write_log"      | bool  | True      | Write a log file.                           |
    # | "profile"        | bool  | False     | Profile code execution.                     |
    # | "output_movie"   | bool  | False     | Generate a movie of intermediate designs.   |
    # | "verbosity"      | int   | 20        | Level of verbosity (see SimpleLogger).      |
    # +------------------+-------+-----------+---------------------------------------------+
    output_kw = {
        "file": output_file,
        "display": False,
        "export": False,
        "write_log": False,
        "profile": False,
        "output_movie": False,
        "verbosity": 20
    }
    
    x, obj = topology_optimization.main(
        nelx=N_ELEMENTS_PER_EDGE,
        nely=N_ELEMENTS_PER_EDGE,
        volfrac=volfrac,
        rmin=rmin,
        bcs=bcs,
        output_kw=output_kw,
        optimizer="oc", #! optimality criteria: ~3x cheaper per iteration than MMA for a single volume constraint
        lin_solver_kw={"name": "cvxopt-cholmod"} #! sparse Cholesky: ~3x faster than scipy-direct (SuperLU)
    )
    
    return x


# ! I created this function mostly with AI in order no to spend too much time on the FEM stuff
def _generate_stress_and_strain_distribution(bcs) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the von Mises stress and strain energy density per element for a fully
    solid material only; ignore the optimized topology.
    """

    nelx = N_ELEMENTS_PER_EDGE
    nely = N_ELEMENTS_PER_EDGE
    nnodes_per_side = nelx + 1
    ndof = 2 * nnodes_per_side * nnodes_per_side

    u, f, fixed, free, _ = bcs(nelx=nelx, nely=nely, ndof=ndof)
    u = np.asarray(u, dtype=float)
    f = np.asarray(f, dtype=float)
    fixed = np.asarray(fixed, dtype=int)
    free = np.asarray(free, dtype=int)

    element_matrix = lk_linear_elast_2d(E=1.0, nu=0.3, plane_stress=True, l=np.array([1.0, 1.0]))
    edof_mat, _, _, _, _ = create_edofMat(nelx=nelx, nely=nely, nnode_dof=2)
    i_k, j_k = create_matrixinds(edofMat=edof_mat, mode="full")

    solid_scale = np.ones(edof_mat.shape[0], dtype=float)
    s_k = element_matrix[None, :, :] * solid_scale[:, None, None]
    k_global = assemble_matrix(sK=s_k.reshape(s_k.size), iK=i_k, jK=j_k, ndof=ndof, solver="scipy-direct", springs=None)

    k_reduced = k_global[free, :][:, free]
    rhs_reduced = f[free, 0]
    u_reduced = spsolve(k_reduced.tocsc(), rhs_reduced) #! sparse solve instead of dense (~25x faster, same result)
    u = np.zeros((ndof, 1), dtype=float)
    u[free, 0] = u_reduced

    ex = np.arange(nelx, dtype=float)
    ey = np.arange(nely, dtype=float)
    x_coords, y_coords = np.meshgrid(ex, ey, indexing="xy")
    xe = np.stack([
        np.column_stack((x_coords.ravel(), y_coords.ravel())),
        np.column_stack((x_coords.ravel() + 1.0, y_coords.ravel())),
        np.column_stack((x_coords.ravel() + 1.0, y_coords.ravel() + 1.0)),
        np.column_stack((x_coords.ravel(), y_coords.ravel() + 1.0)),
    ], axis=1)
    xe = np.asarray(xe, dtype=float)

    xi = np.zeros(1, dtype=float)
    eta = np.zeros(1, dtype=float)
    B, _ = bmatrix(xi=xi, eta=eta, xe=xe, all_elems=True, return_detJ=True)
    ue = u[edof_mat, 0]
    strains = np.einsum("eij,ej->ei", B, ue)

    d_matrix = np.array([
        [1.0, 0.3, 0.0],
        [0.3, 1.0, 0.0],
        [0.0, 0.0, 0.5 * (1.0 - 0.3)],
    ], dtype=float) / (1.0 - 0.3 ** 2)
    stresses = strains @ d_matrix.T
    strain_energy_density = 0.5 * np.sum(strains * stresses, axis=1)

    # plane stress von Mises: sqrt(sx^2 - sx*sy + sy^2 + 3*txy^2)
    sx, sy, txy = stresses[:, 0], stresses[:, 1], stresses[:, 2]
    von_mises = np.sqrt(sx**2 - sx * sy + sy**2 + 3 * txy**2)

    return von_mises.reshape(nely, nelx), strain_energy_density.reshape(nely, nelx)

# ! AI created function
def _element_field_to_image(values: np.ndarray, n_components: int = 1) -> np.ndarray:
    """
    Convert a flat element field in topoptlab ordering (element = elx * nely + ely)
    to an image of shape (nely, nelx[, n_components]) with y pointing down.
    """
    nelx = nely = N_ELEMENTS_PER_EDGE
    values = np.asarray(values, dtype=float).reshape(nelx, nely, n_components)
    image = values.transpose(1, 0, 2)
    return image[..., 0] if n_components == 1 else image

# ! AI created function
def visualize_row(csv_file: str, row_index: int = 0, save_path: str | None = None):
    """
    Plot the von Mises stress, strain energy density, boundary conditions/load
    and optimized structure of one sample stored in the csv file.
    """
    import json
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    # rows are far larger than the default csv field size limit
    csv.field_size_limit(2**31 - 1)

    row = None
    with open(csv_file, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=";")
        for i, current_row in enumerate(reader):
            if i == row_index:
                row = current_row
                break
    if row is None:
        raise IndexError(f"Row {row_index} not found in {csv_file}.")

    von_mises = _element_field_to_image(json.loads(row["stresses"]))
    sed = _element_field_to_image(json.loads(row["strain_energy_densities"]))
    densities = _element_field_to_image(json.loads(row["densities"]))
    fixed_dofs = np.asarray(json.loads(row["fixed_dofs"]), dtype=int)
    f = np.asarray(json.loads(row["f"]), dtype=float).ravel()
    #! compact format [node index, fx, fy] -> full dof vector (old csv files store the full vector)
    if f.size == 3:
        load_node, fx, fy = int(f[0]), f[1], f[2]
        f = np.zeros(2 * EDGE_LENGTH * EDGE_LENGTH)
        f[2 * load_node], f[2 * load_node + 1] = fx, fy
    volfrac = float(row["volfrac"])

    # node index = (nely + 1) * x + y  ->  coordinates in the image frame
    fixed_nodes = np.unique(fixed_dofs // 2)
    fixed_x, fixed_y = fixed_nodes // EDGE_LENGTH, fixed_nodes % EDGE_LENGTH
    load_nodes = np.unique(np.nonzero(f)[0] // 2)

    extent = (0, N_ELEMENTS_PER_EDGE, N_ELEMENTS_PER_EDGE, 0)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    def _positive_log_norm(field):
        positive = field[field > 0]
        if positive.size == 0:
            return None
        return LogNorm(vmin=max(positive.min(), positive.max() * 1e-6), vmax=positive.max())

    ax = axes[0, 0]
    im = ax.imshow(von_mises, cmap="jet", extent=extent, norm=_positive_log_norm(von_mises))
    ax.set_title("Von Mises stress (solid domain, log)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 1]
    im = ax.imshow(sed, cmap="inferno", extent=extent, norm=_positive_log_norm(sed))
    ax.set_title("Strain energy density (solid domain, log)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 0]
    ax.add_patch(plt.Rectangle((0, 0), N_ELEMENTS_PER_EDGE, N_ELEMENTS_PER_EDGE, fill=False, color="black"))
    ax.scatter(fixed_x, fixed_y, marker="s", s=12, color="tab:blue", label="Fixed nodes", zorder=3)
    arrow_length = 0.2 * N_ELEMENTS_PER_EDGE
    for node in load_nodes:
        fx, fy = f[2 * node], f[2 * node + 1]
        magnitude = np.hypot(fx, fy)
        # y axis points down in the image frame, matching the topoptlab mesh
        x0, y0 = node // EDGE_LENGTH, node % EDGE_LENGTH
        ax.annotate(
            "", xy=(x0, y0), xytext=(x0 - arrow_length * fx / magnitude, y0 - arrow_length * fy / magnitude),
            arrowprops=dict(arrowstyle="-|>", color="tab:red", lw=2), zorder=4,
        )
        ax.scatter([x0], [y0], color="tab:red", s=30, label="Load", zorder=4)
    margin = arrow_length + 2
    ax.set_xlim(-margin, N_ELEMENTS_PER_EDGE + margin)
    ax.set_ylim(N_ELEMENTS_PER_EDGE + margin, -margin)
    ax.set_aspect("equal")
    ax.set_title("Boundary conditions and load")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1, 1]
    ax.imshow(densities, cmap="gray_r", vmin=0, vmax=1, extent=extent)
    ax.set_title(f"Optimized structure (volfrac target {volfrac:.3f}, actual {densities.mean():.3f})")

    fig.suptitle(f"{csv_file} - row {row_index}")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120)
    else:
        plt.show()
    return fig

if __name__ == "__main__":
    generate_data()
    # visualize_row(csv_file="topopt_data.csv", row_index=0)