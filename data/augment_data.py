import pandas as pd
import numpy as np
from pathlib import Path


from data.utils import Samples, save_samples, load_samples, LABELED_DATA_DIR
    

def mirror_data(samples: Samples) -> Samples:
    """Mirror the data and return the mirrored version"""
    
    # distributions can just be flipped (2nd axis corresponds to the horizontal direction)
    mirrored_stresses = np.flip(samples.stresses, axis=2)
    mirrored_strain_energies = np.flip(samples.strain_energies, axis=2)
    mirrored_densities = np.flip(samples.densities, axis=2)

    # for loads, we mirror only the horizontal component
    mirrored_loads = samples.loads.copy()
    mirrored_loads[:, 1] = -mirrored_loads[:, 1]  # flip the horizontal component
    # mirror node index too, but below together with the dofs
    
    # for fixed_dofs, we use the node indices on the opposite side of the domain
    mirrored_fixed_dofs = np.zeros_like(samples.fixed_dofs)
    for i, (dofs, mirrored_load) in enumerate(zip(samples.fixed_dofs, mirrored_loads)):
        
        # dof
        nodes = dofs // 2
        x, y = nodes // 65, nodes % 65
        mirrored_nodes = 65 * (64 - x) + y # mirrored node indices horizontally
        mirrored_fixed_dofs[i] = 2 * mirrored_nodes + (dofs % 2)
        
        # load
        load_x, load_y = mirrored_load[0] // 65, mirrored_load[0] % 65
        mirrored_load[0] = 65 * (64 - load_x) + load_y  # mirror horizontally

    return Samples(
        fixed_dofs=mirrored_fixed_dofs,
        loads=mirrored_loads,
        stresses=mirrored_stresses,
        strain_energies=mirrored_strain_energies,
        densities=mirrored_densities,
        volume_fractions=samples.volume_fractions
    )

def rotate_data(samples: Samples) -> Samples:
    """Rotate the data by 90 degrees counterclockwise and return the rotated version"""
    
    # distributions can be rotated by 90 degrees counterclockwise using np.rot90
    rotated_stresses = np.rot90(samples.stresses, k=1, axes=(1, 2))
    rotated_strain_energies = np.rot90(samples.strain_energies, k=1, axes=(1, 2))
    rotated_densities = np.rot90(samples.densities, k=1, axes=(1, 2))

    # for loads, we rotate the components accordingly
    rotated_loads = samples.loads.copy()
    rotated_loads[:, [1, 2]] = samples.loads[:, [2, 1]]
    rotated_loads[:, 1] = -rotated_loads[:, 1]  # fx' = -fy, fy' = fx (y axis points up in topoptlab)
    # and adjust the node index (further down in the fixed_dofs section)

    # for fixed_dofs, we rotate the node indices
    rotated_fixed_dofs = np.zeros_like(samples.fixed_dofs)
    for i, (dofs, rotated_load) in enumerate(zip(samples.fixed_dofs, rotated_loads)):        
        # dofs
        nodes = dofs // 2
        x, y = nodes // 65, nodes % 65    
        rotated_nodes = 65 * y + (64 - x)  # rotate 90 degrees counterclockwise
        rotated_fixed_dofs[i] = 2 * rotated_nodes + (dofs % 2)

        # load
        load_x, load_y = rotated_load[0] // 65, rotated_load[0] % 65
        rotated_load[0] = 65 * load_y + (64 - load_x)  # rotate 90 degrees counterclockwise
        

    return Samples(
        fixed_dofs=rotated_fixed_dofs,
        loads=rotated_loads,
        stresses=rotated_stresses,
        strain_energies=rotated_strain_energies,
        densities=rotated_densities,
        volume_fractions=samples.volume_fractions
    )

def augment_data(samples: Samples) -> Samples:
    """Augment the data by mirroring and rotating it, returning a single augmented sample

        Returns a single augmented Samples object
    """
    samples = samples + mirror_data(samples) # note: overloaded addition operator for Samples class to concatenate samples
    
    samples = samples \
            + rotate_data(samples) \
            + rotate_data(rotate_data(samples)) \
            + rotate_data(rotate_data(rotate_data(samples))) # not the most efficient way but whatever
        
    return samples

def main():
    
    samples_path = LABELED_DATA_DIR / "train"
    samples = load_samples(samples_path)
    augmented_samples = samples + mirror_data(samples) #todo: remove this later
    save_samples(augmented_samples, samples_path / "augmented_mirrored")

if __name__ == "__main__":
    main()


    