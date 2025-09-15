import numpy as np  
import openmc
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from matplotlib.colors import LogNorm



def plot_weight_window(weight_window, index_coord:int=0, energy_index:int=0, 
                       plane:str='xy', saving_fig:bool=False, particle_type:str='neutron'):
    """
    Plot the weight window bounds for a given energy index.
    
    Parameters:
    - wwg: WeightWindowGenerator object
    - energy_index: Index of the energy bin to plot
    """
    plt.figure(figsize=(10, 6))
    if plane == "xy":
        plt.imshow(weight_window.upper_ww_bounds[:, :, index_coord, energy_index].T, origin="lower", norm=LogNorm())
        plt.xlabel('X')
        plt.ylabel('Y')
    elif plane == "xz":
        plt.imshow(weight_window.upper_ww_bounds[:, index_coord, :, energy_index].T, origin="lower", norm=LogNorm())
        plt.xlabel('X')
        plt.ylabel('Z')
    elif plane == "yz":
        plt.imshow(weight_window.upper_ww_bounds[index_coord, :, :, energy_index].T, origin="lower", norm=LogNorm())
        plt.xlabel('Y')
        plt.ylabel('Z')
    plt.colorbar(label='Weight Window Lower Bound')
    plt.title(f'Weight Window Lower Bounds ({particle_type})')
    if saving_fig:
        plt.savefig(f'weight_window_{plane}_{particle_type}.png', dpi=300)
    plt.show()


def create_correction_ww_tally(nx:int=25, ny:int=25, nz:int=25, 
                           lower_left=np.array([-500.0, -500.0, -500]), 
                           upper_right=np.array([500.0, 500.0, 500]), 
                           target=np.array([50.0, 0.0, 0.0])):
    """
    Compute a 3D importance map based on the inverse distance to a target.

    Parameters:
    - nx, ny, nz: Number of grid points in x, y, z directions
    - lower_left: Lower left corner of the grid (numpy array)
    - upper_right: Upper right corner of the grid (numpy array)
    - target: Target position (numpy array)

    Returns:
    - importance_map: 3D numpy array of importance values
    - x_vals, y_vals, z_vals: 1D numpy arrays of grid coordinates
    """
    x_vals = np.linspace(lower_left[0], upper_right[0], nx)
    y_vals = np.linspace(lower_left[1], upper_right[1], ny)
    z_vals = np.linspace(lower_left[2], upper_right[2], nz)

    importance_map = np.zeros((nx, ny, nz))

    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            for k, z in enumerate(z_vals):
                pos = ([x, y, z])
                dist = np.linalg.norm(pos - target)
                importance_map[i, j, k] = (dist + 50.0) / 1e5

    return np.asarray(importance_map)


def apply_correction_ww(ww, correction_weight_window):
    """
    Apply the correction to each weight window generator.
    
    Parameters:
    - ww: List of WeightWindowGenerator objects
    - correction_weight_window: Correction factor to apply
    """
    for wwg in ww:
        for energy_index in range(wwg.lower_ww_bounds.shape[-1]):
            wwg.lower_ww_bounds[..., energy_index] *= correction_weight_window
            wwg.upper_ww_bounds[..., energy_index] *= correction_weight_window
    return ww

def create_and_apply_correction_ww_tally(ww, target=np.array([0.0, 400.0, -300.0]),
                                          nx:int=25, ny:int=25, nz:int=25, 
                                          lower_left=np.array([-500.0, -500.0, -500]), 
                                          upper_right=np.array([500.0, 500.0, 500])):
    """
    Create a correction weight window tally and apply it to the weight windows.
    
    Parameters:
    - target: Target position for the correction
    """
    correction_weight_window = create_correction_ww_tally(target=target, nx=nx, ny=ny, nz=nz,
                                                           lower_left=lower_left, upper_right=upper_right)
    ww = apply_correction_ww(ww, correction_weight_window)
    return ww


def get_ww_size(weight_windows:list, particule_type:str = "neutron") -> tuple:
    """
    Get the size of the weight window for a given particle type.

    Parameters:
    weight_windows (list): List of weight window objects.
    particule_type (str): Particle type to filter (default: "neutron").

    Returns:
    tuple: Size of the weight window for the specified particle type.
    """
    for wwg in weight_windows:
        if wwg.particle_type == particule_type:
            size = wwg.lower_ww_bounds.shape[:-1]
            ww_sizes = size
    return ww_sizes

def remove_zeros_from_ww(weight_windows:list) -> list:
    """
    Remove zeros from the weight window arrays by replacing them with the minimum nonzero value
    in each energy group slice.

    Parameters:
        weight_windows (list): List of weight window objects.

    Returns:
        list: List of weight window objects with zeros replaced by the minimum nonzero value.
    """
    for wwg in weight_windows:
        for index_energy in range(wwg.lower_ww_bounds.shape[-1]):
            current_slice = wwg.lower_ww_bounds[:, :, :, index_energy]
            if np.sum(current_slice[current_slice > 0]) != 0:
                min_nonzero = np.min(current_slice[current_slice > 0])
                current_slice[current_slice <= 0] = min_nonzero
                wwg.lower_ww_bounds[:, :, :, index_energy] = current_slice

            current_slice_upper = wwg.upper_ww_bounds[:, :, :, index_energy]
            if np.sum(current_slice_upper[current_slice_upper > 0]) != 0:
                min_nonzero_upper = np.min(current_slice_upper[current_slice_upper > 0])
                current_slice_upper[current_slice_upper <= 0] = min_nonzero_upper
                wwg.upper_ww_bounds[:, :, :, index_energy] = current_slice_upper
    return weight_windows