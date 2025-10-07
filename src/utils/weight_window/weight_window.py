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
                           target=np.array([50.0, 0.0, 0.0]),
                           alpha:float=1.0,
                           beta:float=50.0,
                           factor_div:float=1e5) -> np.ndarray:
    """
    Compute a 3D importance map based on the inverse distance to a target.

    Parameters:
    - nx, ny, nz: Number of grid points in x, y, z directions
    - lower_left: Lower left corner of the grid (numpy array)
    - upper_right: Upper right corner of the grid (numpy array)
    - target: Target position (numpy array)
    - beta: Smoothing parameter (float)
    - factor_div: Scaling parameter (float)

    Returns:
    - importance_map: 3D numpy array of importance values
    """
    x_vals = np.linspace(lower_left[0], upper_right[0], nx)
    y_vals = np.linspace(lower_left[1], upper_right[1], ny)
    z_vals = np.linspace(lower_left[2], upper_right[2], nz)

    importance_map = np.zeros((nx, ny, nz), dtype=np.float64)

    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            for k, z in enumerate(z_vals):
                pos = np.array([x, y, z])
                dist = np.linalg.norm(pos - target)
                importance_map[i, j, k] = (alpha * dist + beta) / factor_div
    return importance_map


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




def make_oriented_importance(mesh_bounds, shape,
                             sphere_center, sphere_radius,
                             I_min=1e-3, I_max=1.0, lambda_radial=1.0,
                             beam_dir=None, angular_power=2.0, alpha=0.1):
    """
    mesh_bounds: ((x0,x1),(y0,y1),(z0,z1))
    shape: (nx,ny,nz)
    sphere_center: (xc,yc,zc)
    sphere_radius: R
    I_min, I_max: radial importance floor and peak
    lambda_radial: exponential decay constant (1/length)
    beam_dir: None or 3-vector unit (direction which is 'looking at' the sphere)
    angular_power: p in (cos(theta))**p
    alpha: floor weight for angular factor (0..1)
    """
    (x0,x1),(y0,y1),(z0,z1) = mesh_bounds
    nx, ny, nz = shape

    xs = np.linspace(x0, x1, nx, endpoint=False) + (x1 - x0)/nx*0.5
    ys = np.linspace(y0, y1, ny, endpoint=False) + (y1 - y0)/ny*0.5
    zs = np.linspace(z0, z1, nz, endpoint=False) + (z1 - z0)/nz*0.5

    X = xs[:, None, None]
    Y = ys[None, :, None]
    Z = zs[None, None, :]

    # position of cell centers
    # broadcasted arrays of same shape (nx,ny,nz)
    xv = X
    yv = Y
    zv = Z

    # vector from voxel to sphere center
    sx, sy, sz = sphere_center
    vx = sx - xv
    vy = sy - yv
    vz = sz - zv
    r = np.sqrt(vx**2 + vy**2 + vz**2)

    # distance to sphere surface
    d_surface = np.maximum(0.0, r - sphere_radius)

    # radial factor (exponential)
    fr = I_min + (I_max - I_min) / np.exp(-lambda_radial * d_surface)

    # angular factor
    if beam_dir is None:
        fa = 1.0
    else:
        u = np.array(beam_dir, dtype=float)
        unorm = np.linalg.norm(u)
        if unorm == 0.0:
            raise ValueError("beam_dir must be non-zero")
        u /= unorm
        # unit vectors from voxel to sphere (handle r=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            vxn = np.where(r > 0, vx / r, 0.0)
            vyn = np.where(r > 0, vy / r, 0.0)
            vzn = np.where(r > 0, vz / r, 0.0)
        cos_theta = vxn*u[0] + vyn*u[1] + vzn*u[2]
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        fa = np.maximum(0.0, cos_theta) ** angular_power
        # combine with floor alpha to avoid zeros everywhere
        fa = alpha + (1.0 - alpha) * fa

    importance = fr * fa
    return importance

def apply_spherical_correction_to_weight_windows(ww, particule_type='photon', sphere_center=(0.0, 0.0, 500.0), sphere_radius=50.0):
    shape_ww = get_ww_size(weight_windows=ww, particule_type=particule_type)
    lower_left = ww[0].mesh.bounding_box.lower_left
    upper_right = ww[0].mesh.bounding_box.upper_right
    mesh_bounds = ((lower_left[0], upper_right[0]), (lower_left[1], upper_right[1]), (lower_left[2], upper_right[2]))
    shape = (shape_ww[0], shape_ww[1], shape_ww[2])
    correction_matrix = make_oriented_importance(
        mesh_bounds, shape, sphere_center, sphere_radius,
        I_min=1e-5, I_max=1e-1, lambda_radial=0.005,
        beam_dir=None, angular_power=4.0, alpha=0.02
    )
    ww_corrected = apply_correction_ww(ww=ww, correction_weight_window=correction_matrix)
    return ww_corrected