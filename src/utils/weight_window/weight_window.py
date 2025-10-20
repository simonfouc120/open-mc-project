import numpy as np  
import openmc
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

CWD = Path(__file__).parent.resolve()
project_root = Path(__file__).resolve()
sys.path.append(str(project_root))

from src.utils.pre_processing.pre_processing import remove_previous_results
from typing import Iterable, Optional, Tuple, Union, Sequence, Any, List

def plot_weight_window(weight_window, 
                       index_coord:int=0, 
                       energy_index:int=0, 
                       plane:str='xy', 
                       saving_fig:bool=False, 
                       particle_type:str='neutron',
                       bound_type:str='lower',
                       suffix_fig:str=''):
    """
    Plot the weight window bounds for a given energy index.
    
    Parameters:
    - wwg: WeightWindowGenerator object
    - energy_index: Index of the energy bin to plot
    """
    plt.figure(figsize=(10, 6))
    if plane == "xy":
        if bound_type == 'lower':
            plt.imshow(weight_window.lower_ww_bounds[:, :, index_coord, energy_index].T, origin="lower", norm=LogNorm())
        elif bound_type == 'upper':
            plt.imshow(weight_window.upper_ww_bounds[:, :, index_coord, energy_index].T, origin="lower", norm=LogNorm())
        plt.xlabel('X')
        plt.ylabel('Y')
    elif plane == "xz":
        if bound_type == 'lower':
            plt.imshow(weight_window.lower_ww_bounds[:, index_coord, :, energy_index].T, origin="lower", norm=LogNorm())
        elif bound_type == 'upper':
            plt.imshow(weight_window.upper_ww_bounds[:, index_coord, :, energy_index].T, origin="lower", norm=LogNorm())
        plt.xlabel('X')
        plt.ylabel('Z')
    elif plane == "yz":
        if bound_type == 'lower':
            plt.imshow(weight_window.lower_ww_bounds[index_coord, :, :, energy_index].T, origin="lower", norm=LogNorm())
        elif bound_type == 'upper':
            plt.imshow(weight_window.upper_ww_bounds[index_coord, :, :, energy_index].T, origin="lower", norm=LogNorm())
        plt.xlabel('Y')
        plt.ylabel('Z')
    plt.colorbar(label=f'Weight Window {bound_type} Bound')
    plt.title(f'Weight Window {bound_type} Bounds ({particle_type})')
    if saving_fig:
        plt.savefig(f'weight_window_{plane}_{particle_type}{suffix_fig}.png', dpi=300)
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
    Replace zero (or non-positive) entries in the weight window arrays by the nearest
    (Euclidean-neighbour) entry that is positive.

    Parameters:
        weight_windows (list): List of weight window objects.

    Returns:
        list: List of weight window objects with zeros replaced by the nearest non-zero value.
    """
    for wwg in weight_windows:
        for index_energy in range(wwg.lower_ww_bounds.shape[-1]):
            # lower bounds
            arr = wwg.lower_ww_bounds[:, :, :, index_energy]
            mask_zero = arr <= 0
            if np.any(mask_zero):
                nonzero_coords = np.argwhere(~mask_zero)
                if nonzero_coords.size > 0:
                    zero_coords = np.argwhere(mask_zero)
                    # choose vectorized or loop strategy depending on size to avoid huge memory use
                    if zero_coords.shape[0] * nonzero_coords.shape[0] < 5e7:
                        # vectorized nearest neighbour search (squared distances)
                        d2 = np.sum((zero_coords[:, None, :] - nonzero_coords[None, :, :]) ** 2, axis=2)
                        nn_idx = np.argmin(d2, axis=1)
                        nearest_coords = nonzero_coords[nn_idx]
                        arr[tuple(zero_coords.T)] = arr[tuple(nearest_coords.T)]
                    else:
                        # fallback per-zero search to save memory
                        for z in zero_coords:
                            d2 = np.sum((nonzero_coords - z) ** 2, axis=1)
                            nn = np.argmin(d2)
                            arr[tuple(z)] = arr[tuple(nonzero_coords[nn])]
                    wwg.lower_ww_bounds[:, :, :, index_energy] = arr

            # upper bounds
            arr_u = wwg.upper_ww_bounds[:, :, :, index_energy]
            mask_zero_u = arr_u <= 0
            if np.any(mask_zero_u):
                nonzero_coords_u = np.argwhere(~mask_zero_u)
                if nonzero_coords_u.size > 0:
                    zero_coords_u = np.argwhere(mask_zero_u)
                    if zero_coords_u.shape[0] * nonzero_coords_u.shape[0] < 5e7:
                        d2_u = np.sum((zero_coords_u[:, None, :] - nonzero_coords_u[None, :, :]) ** 2, axis=2)
                        nn_idx_u = np.argmin(d2_u, axis=1)
                        nearest_coords_u = nonzero_coords_u[nn_idx_u]
                        arr_u[tuple(zero_coords_u.T)] = arr_u[tuple(nearest_coords_u.T)]
                    else:
                        for z in zero_coords_u:
                            d2_u = np.sum((nonzero_coords_u - z) ** 2, axis=1)
                            nn_u = np.argmin(d2_u)
                            arr_u[tuple(z)] = arr_u[tuple(nonzero_coords_u[nn_u])]
                    wwg.upper_ww_bounds[:, :, :, index_energy] = arr_u
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


def setup_and_run_wwg(
    source: Union[str, Any],
    model: openmc.Model,
    settings: Optional[openmc.Settings] = None,
    mesh_dimension: Tuple[int, int, int] = (50, 50, 50),
    mesh_size: float = 850.0,
    particle_types: Iterable[str] = ("neutron", "photon"),
    batches: int = 10,
    particles_per_batch: int = 500,
    max_history_splits: int = 1_000,
    ww: Optional[Sequence[Any]] = None,
) -> Tuple[openmc.RegularMesh, int]:
    """
    Configure weight-window generation and run OpenMC.

    Parameters
    - source: path to a surface source file (str) or an openmc.Source / openmc.FileSource instance.
    - model: openmc.Model instance.
    - settings: optional openmc.Settings (if None a new Settings() is created).
    - mesh_dimension: tuple (nx, ny, nz) for the RegularMesh.dimension.
    - mesh_size: half-extent in each axis (mesh lower_left = -mesh_size, upper_right = +mesh_size).
    - particle_types: iterable of particle type strings, e.g. ("neutron", "photon").
    - batches: number of batches to request for the generator run (int).
    - particles_per_batch: number of particles per batch (int).
    - max_history_splits: settings.max_history_splits value (int).
    - ww: optional preexisting weight windows to set on settings.

    Returns
    - mesh, batches_number  (so caller can open the corresponding statepoint)
    """
    # normalize source input
    if isinstance(source, str):
        src: Any = openmc.FileSource(source)
    else:
        src = source

    # Settings
    batches_number: int = batches
    if settings is None:
        settings = openmc.Settings()
    if ww is not None:
        settings.weight_windows = ww
    settings.batches = batches_number
    settings.particles = particles_per_batch
    settings.run_mode = "fixed source"
    settings.source = src
    settings.photon_transport = "photon" in [p.lower() for p in particle_types]
    # if the source object supports a particles attribute, set it
    if hasattr(src, "particles"):
        try:
            src.particles = list(particle_types)  # type: ignore[attr-defined]
        except Exception:
            pass

    # Mesh (from model geometry)
    mesh: openmc.RegularMesh = openmc.RegularMesh().from_domain(model.geometry)
    mesh.dimension = tuple(mesh_dimension)
    mesh.lower_left = (-mesh_size, -mesh_size, -mesh_size)
    mesh.upper_right = (mesh_size, mesh_size, mesh_size)

    # Weight-window generators for each particle type
    wwgs: List[openmc.WeightWindowGenerator] = []
    for ptype in particle_types:
        wwg = openmc.WeightWindowGenerator(
            mesh=mesh,
            max_realizations=settings.batches,
            particle_type=ptype,
            method="magic",
            on_the_fly=True,
        )
        wwgs.append(wwg)

    settings.max_history_splits = max_history_splits
    settings.weight_window_generators = wwgs

    # Attach settings to model and run
    model.settings = settings
    model.export_to_xml()

    remove_previous_results(batches_number=batches_number)
    openmc.run()

    return mesh, batches_number


def create_weight_window(
    model: openmc.Model,
    mesh_dimension: tuple[int, int, int] = (50, 50, 50),
    side_length_mesh: float = 850.0,
    batches_number: int = 10,
    particles_per_batch: int = 10000,
    particle_types: tuple[str, ...] = ("neutron", "photon"),
    num_iterations: int = 15, 
    rm_intermediate_files: bool = True,
    src: openmc.Source = openmc.FileSource('surface_source.h5'),
) -> None:
    mesh = openmc.RegularMesh().from_domain(model.geometry)
    mesh.dimension = tuple(mesh_dimension)
    mesh.lower_left = (-side_length_mesh, -side_length_mesh, -side_length_mesh)
    mesh.upper_right = (side_length_mesh, side_length_mesh, side_length_mesh)

    mesh_filter = openmc.MeshFilter(mesh)

    tallies = openmc.Tallies()

    if "neutron" in particle_types:
        flux_tally_neutrons = openmc.Tally(name="flux_tally_neutron")
        flux_tally_neutrons.filters = [mesh_filter, openmc.ParticleFilter("neutron")]
        flux_tally_neutrons.scores = ["flux"]
        flux_tally_neutrons.id = 55  
        tallies.append(flux_tally_neutrons)

    if "photon" in particle_types:
        flux_tally_photons = openmc.Tally(name="flux_tally_photon")
        flux_tally_photons.filters = [mesh_filter, openmc.ParticleFilter("photon")]
        flux_tally_photons.scores = ["flux"]
        flux_tally_photons.id = 56
        tallies.append(flux_tally_photons)

    settings = openmc.Settings()
    settings.batches = batches_number
    settings.particles = particles_per_batch
    settings.run_mode = "fixed source"
    src.particles = list(particle_types)  
    settings.source = src
    settings.photon_transport = True
    settings.output = {'tallies': True}

    model.settings = settings
    model.tallies = tallies
    model.export_to_xml()

    def plot_mesh_tally_and_weight_window(
        statepoint_filename: str, 
        weight_window_filename: str, 
        image_filename: str,
        particle_type: str = 'neutron'
    ) -> None:

        with openmc.StatePoint(statepoint_filename) as sp:
            flux_tally = sp.get_tally(name=f"flux_tally_{particle_type}")

        tally_mesh = flux_tally.find_filter(openmc.MeshFilter).mesh
        tally_mesh_extent = tally_mesh.bounding_box.extent['xy']

        flux_mean_xy = flux_tally.get_reshaped_data(value='mean', expand_dims=True).squeeze()[:,:,int(mesh.dimension[2]/2)]
        flux_std_dev_xy = flux_tally.get_reshaped_data(value='std_dev', expand_dims=True).squeeze()[:,:,int(mesh.dimension[2]/2)]
        
        flux_rel_err_xy = np.divide(flux_std_dev_xy, flux_mean_xy, out=np.zeros_like(flux_std_dev_xy), where=flux_mean_xy!=0)
        flux_rel_err_xy[flux_rel_err_xy == 0.0] = np.nan
        
        flux_mean_xz = flux_tally.get_reshaped_data(value='mean', expand_dims=True).squeeze()[:,int(mesh.dimension[1]/2),:]
        flux_std_dev_xz = flux_tally.get_reshaped_data(value='std_dev', expand_dims=True).squeeze()[:,int(mesh.dimension[1]/2),:]   
        flux_rel_err_xz = np.divide(flux_std_dev_xz, flux_mean_xz, out=np.zeros_like(flux_std_dev_xz), where=flux_mean_xz!=0)
        flux_rel_err_xz[flux_rel_err_xz == 0.0] = np.nan

        # get slice of ww lower bounds
        wws=openmc.hdf5_to_wws(weight_window_filename)
        if particle_type == 'photon':
            ww = wws[1]  
        else:
            ww = wws[0]  
        ww_mesh = ww.mesh  # get the mesh that the weight window is mapped on
        ww_mesh_extent = ww_mesh.bounding_box.extent['xy']
        reshaped_ww_vals = ww.lower_ww_bounds.reshape(mesh.dimension)

        # slice on XZ basis, midplane Y axis
        slice_of_ww_xy = reshaped_ww_vals[:,:,int(mesh.dimension[1]/2)]
        slice_of_ww_xz = reshaped_ww_vals[:,int(mesh.dimension[1]/2),:]

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        def add_colourbar(ax, im):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            return fig.colorbar(im, cax=cax)

        # add slice of flux to subplots
        im_flux = axes[0, 0].imshow(
            flux_mean_xy.T,
            extent=tally_mesh_extent,
            norm=LogNorm(vmin=np.min(flux_mean_xy[flux_mean_xy>0]), vmax=np.max(flux_mean_xy[flux_mean_xy>0]))
        )
        axes[0, 0].set_title("Flux Mean")
        axes[0, 0].set_xlabel("X [cm]")
        axes[0, 0].set_ylabel("Y [cm]")
        add_colourbar(axes[0, 0], im_flux)

        # add slice of flux std dev to subplots
        im_std_dev = axes[0, 1].imshow(
            flux_rel_err_xy.T,
            extent=tally_mesh_extent,
            vmin=0.0,
            vmax=1.0,
            cmap='RdYlGn_r'
        )
        axes[0, 1].set_title("Flux Mean rel. error")
        axes[0, 1].set_xlabel("X [cm]")
        axes[0, 1].set_ylabel("Y [cm]")
        add_colourbar(axes[0, 1], im_std_dev)

        im_ww_lower = axes[0, 2].imshow(
            slice_of_ww_xy.T,
            extent=ww_mesh_extent,
            norm=LogNorm(vmin=np.min(slice_of_ww_xy[slice_of_ww_xy>0]), vmax=np.max(slice_of_ww_xy[slice_of_ww_xy>0])),
        )
        axes[0, 2].set_xlabel("X [cm]")
        axes[0, 2].set_ylabel("Y [cm]")
        axes[0, 2].set_title("WW lower bound")
        add_colourbar(axes[0, 2], im_ww_lower)

        im_flux_xz = axes[1, 0].imshow(
            flux_mean_xz.T,
            extent=ww_mesh.bounding_box.extent['xz'],
            norm=LogNorm(vmin=np.min(flux_mean_xz[flux_mean_xz>0]), vmax=np.max(flux_mean_xz[flux_mean_xz>0]))
        )
        axes[1, 0].set_title("Flux Mean")
        axes[1, 0].set_xlabel("X [cm]")
        axes[1, 0].set_ylabel("Z [cm]")
        add_colourbar(axes[1, 0], im_flux_xz)

        im_std_dev_xz = axes[1, 1].imshow(
            flux_rel_err_xz.T,
            extent=ww_mesh.bounding_box.extent['xz'],
            vmin=0.0,
            vmax=1.0,
            cmap='RdYlGn_r'
        )
        axes[1, 1].set_title("Flux Mean rel. error")
        axes[1, 1].set_xlabel("X [cm]")
        axes[1, 1].set_ylabel("Z [cm]")

        add_colourbar(axes[1, 1], im_std_dev_xz)
        im_ww_lower_xz = axes[1, 2].imshow(
            slice_of_ww_xz.T,
            extent=ww_mesh.bounding_box.extent['xz'],
            norm=LogNorm(vmin=np.min(slice_of_ww_xz[slice_of_ww_xz>0]), vmax=np.max(slice_of_ww_xz[slice_of_ww_xz>0])),
        )
        axes[1, 2].set_xlabel("X [cm]")
        axes[1, 2].set_ylabel("Z [cm]")
        axes[1, 2].set_title("WW lower bound")
        add_colourbar(axes[1, 2], im_ww_lower_xz)

        plt.tight_layout()
        plt.savefig(image_filename + f'_{particle_type}.png')
        plt.close()

    with openmc.lib.run_in_memory():
        tally_neutron = openmc.lib.tallies[55]
        tally_photon = openmc.lib.tallies[56]
        wws = openmc.lib.WeightWindows.from_tally(tally_neutron, particle="neutron", )
        wws_photon = openmc.lib.WeightWindows.from_tally(tally_photon, particle='photon')
        
        for i in range(1, num_iterations + 1):
            openmc.lib.run()
            wws.update_magic(tally_neutron)
            wws_photon.update_magic(tally_photon)
            openmc.lib.export_weight_windows(filename=f'weight_windows{i}.h5')
            openmc.lib.statepoint_write(filename=f'statepoint_itteration_{i}.h5')
            openmc.lib.settings.weight_windows_on = True

            plot_mesh_tally_and_weight_window(
                f'statepoint_itteration_{i}.h5',
                f'weight_windows{i}.h5',
                f'ww_{i}',
                particle_type='neutron'
            )
            plot_mesh_tally_and_weight_window(
                f'statepoint_itteration_{i}.h5',
                f'weight_windows{i}.h5',
                f'ww_{i}',
                particle_type='photon'
            )
            if rm_intermediate_files :
                remove_previous_results()