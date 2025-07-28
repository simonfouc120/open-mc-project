import openmc
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

# TODO : à terme, il faudrait utiliser les constantes définies dans lib/constants/constant.py
# Je dois remonter le chemin pour l'importer correctement
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[3]))  # Adjust path to
from lib.constants.constant import DOSE_AREAS_LIMIT, AREAS_COLORS

def print_calculation_finished():
    """
    Print a message indicating that the calculation has finished.
    """
    print("######################################################")
    print("Calculation finished successfully.")  # TODO: Implement a more sophisticated logging mechanism if needed.
    print("######################################################")


def load_mesh_tally(cwd, statepoint_file: object, name_mesh_tally:str = "flux_mesh_tally", particule_type:str='neutron',
                    bin_number:int=400, lower_left:tuple=(-10.0, -10.0), 
                    upper_right:tuple=(10.0, 10.0), zoom_x:tuple=(-10, 10), plot_error:bool=False,
                    zoom_y:tuple=(-10.0, 10.0), plane:str = "xy", saving_figure:bool = True):
    """
    Load and plot the mesh tally from the statepoint file.

    Parameters:
    - cwd: Path or directory where the mesh tally image will be saved.
    - statepoint_file: The OpenMC statepoint file object containing the tally data.
    - name_mesh_tally_saving: Name of the tally (default is "mesh_tally.png").
    - bin_number: Number of bins for the mesh tally in each dimension (default is 400).
    - lower_left: Tuple specifying the lower left corner of the mesh (default is (-10.0, -10.0)).
    - upper_right: Tuple specifying the upper right corner of the mesh (default is (10.0, 10.0)).
    - zoom_x: Tuple specifying the x-axis limits for zooming (default is (-10, 10)).
    - zoom_y: Tuple specifying the y-axis limits for zooming (default is (-10.0, 10.0)).

    This function extracts the mesh tally from the statepoint file, reshapes the data,
    and plots it using matplotlib with a logarithmic color scale. The resulting plot
    is saved as a PNG file in the specified directory and displayed.
    """
    mesh_tally = statepoint_file.get_tally(name=name_mesh_tally)
    flux_data = mesh_tally.mean.reshape((bin_number, bin_number))
    flux_error = mesh_tally.std_dev.reshape((bin_number, bin_number))
    flux_error = flux_error / flux_data
    if plot_error:
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        # Plot flux_data
        im0 = axs[0].imshow(
            flux_data,
            origin='lower',
            extent=[lower_left[0], upper_right[1], lower_left[1], upper_right[1]],
            cmap='plasma',
            norm=LogNorm(vmin=np.min(flux_data[flux_data != 0]), vmax=flux_data.max())
        )
        axs[0].set_title(f"Flux map {plane.upper()} {particule_type}")
        if plane == "xy":
            axs[0].set_xlabel('X [cm]')
            axs[0].set_ylabel('Y [cm]')
        elif plane == "xz":
            axs[0].set_xlabel('X [cm]')
            axs[0].set_ylabel('Z [cm]')
        elif plane == "yz":
            axs[0].set_xlabel('Y [cm]')
            axs[0].set_ylabel('Z [cm]')
        else:
            raise ValueError("plane must be 'xy', 'xz', or 'yz'")
        axs[0].set_xlim(zoom_x[0], zoom_x[1])
        axs[0].set_ylim(zoom_y[0], zoom_y[1])
        fig.colorbar(im0, ax=axs[0], label="Flux [p/p-source] (log scale)")

        # Plot flux_error
        im1 = axs[1].imshow(
            flux_error,
            origin='lower',
            extent=[lower_left[0], upper_right[1], lower_left[1], upper_right[1]],
            cmap='plasma')
        axs[1].set_title(f"Flux error map {plane.upper()} {particule_type}")
        if plane == "xy":
            axs[1].set_xlabel('X [cm]')
            axs[1].set_ylabel('Y [cm]')
        elif plane == "xz":
            axs[1].set_xlabel('X [cm]')
            axs[1].set_ylabel('Z [cm]')
        elif plane == "yz":
            axs[1].set_xlabel('Y [cm]')
            axs[1].set_ylabel('Z [cm]')
        axs[1].set_xlim(zoom_x[0], zoom_x[1])
        axs[1].set_ylim(zoom_y[0], zoom_y[1])
        fig.colorbar(im1, ax=axs[1], label="Flux error")

    else: 
        plt.figure(figsize=(8, 6))
        im0 = plt.imshow(
            flux_data,
            origin='lower',
            extent=[lower_left[0], upper_right[1], lower_left[1], upper_right[1]],
            cmap='plasma',
            norm=LogNorm(vmin=np.min(flux_data[flux_data != 0]), vmax=flux_data.max())
        )
        plt.title(f"Flux map {plane.upper()} {particule_type}")
        if plane == "xy":
            plt.xlabel('X [cm]')
            plt.ylabel('Y [cm]')
        elif plane == "xz":
            plt.xlabel('X [cm]')
            plt.ylabel('Z [cm]')
        elif plane == "yz":
            plt.xlabel('Y [cm]')
            plt.ylabel('Z [cm]')
        else:
            raise ValueError("plane must be 'xy', 'xz', or 'yz'")
        plt.xlim(zoom_x[0], zoom_x[1])
        plt.ylim(zoom_y[0], zoom_y[1])
        plt.colorbar(im0, label="Flux [p/p-source] (log scale)")

    plt.tight_layout()
    name_mesh_tally_saving = name_mesh_tally + ".png"
    if saving_figure:
        plt.savefig(cwd / name_mesh_tally_saving)
    plt.show()

def load_dammage_energy_tally(cwd, statepoint_file: object, name_mesh_tally:str = "dammage_energy_mesh", 
                                bin_number:int=500, lower_left:tuple=(-10.0, -10.0), 
                                upper_right:tuple=(10.0, 10.0), zoom_x:tuple=(-10, 10), 
                                zoom_y:tuple=(-10.0, 10.0), plane:str = "xy", saving_figure:bool = True):
    """
    Load and plot the damage energy mesh tally from the statepoint file.

    Parameters:
    - cwd: Path or directory where the mesh tally image will be saved.
    - statepoint_file: The OpenMC statepoint file object containing the tally data.
    - name_mesh_tally: Name of the tally (default is "dammage_energy_mesh").
    - bin_number: Number of bins for the mesh tally in each dimension (default is 500).
    - lower_left: Tuple specifying the lower left corner of the mesh (default is (-10.0, -10.0)).
    - upper_right: Tuple specifying the upper right corner of the mesh (default is (10.0, 10.0)).
    - zoom_x: Tuple specifying the x-axis limits for zooming (default is (-10, 10)).
    - zoom_y: Tuple specifying the y-axis limits for zooming (default is (-10.0, 10.0)).
    
    This function extracts the damage energy mesh tally from the statepoint file,
    reshapes the data, and plots it using matplotlib with a logarithmic color scale.
    The resulting plot is saved as a PNG file in the specified directory and displayed.
    """
    dpa_result = statepoint_file.get_tally(name=name_mesh_tally)

    dpa_data = dpa_result.get_slice(scores=['damage-energy']).mean.ravel()
    dpa_data.shape = (500, 500)

    plt.figure(figsize=(8, 6))
    plt.imshow(dpa_data, norm=LogNorm(), extent=[lower_left[0], upper_right[1], lower_left[1], upper_right[1]], origin='lower', cmap='plasma')
    plt.title('Dammage energy Map')
    if plane == "xy":
        plt.title('Damage energy map XY')
        plt.xlabel('X [cm]')
        plt.ylabel('Y [cm]')    
    elif plane == "xz":
        plt.title('Damage energy map XZ')
        plt.xlabel('X [cm]')
        plt.ylabel('Z [cm]')
    elif plane == "yz":
        plt.title('Damage energy map YZ')
        plt.xlabel('Y [cm]')
        plt.ylabel('Z [cm]')
    else:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")
    if saving_figure:
        plt.savefig(cwd / f"{name_mesh_tally}.png")
    plt.colorbar(label='eV/p-source')
    plt.tight_layout()
    plt.show()


def load_mesh_tally_dose(cwd, statepoint_file: object, name_mesh_tally:str = "flux_mesh_neutrons_dose_xy", 
                        particles_per_second:int=1, particule_type:str='neutrons',
                        bin_number:int=400, lower_left:tuple=(-10.0, -10.0), 
                        upper_right:tuple=(10.0, 10.0), zoom_x:tuple=(-10, 10), 
                        zoom_y:tuple=(-10.0, 10.0), plane:str = "xy", saving_figure:bool = True, 
                        mesh_bin_volume:float=1.0, plot_error:bool=False):
    """
    Load and plot the dose mesh tally from the statepoint file.

    Parameters:
    - cwd: Path or directory where the mesh tally image will be saved.
    - statepoint_file: The OpenMC statepoint file object containing the tally data.
    - name_mesh_tally: Name of the tally (default is "flux_mesh_neutrons_xy").
    - bin_number: Number of bins for the mesh tally in each dimension (default is 400).
    - lower_left: Tuple specifying the lower left corner of the mesh (default is (-10.0, -10.0)).
    - upper_right: Tuple specifying the upper right corner of the mesh (default is (10.0, 10.0)).
    - zoom_x: Tuple specifying the x-axis limits for zooming (default is (-10, 10)).
    - zoom_y: Tuple specifying the y-axis limits for zooming (default is (-10.0, 10.0)).
    - plane: The plane of the mesh ('xy', 'xz', or 'yz') (default is 'xy').
    - saving_figure: Boolean indicating whether to save the figure (default is True).
    - particule_type: Type of particle for the tally (default is 'neutrons').
    - particles_per_second: Number of particles per second (default is 1). Useless if source.strengh is set in "fixed source" mode.
    - mesh_bin_volume: Volume of each mesh bin (default is 1.0).
    - plot_error: Boolean indicating whether to plot the error (default is False).
    
    This function extracts the dose mesh tally from the statepoint file,
    reshapes the data, and plots it using matplotlib with a logarithmic color scale.
    The resulting plot is saved as a PNG file in the specified directory and displayed.
    """
    mesh_tally = statepoint_file.get_tally(name=name_mesh_tally)
    flux_data = mesh_tally.mean.reshape((bin_number, bin_number))
    flux_data = flux_data * particles_per_second * 1e-6 * 3600 / mesh_bin_volume  # Convert to dose rate in µSv/h per mesh bin volume
    flux_error = mesh_tally.std_dev.reshape((bin_number, bin_number))
    flux_error = (flux_error * particles_per_second * 1e-6 * 3600 / mesh_bin_volume) / flux_data

    if plot_error:
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        # Plot flux_data
        im0 = axs[0].imshow(
            flux_data,
            origin='lower',
            extent=[lower_left[0], upper_right[1], lower_left[1], upper_right[1]],
            cmap='plasma',
            norm=LogNorm(vmin=np.min(flux_data[flux_data != 0]), vmax=flux_data.max())
        )
        axs[0].set_title(f"Flux map {plane.upper()} {particule_type}")
        if plane == "xy":
            axs[0].set_xlabel('X [cm]')
            axs[0].set_ylabel('Y [cm]')
        elif plane == "xz":
            axs[0].set_xlabel('X [cm]')
            axs[0].set_ylabel('Z [cm]')
        elif plane == "yz":
            axs[0].set_xlabel('Y [cm]')
            axs[0].set_ylabel('Z [cm]')
        else:
            raise ValueError("plane must be 'xy', 'xz', or 'yz'")
        axs[0].set_xlim(zoom_x[0], zoom_x[1])
        axs[0].set_ylim(zoom_y[0], zoom_y[1])
        fig.colorbar(im0, ax=axs[0], label="Dose rate [µSv/h]")

        # Plot flux_error
        im1 = axs[1].imshow(
            flux_error,
            origin='lower',
            extent=[lower_left[0], upper_right[1], lower_left[1], upper_right[1]],
            cmap='plasma')
        axs[1].set_title(f"Flux error map {plane.upper()} {particule_type}")
        if plane == "xy":
            axs[1].set_xlabel('X [cm]')
            axs[1].set_ylabel('Y [cm]')
        elif plane == "xz":
            axs[1].set_xlabel('X [cm]')
            axs[1].set_ylabel('Z [cm]')
        elif plane == "yz":
            axs[1].set_xlabel('Y [cm]')
            axs[1].set_ylabel('Z [cm]')
        axs[1].set_xlim(zoom_x[0], zoom_x[1])
        axs[1].set_ylim(zoom_y[0], zoom_y[1])
        fig.colorbar(im1, ax=axs[1], label="Flux error")

    else: 
        plt.figure(figsize=(8, 6))
        im0 = plt.imshow(
            flux_data,
            origin='lower',
            extent=[lower_left[0], upper_right[1], lower_left[1], upper_right[1]],
            cmap='plasma',
            norm=LogNorm(vmin=np.min(flux_data[flux_data != 0]), vmax=flux_data.max())
        )
        plt.title(f"Flux map {plane.upper()} {particule_type}")
        if plane == "xy":
            plt.xlabel('X [cm]')
            plt.ylabel('Y [cm]')
        elif plane == "xz":
            plt.xlabel('X [cm]')
            plt.ylabel('Z [cm]')
        elif plane == "yz":
            plt.xlabel('Y [cm]')
            plt.ylabel('Z [cm]')
        else:
            raise ValueError("plane must be 'xy', 'xz', or 'yz'")
        plt.xlim(zoom_x[0], zoom_x[1])
        plt.ylim(zoom_y[0], zoom_y[1])
        plt.colorbar(im0, label="Dose rate [µSv/h]")

    plt.tight_layout()
    name_mesh_tally_saving = name_mesh_tally + ".png"
    if saving_figure:
        plt.savefig(cwd / name_mesh_tally_saving)
    plt.show()

from matplotlib.colors import ListedColormap, BoundaryNorm

def load_mesh_tally_dose_areas(cwd, statepoint_file: object, name_mesh_tally: str = "flux_mesh_neutrons_dose_xy",
                         particles_per_second: int = 1, particule_type: str = 'neutrons',
                         bin_number: int = 400, lower_left: tuple = (-10.0, -10.0),
                         upper_right: tuple = (10.0, 10.0), zoom_x: tuple = (-10, 10),
                         zoom_y: tuple = (-10.0, 10.0), plane: str = "xy", saving_figure: bool = True,
                         mesh_bin_volume: float = 1.0, plot_error: bool = False):
    """
    Load and plot the dose mesh tally from the statepoint file using radiological zone colors
    defined in French regulation (Code du travail).
    """

    # Récupération des données depuis le fichier statepoint
    mesh_tally = statepoint_file.get_tally(name=name_mesh_tally)
    flux_data = mesh_tally.mean.reshape((bin_number, bin_number))
    flux_data = flux_data * particles_per_second * 1e-6 * 3600 / mesh_bin_volume  # µSv/h
    flux_error = mesh_tally.std_dev.reshape((bin_number, bin_number))
    flux_error = (flux_error * particles_per_second * 1e-6 * 3600 / mesh_bin_volume) / np.where(flux_data == 0, 1, flux_data)
    cmap = ListedColormap(AREAS_COLORS)
    norm = BoundaryNorm(DOSE_AREAS_LIMIT, ncolors=len(AREAS_COLORS))

    if plot_error:
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        # Carte des doses
        im0 = axs[0].imshow(
            flux_data,
            origin='lower',
            extent=[lower_left[0], upper_right[0], lower_left[1], upper_right[1]],
            cmap=cmap,
            norm=norm
        )
        axs[0].set_title(f"Dose map {plane.upper()} {particule_type}")
        axs[0].set_xlim(zoom_x)
        axs[0].set_ylim(zoom_y)
        axs[0].set_xlabel(f'{plane[0].upper()} [cm]')
        axs[0].set_ylabel(f'{plane[1].upper()} [cm]')
        cbar0 = fig.colorbar(im0, ax=axs[0], ticks=DOSE_AREAS_LIMIT)
        # cbar0.ax.set_yticklabels(['<7.5', '7.5–25', '25–100', '100–2000', '2k–1e5', '>1e5'])
        cbar0.set_label("Radiological area")

        # Carte des erreurs relatives
        im1 = axs[1].imshow(
            flux_error,
            origin='lower',
            extent=[lower_left[0], upper_right[0], lower_left[1], upper_right[1]],
            cmap='plasma'
        )
        axs[1].set_title(f"Relative error map {plane.upper()} {particule_type}")
        axs[1].set_xlim(zoom_x)
        axs[1].set_ylim(zoom_y)
        axs[1].set_xlabel(f'{plane[0].upper()} [cm]')
        axs[1].set_ylabel(f'{plane[1].upper()} [cm]')
        fig.colorbar(im1, ax=axs[1], label="Relative error")

    else:
        plt.figure(figsize=(8, 6))
        im0 = plt.imshow(
            flux_data,
            origin='lower',
            extent=[lower_left[0], upper_right[0], lower_left[1], upper_right[1]],
            cmap=cmap,
            norm=norm
        )
        plt.title(f"Dose map {plane.upper()} {particule_type}")
        plt.xlabel(f'{plane[0].upper()} [cm]')
        plt.ylabel(f'{plane[1].upper()} [cm]')
        plt.xlim(zoom_x)
        plt.ylim(zoom_y)
        cbar = plt.colorbar(im0, ticks=DOSE_AREAS_LIMIT)
        # cbar.ax.set_yticklabels(['<7.5', '7.5–25', '25–100', '100–2000', '2k–1e5', '>1e5'])
        cbar.set_label("Radiological area")

    plt.tight_layout()
    if saving_figure:
        plt.savefig(cwd / f"{name_mesh_tally}.png")
    plt.show()

def gaussian_energy_broadening(E, a:float=1000., b:float=4., c:float=0.0002):
    """
    Apply Gaussian energy broadening to a given energy value E.
    Parameters:
    - E: Energy value to be broadened.
    - a: Coefficient for the Gaussian broadening (default is 1000).
    - b: Coefficient for the Gaussian broadening (default is 4).
    - c: Coefficient for the Gaussian broadening (default is 0.0002
    Returns:

    - A new energy value with Gaussian broadening applied.
    """
    sigma = (a + b * (E + c * E**2)**0.5) / (2 * (2 * np.log(2))**0.5)
    return np.random.normal(loc=E, scale=sigma)
