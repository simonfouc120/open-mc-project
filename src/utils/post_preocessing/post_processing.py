import openmc
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
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


def load_mesh_tally(statepoint_file: object,
                    cwd: Path = Path.cwd(), 
                    name_mesh_tally:str = "flux_mesh_tally", 
                    particule_type:str='neutron',
                    bin_number:int=400, 
                    lower_left:tuple=(-10.0, -10.0), 
                    upper_right:tuple=(10.0, 10.0), 
                    zoom_x:tuple=(-10, 10), 
                    plot_error:bool=False,
                    zoom_y:tuple=(-10.0, 10.0), 
                    plane:str = "xy", 
                    saving_figure:bool = True):
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


def load_mesh_tally_fission_rate(
    cwd,
    statepoint_file: object,
    name_mesh_tally: str = "fission_rate_mesh_tally",
    particule_type: str = 'neutron',
    bin_number: int = 400,
    lower_left: tuple = (-10.0, -10.0),
    upper_right: tuple = (10.0, 10.0),
    zoom_x: tuple = (-10, 10),
    plot_error: bool = False,
    zoom_y: tuple = (-10.0, 10.0),
    plane: str = "xy",
    saving_figure: bool = True
):
    """
    Load and plot the fission rate mesh tally from the statepoint file.

    Parameters:
    - cwd: Path or directory where the mesh tally image will be saved.
    - statepoint_file: The OpenMC statepoint file object containing the tally data.
    - name_mesh_tally: Name of the tally (default is "fission_rate_mesh_tally").
    - bin_number: Number of bins for the mesh tally in each dimension (default is 400).
    - lower_left: Tuple specifying the lower left corner of the mesh (default is (-10.0, -10.0)).
    - upper_right: Tuple specifying the upper right corner of the mesh (default is (10.0, 10.0)).
    - zoom_x: Tuple specifying the x-axis limits for zooming (default is (-10, 10)).
    - zoom_y: Tuple specifying the y-axis limits for zooming (default is (-10.0, 10.0)).
    - plane: The plane of the mesh ('xy', 'xz', or 'yz') (default is 'xy').
    - saving_figure: Boolean indicating whether to save the figure (default is True).
    - plot_error: Boolean indicating whether to plot the error (default is False).

    This function extracts the fission rate mesh tally from the statepoint file, reshapes the data,
    and plots it using matplotlib with a logarithmic color scale. The resulting plot
    is saved as a PNG file in the specified directory and displayed.
    """
    mesh_tally = statepoint_file.get_tally(name=name_mesh_tally)
    # Get only the fission rate score
    fission_rate_data = mesh_tally.get_slice(scores=['fission']).mean.reshape((bin_number, bin_number))
    fission_rate_error = mesh_tally.get_slice(scores=['fission']).std_dev.reshape((bin_number, bin_number))
    fission_rate_rel_error = np.zeros_like(fission_rate_data)
    nonzero = fission_rate_data != 0
    fission_rate_rel_error[nonzero] = fission_rate_error[nonzero] / fission_rate_data[nonzero]

    if plot_error:
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        # Plot fission_rate_data
        im0 = axs[0].imshow(
            fission_rate_data,
            origin='lower',
            extent=[lower_left[0], upper_right[1], lower_left[1], upper_right[1]],
            cmap='plasma',
            norm=LogNorm(vmin=np.min(fission_rate_data[nonzero]), vmax=fission_rate_data.max())
        )
        axs[0].set_title(f"Fission rate map {plane.upper()} {particule_type}")
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
        fig.colorbar(im0, ax=axs[0], label="Fission rate [1/source] (log scale)")

        # Plot fission_rate_rel_error
        im1 = axs[1].imshow(
            fission_rate_rel_error,
            origin='lower',
            extent=[lower_left[0], upper_right[1], lower_left[1], upper_right[1]],
            cmap='plasma'
        )
        axs[1].set_title(f"Fission rate error map {plane.upper()} {particule_type}")
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
        fig.colorbar(im1, ax=axs[1], label="Fission rate relative error")

    else:
        plt.figure(figsize=(8, 6))
        im0 = plt.imshow(
            fission_rate_data,
            origin='lower',
            extent=[lower_left[0], upper_right[1], lower_left[1], upper_right[1]],
            cmap='plasma',
            norm=LogNorm(vmin=np.min(fission_rate_data[nonzero]), vmax=fission_rate_data.max())
        )
        plt.title(f"Fission rate map {plane.upper()} {particule_type}")
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
        plt.colorbar(im0, label="Fission rate [1/source] (log scale)")

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


def load_mesh_tally_dose(statepoint_file: object, cwd: Path = Path.cwd(), name_mesh_tally:str = "flux_mesh_neutrons_dose_xy", 
                        particles_per_second:int=1, particule_type:str='neutrons',
                        bin_number:int=400, lower_left:tuple=(-10.0, -10.0), 
                        upper_right:tuple=(10.0, 10.0), zoom_x:tuple=(-10, 10), 
                        zoom_y:tuple=(-10.0, 10.0), plane:str = "xy", saving_figure:bool = True, 
                        mesh_bin_volume:float=1.0, plot_error:bool=False, radiological_area: bool = False,
                        suffix_saving: str = ""):
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
    flux_data = flux_data * particles_per_second * 1e-6 * 3600 / mesh_bin_volume  # Convert to dose rate from pSv/s to µSv/h per mesh bin volume
    flux_error = mesh_tally.std_dev.reshape((bin_number, bin_number))
    flux_error = (flux_error * particles_per_second * 1e-6 * 3600 / mesh_bin_volume) / flux_data
    
    if radiological_area:
        cmap = ListedColormap(AREAS_COLORS)
        norm = BoundaryNorm(DOSE_AREAS_LIMIT, ncolors=len(AREAS_COLORS))
        label_flux = "Radiological area"
    else :
        cmap = 'plasma'
        norm = LogNorm(vmin=np.min(flux_data[flux_data != 0]), vmax=flux_data.max())
        label_flux = "Dose rate [µSv/h]"
    
    if plot_error:
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        # Plot flux_data
        im0 = axs[0].imshow(
            flux_data,
            origin='lower',
            extent=[lower_left[0], upper_right[1], lower_left[1], upper_right[1]],
            cmap=cmap,
            norm=norm
        )
        axs[0].set_title(f"Dose map {plane.upper()} {particule_type}")
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
        fig.colorbar(im0, ax=axs[0], label=label_flux)

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
            cmap=cmap,
            norm=norm
        )
        plt.title(f"Dose map {plane.upper()} {particule_type}")
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
    name_mesh_tally_saving = name_mesh_tally + suffix_saving + ".png"
    if saving_figure:
        plt.savefig(cwd / name_mesh_tally_saving)
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


class Pulse_height_tally: 
    def __init__(self, name: str):
        """
        Initialize the Pulse_height_tally class with a name.
        
        Parameters:
        - name: Name of the pulse height tally.
        """
        self.name = name
    
    def get_spectrum(self, statepoint_file, normalize: bool = True):
        tally = statepoint_file.get_tally(name=self.name)
        spectrum = tally.mean.flatten()
        spectrum_std = tally.std_dev.flatten()
        if normalize:
            spectrum /= np.sum(spectrum)
        spectrum_std /= np.sum(spectrum) if np.sum(spectrum) != 0 else 0.0
        return (spectrum, spectrum_std)
    
    def get_efficiency(self, statepoint_file):
        """
        Get the efficiency of the pulse height tally from the statepoint file.
        
        Parameters:
        - statepoint_file: The OpenMC statepoint file object containing the tally data.
        - normalize: Boolean indicating whether to normalize the efficiency (default is True).
        
        Returns:
        - Efficiency values from the pulse height tally.
        """
        tally = statepoint_file.get_tally(name=self.name)
        efficiency = sum(tally.get_values(scores=['pulse-height-efficiency']).flatten())
        return efficiency
    
class mesh_tally_data:
    def __init__(self, statepoint, name_mesh_tally, plane, bin_number, lower_left, upper_right):
        self.mesh_tally = statepoint.get_tally(name=name_mesh_tally)
        self.plane = plane
        self.bin_number = bin_number
        self.lower_left = lower_left
        self.upper_right = upper_right

    @property
    def get_coordinates(self):
        """
        Generate coordinates for mesh bins.
        
        Parameters
        ----------
        bin_number : int
            Number of bins in each dimension
        lower_left : tuple
            (x, y) coordinates of the lower left corner
        upper_right : tuple
            (x, y) coordinates of the upper right corner
            
        Returns
        -------
        numpy.ndarray
            Array of coordinates for each bin position
        """
        coord_axis_one = np.linspace(self.lower_left[0], self.upper_right[0], self.bin_number)
        coord_axis_two = np.linspace(self.lower_left[1], self.upper_right[1], self.bin_number)
        coords = (coord_axis_one, coord_axis_two)
        return coords
    
    @property
    def get_flux_data(self):
        return self.mesh_tally.mean.reshape((self.bin_number, self.bin_number))

    @property
    def get_flux_error(self):
        return self.mesh_tally.std_dev.reshape((self.bin_number, self.bin_number))

    def plot_flux(self, 
                  axis_one_index=None, 
                  axis_two_index=None, 
                  x_lim:tuple=None, 
                  y_lim:tuple=None,
                  save_fig:bool=False, 
                  fig_name:str="flux_plot.png"):

        if x_lim is None:
            x_lim = (self.get_coordinates[0][0], self.get_coordinates[0][-1])
        coords = self.get_coordinates
        plane = self.plane

        if axis_one_index is not None:
            plt.errorbar(coords[0], self.get_flux_data[:, axis_one_index], yerr= self.get_flux_error[:, axis_one_index],
                 fmt='o-', color='blue', ecolor='red', capsize=3, markersize=2,
                 label='Flux values')
            plt.xlabel(f'{plane[1].upper()} [cm]')
            plt.legend()
            plt.ylabel('Flux [n/cm^2/P-source]')
            plt.title(f'Flux=f({plane[0].lower()}) at {plane[0].lower()}={coords[1][axis_one_index]:.2f} cm')
            plt.yscale('log')
            plt.grid()
            if y_lim is not None:
                plt.ylim(y_lim)
            if x_lim is not None:
                plt.xlim(x_lim)
            if save_fig:
                plt.savefig(fig_name, dpi=300)
            plt.show()

        if axis_two_index is not None:
            plt.errorbar(coords[0], self.get_flux_data[axis_two_index, :], yerr= self.get_flux_error[axis_two_index, :],
                 fmt='o-', color='blue', ecolor='red', capsize=3, markersize=2,
                 label='Flux values')
            plt.xlabel(f'{plane[0].upper()} [cm]')
            plt.legend()
            plt.ylabel('Flux [n/cm^2/P-source]')
            plt.title(f'Flux=f({plane[1].upper()}) at {plane[1].lower()}={coords[1][axis_two_index]:.2f} cm')
            plt.yscale('log')
            plt.grid()
            if y_lim is not None:
                plt.ylim(y_lim)
            if x_lim is not None:
                plt.xlim(x_lim)
            if save_fig:
                plt.savefig(fig_name, dpi=300)
            plt.show()

    def plot_dose(self, 
                  particles_per_second:int=1, 
                  particule_type:str='neutrons',
                  mesh_bin_volume:float=1.0, 
                  axis_one_index=None, 
                  axis_two_index=None, 
                  x_lim:tuple=None, 
                  y_lim:tuple=None,
                  save_fig:bool=False, 
                  fig_name:str="dose_plot.png",
                  radiological_area: bool = False,
                  log_scale: bool = True):
        if x_lim is None:
            x_lim = (self.get_coordinates[0][0], self.get_coordinates[0][-1])
        if y_lim is None:
            y_lim = (0.1, None)
        coords = self.get_coordinates
        plane = self.plane

        dose_data = self.get_flux_data * particles_per_second * 1e-6 * 3600 / mesh_bin_volume
        dose_error = self.get_flux_error * particles_per_second * 1e-6 * 3600 / mesh_bin_volume

        def plot_radiological_areas():
            for i in range(len(DOSE_AREAS_LIMIT) - 1):
                plt.fill_betweenx(
                    y=[DOSE_AREAS_LIMIT[i], DOSE_AREAS_LIMIT[i + 1]],
                    x1=coords[0][0],
                    x2=coords[0][-1],
                    color=AREAS_COLORS[i],
                    alpha=0.3,
                    label=[
                    'Free area', 'Supervised area', 'Controlled area',
                    'High controlled area', 'Very high controlled area', 'Extremely high controlled area'
                    ][i]
            )

        def finalize_plot(xlabel, ylabel, title):
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            if log_scale:
                plt.yscale('log')
            if x_lim is not None:
                plt.xlim(x_lim)
            if y_lim is not None:
                plt.ylim(y_lim)
            plt.grid()
            if save_fig:
                plt.savefig(fig_name, dpi=300, bbox_inches='tight')
            plt.show()

        if radiological_area:
            plot_radiological_areas()

        if axis_one_index is not None:
            plt.errorbar(
            coords[0], dose_data[:, axis_one_index], yerr=dose_error[:, axis_one_index],
            fmt='o-', color='blue', ecolor='red', capsize=3, markersize=2, label='Dose values'
            )
            if radiological_area:
                legend = plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0., frameon=True)
                legend.get_frame().set_edgecolor('black')
                legend.get_frame().set_linewidth(1.5)
            else:
                plt.legend()
            finalize_plot(
            f'{plane[0].upper()} [cm]',
            "Dose rate [µSv/h]",
            f'Dose {particule_type} = f({plane[0].lower()}) at {plane[1].lower()}={coords[1][axis_one_index]:.2f} cm'
            )

        if axis_two_index is not None:
            plt.errorbar(
            coords[1], dose_data[axis_two_index, :], yerr=dose_error[axis_two_index, :],
            fmt='o-', color='blue', ecolor='red', capsize=3, markersize=2, label='Dose values'
            )
            if radiological_area:
                legend = plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0., frameon=True)
                legend.get_frame().set_edgecolor('black')
                legend.get_frame().set_linewidth(1.5)
            else:
                plt.legend()
            finalize_plot(
            f'{plane[1].upper()} [cm]',
            "Dose rate [µSv/h]",
            f'Dose {particule_type} = f({plane[1].upper()}) at {plane[0].lower()}={coords[0][axis_two_index]:.2f} cm'
            )


def flux_over_geometry(model, statepoint_file: object,
                    cwd: Path = Path.cwd(), 
                    name_mesh_tally:str = "flux_mesh_tally", 
                    particule_type:str='neutron',
                    bin_number:int=400, 
                    lower_left:tuple=(-10.0, -10.0), 
                    upper_right:tuple=(10.0, 10.0), 
                    zoom_x:tuple=(-10, 10), 
                    plot_error:bool=False,
                    zoom_y:tuple=(-10.0, 10.0), 
                    plane:str = "xy", 
                    plane_coord:float=0.0,
                    pixels_model_geometry:int=1_000_000,
                    suffix_saving: str = "",
                    color_by: str = "material",
                    saving_figure:bool = True):
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
    if plane not in ["xy", "xz", "yz"]:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")

    mesh_tally = statepoint_file.get_tally(name=name_mesh_tally)
    flux_data = mesh_tally.mean.reshape((bin_number, bin_number))
    std_dev_data = mesh_tally.std_dev.reshape((bin_number, bin_number))

    # Calculate relative error, avoiding division by zero
    relative_error = np.zeros_like(flux_data)
    nonzero_flux = flux_data != 0
    relative_error[nonzero_flux] = std_dev_data[nonzero_flux] / flux_data[nonzero_flux]

    mesh = mesh_tally.find_filter(openmc.MeshFilter).mesh
    extent = mesh.bounding_box.extent[plane]

    # Common settings for model.plot
    plot_kwargs = {
        'outline': "only",
        'axes': None,  # This will be set per-plot
        'pixels': pixels_model_geometry,
        'color_by': color_by,
        'origin': ((lower_left[0] + upper_right[0]) / 2, (lower_left[1] + upper_right[1]) / 2, plane_coord),
        'width': (upper_right[0] - lower_left[0], upper_right[1] - lower_left[1])
    }

    def setup_ax(ax, title):
        ax.set_title(title)
        ax.set_xlabel(f'{plane[0].upper()} [cm]')
        ax.set_ylabel(f'{plane[1].upper()} [cm]')
        ax.set_xlim(zoom_x)
        ax.set_ylim(zoom_y)

    if plot_error:
        fig, (ax_flux, ax_error) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot flux data
        norm_flux = LogNorm(vmin=np.min(flux_data[nonzero_flux]), vmax=flux_data.max())
        im_flux = ax_flux.imshow(flux_data, origin='lower', extent=extent, cmap='plasma', norm=norm_flux)
        plot_kwargs['axes'] = ax_flux
        model.plot(**plot_kwargs)
        setup_ax(ax_flux, f"Flux map {plane.upper()} {particule_type}")
        fig.colorbar(im_flux, ax=ax_flux, label="Flux [p/p-source] (log scale)")

        # Plot relative error
        im_error = ax_error.imshow(relative_error, origin='lower', extent=extent, cmap='plasma')
        plot_kwargs['axes'] = ax_error
        model.plot(**plot_kwargs)
        setup_ax(ax_error, f"Flux error map {plane.upper()} {particule_type}")
        fig.colorbar(im_error, ax=ax_error, label="Relative Error")

    else:
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot flux data
        norm_flux = LogNorm(vmin=np.min(flux_data[nonzero_flux]), vmax=flux_data.max())
        im_flux = ax.imshow(flux_data, origin='lower', extent=extent, cmap='plasma', norm=norm_flux)
        plot_kwargs['axes'] = ax
        model.plot(**plot_kwargs)
        setup_ax(ax, f"Flux map {plane.upper()} {particule_type}")
        fig.colorbar(im_flux, ax=ax, label="Flux [p/p-source] (log scale)")

    plt.tight_layout()
    name_mesh_tally_saving = f"{name_mesh_tally}{suffix_saving}.png"
    if saving_figure:
        plt.savefig(cwd / name_mesh_tally_saving)
    plt.show()



def dose_over_geometry(model, statepoint_file: object,
                    cwd: Path = Path.cwd(), 
                    name_mesh_tally:str = "flux_mesh_neutrons_dose_xy", 
                    particule_type:str='neutrons',
                    particles_per_second:int=1,
                    bin_number:int=400, 
                    lower_left:tuple=(-10.0, -10.0), 
                    upper_right:tuple=(10.0, 10.0), 
                    zoom_x:tuple=(-10, 10), 
                    zoom_y:tuple=(-10.0, 10.0), 
                    plane:str = "xy", 
                    plane_coord:float=0.0,
                    pixels_model_geometry:int=1_000_000,
                    mesh_bin_volume:float=1.0, 
                    radiological_area: bool = False,
                    suffix_saving: str = "",
                    color_by: str = "material",
                    saving_figure:bool = True,
                    plot_error:bool=False):
    """
    Load and plot the dose mesh tally from the statepoint file.

    Parameters:
    - cwd: Path or directory where the mesh tally image will be saved.
    - statepoint_file: The OpenMC statepoint file object containing the tally data.
    - name_mesh_tally_saving: Name of the tally (default is "mesh_tally.png").
    - bin_number: Number of bins for the mesh tally in each dimension (default is 400).
    - lower_left: Tuple specifying the lower left corner of the mesh (default is (-10.0, -10.0)).
    - upper_right: Tuple specifying the upper right corner of the mesh (default is (10.0, 10.0)).
    - zoom_x: Tuple specifying the x-axis limits for zooming (default is (-10, 10)).
    - zoom_y: Tuple specifying the y-axis limits for zooming (default is (-10.0, 10.0)).

    This function extracts the dose mesh tally from the statepoint file, reshapes the data,
    and plots it using matplotlib with a logarithmic color scale. The resulting plot
    is saved as a PNG file in the specified directory and displayed.
    """
    if plane not in ["xy", "xz", "yz"]:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")

    mesh_tally = statepoint_file.get_tally(name=name_mesh_tally)
    flux_data = mesh_tally.mean.reshape((bin_number, bin_number))
    flux_data = flux_data * particles_per_second * 1e-6 * 3600 / mesh_bin_volume  # Convert to dose rate from pSv/s to µSv/h per mesh bin volume
    flux_error = mesh_tally.std_dev.reshape((bin_number, bin_number))
    flux_error = (flux_error * particles_per_second * 1e-6 * 3600 / mesh_bin_volume) / flux_data
    relative_error = np.zeros_like(flux_data)
    nonzero_flux = flux_data != 0
    relative_error[nonzero_flux] = flux_error[nonzero_flux]

    mesh = mesh_tally.find_filter(openmc.MeshFilter).mesh
    extent = mesh.bounding_box.extent[plane]
    # Common settings for model.plot
    plot_kwargs = {
        'outline': "only",
        'basis': plane,
        'axes': None,  # This will be set per-plot
        'pixels': pixels_model_geometry,
        'color_by': color_by,
        'origin': ((lower_left[0] + upper_right[0]) / 2, (lower_left[1] + upper_right[1]) / 2, plane_coord),    
        'width': (upper_right[0] - lower_left[0], upper_right[1] - lower_left[1])
    }
    def setup_ax(ax, title):
        ax.set_title(title)
        ax.set_xlabel(f'{plane[0].upper()} [cm]')
        ax.set_ylabel(f'{plane[1].upper()} [cm]')
        ax.set_xlim(zoom_x)
        ax.set_ylim(zoom_y)
    if radiological_area:
        cmap = ListedColormap(AREAS_COLORS)
        norm = BoundaryNorm(DOSE_AREAS_LIMIT, ncolors=len(AREAS_COLORS))
        label_flux = "Radiological area"
    else :
        cmap = 'plasma'
        norm = LogNorm(vmin=np.min(flux_data[nonzero_flux]), vmax=flux_data.max())
        label_flux = "Dose rate [µSv/h]"
    if plot_error:
        fig, (ax_flux, ax_error) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot flux data
        im_flux = ax_flux.imshow(flux_data, origin='lower', extent=extent, cmap=cmap, norm=norm)
        plot_kwargs['axes'] = ax_flux
        model.plot(**plot_kwargs)
        setup_ax(ax_flux, f"Dose map {plane.upper()} {particule_type}")
        fig.colorbar(im_flux, ax=ax_flux, label=label_flux)

        # Plot relative error
        im_error = ax_error.imshow(relative_error, origin='lower', extent=extent, cmap='plasma')
        plot_kwargs['axes'] = ax_error
        model.plot(**plot_kwargs)
        setup_ax(ax_error, f"Dose error map {plane.upper()} {particule_type}")
        fig.colorbar(im_error, ax=ax_error, label="Relative Error")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot flux data
        im_flux = ax.imshow(flux_data, origin='lower', extent=extent, cmap=cmap, norm=norm)
        plot_kwargs['axes'] = ax
        model.plot(**plot_kwargs)
        setup_ax(ax, f"Dose map {plane.upper()} {particule_type}")
        fig.colorbar(im_flux, ax=ax, label=label_flux)
    plt.tight_layout()
    name_mesh_tally_saving = f"{name_mesh_tally}{suffix_saving}.png"
    if saving_figure:
        plt.savefig(cwd / name_mesh_tally_saving)
    plt.show()