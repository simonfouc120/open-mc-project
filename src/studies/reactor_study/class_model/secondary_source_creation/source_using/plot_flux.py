import openmc
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os


def get_coordinates_from_mesh(bin_number, lower_left, upper_right):
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
    x = np.linspace(lower_left[0], upper_right[0], bin_number)
    y = np.linspace(lower_left[1], upper_right[1], bin_number)
    
    return x, y

statepoint = openmc.StatePoint(f"statepoint.100.h5")

# mesh_tally = statepoint.get_tally(name="flux_mesh_xy_neutrons")
# flux_data = mesh_tally.mean.reshape((500, 500))
# flux_err = mesh_tally.std_dev.reshape((500, 500))

# plane = "xy"

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

    def plot_flux(self, axis_one_index=None, axis_two_index=None, lim:tuple=None):

        if lim is None:
            lim = (self.get_coordinates[0][0], self.get_coordinates[0][-1])
        coords = self.get_coordinates
        plane = self.plane
        if axis_two_index is not None:
            plt.errorbar(coords[0], self.get_flux_data[:, axis_two_index], yerr= self.get_flux_error[:, axis_two_index],
                 fmt='o-', color='blue', ecolor='red', capsize=3, markersize=2,
                 label='Flux values')
            plt.xlabel(f'{plane[0].upper()} [cm]')
            plt.legend()
            plt.ylabel('Flux [n/cm^2/P-source]')
            plt.title(f'Flux=f({plane[0].lower()}) at {plane[1].lower()}={coords[1][axis_two_index]:.2f} cm')
            plt.yscale('log')
            plt.grid()
            plt.xlim(lim)
            plt.show()

        if axis_one_index is not None:
            plt.errorbar(coords[1], self.get_flux_data[axis_one_index, :], yerr= self.get_flux_error[axis_one_index, :],
                 fmt='o-', color='blue', ecolor='red', capsize=3, markersize=2,
                 label='Flux values')
            plt.xlabel(f'{plane[1].upper()} [cm]')
            plt.legend()
            plt.ylabel('Flux [n/cm^2/P-source]')
            plt.title(f'Flux=f({plane[1].upper()}) at {plane[0].lower()}={coords[0][axis_one_index]:.2f} cm')
            plt.yscale('log')
            plt.grid()
            plt.xlim(lim)
            plt.show()


mesh_tally_neutrons = mesh_tally_data(statepoint, "flux_mesh_xy_neutrons", "xy", 500, (-850.0, -850.0), (850.0, 850.0))
flux_data = mesh_tally_neutrons.get_flux_data
flux_err = mesh_tally_neutrons.get_flux_error

coords = mesh_tally_neutrons.get_coordinates

mesh_tally_neutrons.plot_flux(axis_one_index=250, lim=(0, 850))

# plane = "xy"
# # Plot flux vs y with error bars
# axis_two_index = 250  # ou tout autre index
# plt.errorbar(coords[0], flux_data[:, axis_two_index], yerr=flux_err[:, axis_two_index],
#              fmt='o-', color='blue', ecolor='red', capsize=3, markersize=2,
#              label='Flux values')
# plt.xlabel('X [cm]')
# plt.legend()
# plt.ylabel('Flux [n/cm^2/P-source]')
# plt.title(f'Flux=f(X) at y={coords[1][axis_two_index]:.2f} cm')
# plt.yscale('log')
# plt.grid()
# plt.xlim(0, np.max(coords[1]) * 1.1)
# plt.show()

# # Plot flux vs y with error bars
# axis_one_index = 250  # ou tout autre index
# plt.errorbar(coords[1], flux_data[axis_one_index, :], yerr=flux_err[axis_one_index, :],
#              fmt='o-', color='blue', ecolor='red', capsize=3, markersize=2,
#              label='Flux values')
# plt.xlabel('Y [cm]')
# plt.legend()
# plt.ylabel('Flux [n/cm^2/P-source]')
# plt.title(f'Flux=f(Y) at X={coords[0][axis_one_index]:.2f} cm')
# plt.yscale('log')
# plt.grid()
# plt.xlim(0, np.max(coords[1]) * 1.1)
# plt.show()