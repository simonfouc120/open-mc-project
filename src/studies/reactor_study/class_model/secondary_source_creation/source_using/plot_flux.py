import openmc
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

CWD = Path(__file__).parent.resolve() 
project_root = Path(__file__).resolve().parents[6]  
sys.path.append(str(project_root))

statepoint = openmc.StatePoint(f"statepoint.100.h5")
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
                  lim:tuple=None, 
                  save_fig:bool=False, 
                  fig_name:str="flux_plot.png"):

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
            if save_fig:
                plt.savefig(fig_name, dpi=300)
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
            if save_fig:
                plt.savefig(fig_name, dpi=300)
            plt.show()

    # si dose : à diviser par le volume du voxel 

mesh_tally_neutrons = mesh_tally_data(statepoint, "flux_mesh_xy_neutrons", "xy", 500, (-850.0, -850.0), (850.0, 850.0))
mesh_tally_neutrons.plot_flux(axis_one_index=250, lim=(0, 850), save_fig=True, fig_name="flux_plot_neutrons.png")