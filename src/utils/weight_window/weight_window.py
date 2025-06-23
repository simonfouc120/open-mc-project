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
