import sys
from pathlib import Path
import openmc
import numpy as np
import matplotlib.pyplot as plt

CWD = Path(__file__).parent.resolve()
project_root = Path(__file__).resolve().parents[3]  
sys.path.append(str(project_root))
from src.utils.weight_window.weight_window import plot_weight_window, create_correction_ww_tally, apply_correction_ww, create_and_apply_correction_ww_tally

# Lecture du fichier hdf5 original
ww = openmc.hdf5_to_wws("weight_windows.h5")  


def get_ww_size(weight_windows:list, particule_type:str = "neutron") -> tuple:
    """
    Get the size of the weight window for a given particle type.

    Parameters:
    weight_windows (list): List of weight window objects.
    particule_type (str): Particle type to filter (default: "neutron").

    Returns:
    tuple: Size of the weight window for the specified particle type.
    """
    for wwg in ww:
        if wwg.particle_type == particule_type:
            particle_type = wwg.particle_type
            size = wwg.lower_ww_bounds.shape[:-1]
            ww_sizes = size
    return ww_sizes