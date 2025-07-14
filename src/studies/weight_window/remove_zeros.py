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
ww = openmc.hdf5_to_wws("weight_windows_w_zeros.h5")  


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
            positive_values = current_slice[current_slice > 0]
            if positive_values.size > 0:
                min_nonzero = np.min(positive_values)
                current_slice[current_slice <= 0] = min_nonzero
                wwg.lower_ww_bounds[:, :, :, index_energy] = current_slice
            # Repeat for upper bounds
            current_slice_upper = wwg.upper_ww_bounds[:, :, :, index_energy]
            positive_values_upper = current_slice_upper[current_slice_upper > 0]
            if positive_values_upper.size > 0:
                min_nonzero_upper = np.min(positive_values_upper)
                current_slice_upper[current_slice_upper <= 0] = min_nonzero_upper
                wwg.upper_ww_bounds[:, :, :, index_energy] = current_slice_upper
    return weight_windows

ww_with_zeros_removed = remove_zeros_from_ww(weight_windows=ww)