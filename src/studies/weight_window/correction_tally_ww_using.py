import sys
from pathlib import Path
import openmc
import numpy as np
import matplotlib.pyplot as plt

CWD = Path(__file__).parent.resolve()
project_root = Path(__file__).resolve().parents[3]  
sys.path.append(str(project_root))
from src.utils.weight_window.weight_window import plot_weight_window, correction_ww_tally

correction_weight_window = correction_ww_tally(target=np.array([0.0, 400.0, -300.0]))[0]

ww = openmc.hdf5_to_wws("weight_windows.h5")  
for wwg in ww: 
    for energy_index in range(wwg.lower_ww_bounds.shape[-1]):
        # Apply the correction to each weight window generator
        wwg.lower_ww_bounds[:,:,:,energy_index] = (correction_weight_window) * wwg.lower_ww_bounds[:,:,:,energy_index]
        wwg.upper_ww_bounds[:,:,:,energy_index] = (correction_weight_window) * wwg.upper_ww_bounds[:,:,:,energy_index]

plot_weight_window(weight_window=ww[0], index_coord=15, energy_index=0, saving_fig=False, plane="yz", particle_type='neutron')
plot_weight_window(weight_window=ww[1], index_coord=15, energy_index=0, saving_fig=False, plane="yz", particle_type='photon')