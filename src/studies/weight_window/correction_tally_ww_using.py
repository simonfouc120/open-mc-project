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

# Application de la correction à chaque groupe et énergie
ww = create_and_apply_correction_ww_tally(ww, target=np.array([0.0, 400.0, -300.0]))

plot_weight_window(weight_window=ww[0], index_coord=15, energy_index=0, saving_fig=False, plane="yz", particle_type='neutron')
plot_weight_window(weight_window=ww[1], index_coord=15, energy_index=0, saving_fig=False, plane="yz", particle_type='photon')
