import openmc
import os 
import json
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
import sys 
from PIL import Image
import numpy as np
from copy import deepcopy

CWD = Path(__file__).parent.resolve() 
project_root = Path(__file__).resolve().parents[2]  
sys.path.append(str(project_root))

from src.utils.weight_window.weight_window import *

# fonction material pas de fission

# ww = openmc.hdf5_to_wws("weight_windows.h5")  
# shape_ww = get_ww_size(weight_windows=ww)
# ww_corrected = apply_correction_ww(ww=ww, correction_weight_window=create_correction_ww_tally(nx=30, ny=30, nz=30,
#                                                                                               lower_left=(-700, -700, -700),
#                                                                                               upper_right=(700, 700, 700),
#                                                                                               target=np.array([-600, 0, 0]),
#                                                                                               beta=10.0, factor_div=200e3))

# plot_weight_window(weight_window=ww_corrected[1], index_coord=15, energy_index=0, saving_fig=True, plane="xy", particle_type='photon')
ww = openmc.hdf5_to_wws("weight_windows.h5")  
shape_ww = get_ww_size(weight_windows=ww)

ww_corrected_ = apply_correction_ww(ww=ww, correction_weight_window=create_correction_ww_tally(nx=30, ny=30, nz=30,
                                                                                              lower_left=(-700, -700, -700),
                                                                                              upper_right=(700, 700, 700),
                                                                                              target=np.array([-600, 200, 0]),
                                                                                              beta=50.0, factor_div=1e6))
ww_corrected_factor = apply_correction_ww(ww=ww_corrected_, correction_weight_window=np.ones(shape_ww) * 1e3)
# plot_weight_window(weight_window=ww[0], index_coord=15, energy_index=0, saving_fig=True, plane="xy", particle_type='neutron')
plot_weight_window(weight_window=ww_corrected_factor[1], index_coord=15, energy_index=0, saving_fig=True, plane="xy", particle_type='photon')