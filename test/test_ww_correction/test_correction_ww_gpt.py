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

ww = openmc.hdf5_to_wws("weight_windows.h5")  
shape_ww = get_ww_size(weight_windows=ww)

target=np.array([-600, 200, 0])

importance_map = np.ones(shape_ww)

lower_left = np.array([-700, -700, -700])
upper_right = np.array([700, 700, 700])

L = 10 


for i in range(shape_ww[0]):
    for j in range(shape_ww[1]):
        for k in range(shape_ww[2]):
            # Coordonnées du centre de maille
            x = (lower_left[0] + (i+0.5)*(upper_right[0]-lower_left[0])/shape_ww[0])
            y = (lower_left[1] + (j+0.5)*(upper_right[1]-lower_left[1])/shape_ww[1])
            z = (lower_left[2] + (k+0.5)*(upper_right[2]-lower_left[2])/shape_ww[2])
            d = np.linalg.norm([x-target[0], y-target[1], z-target[2]])
            importance_map[i,j,k] = 1 + d/L

ww_corrected_ = apply_correction_ww(ww=ww, correction_weight_window=importance_map)

ww_corrected_factor = apply_correction_ww(ww=ww_corrected_, correction_weight_window=np.ones(shape_ww) * 1e-3)

plot_weight_window(weight_window=ww_corrected_factor[0], index_coord=15, energy_index=0, saving_fig=True, plane="xy", particle_type='photon')