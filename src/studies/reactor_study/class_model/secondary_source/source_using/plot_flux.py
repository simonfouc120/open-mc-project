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

from src.utils.post_preocessing.post_processing import *

mesh_tally_neutrons = mesh_tally_data(statepoint, "flux_mesh_xy_neutrons", "xy", 500, (-850.0, -850.0), (850.0, 850.0))
mesh_tally_neutrons.plot_flux(axis_one_index=250, x_lim=(0, 850), save_fig=True, fig_name="flux_plot_neutrons.png")

mesh_tally_neutrons.plot_dose(axis_one_index=250, 
                              particles_per_second=1e18, 
                              x_lim=(150, 850),
                              y_lim=(1e5, 1e9),
                              mesh_bin_volume=680.0,
                              save_fig=True,
                              radiological_area=True,
                              fig_name="dose_plot_neutrons.png")