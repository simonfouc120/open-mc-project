import openmc
import os 
import json
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
import sys 
from PIL import Image
import numpy as np

CWD = Path(__file__).parent.resolve() 
project_root = Path(__file__).resolve().parents[8]  
sys.path.append(str(project_root))

from parameters.parameters_paths import PATH_TO_CROSS_SECTIONS, IMAGE_PATH
from parameters.parameters_materials import *
from src.utils.pre_processing.pre_processing import *
from src.utils.post_preocessing.post_processing import *
from src.models.model_complete_reactor_class import *
from src.utils.weight_window.weight_window import *

os.environ["OPENMC_CROSS_SECTIONS"] = PATH_TO_CROSS_SECTIONS


ww = openmc.hdf5_to_wws("weight_windows.h5") 

plot_weight_window(weight_window=ww[0], index_coord=32, energy_index=0, saving_fig=True, plane="yz", particle_type='photon')

plot_weight_window(weight_window=ww[0], index_coord=32, energy_index=0, saving_fig=True, plane="xy", particle_type='photon')
