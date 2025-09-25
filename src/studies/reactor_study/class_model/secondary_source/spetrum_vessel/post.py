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
project_root = Path(__file__).resolve().parents[6]  
sys.path.append(str(project_root))

from parameters.parameters_paths import PATH_TO_CROSS_SECTIONS, IMAGE_PATH
from parameters.parameters_materials import *
from src.utils.pre_processing.pre_processing import *
from src.utils.post_preocessing.post_processing import *
from src.models.model_complete_reactor_class import *


# Load the statepoint file once
statepoint = openmc.StatePoint("statepoint.100.h5")

# Plot the neutron spectrum in eV
plot_flux_spectrum(
    statepoint=statepoint,
    tally_name="tally_flux_neutrons",
    particle_name="Neutron",
    figure_num=1,
    x_scale='log',
    y_scale='log',
    ylim=(1e-7, 1e-2),
    energy_unit='eV',
    savefig=True,
    figsize=(9, 6),
    sigma=2
)

# Plot the photon spectrum in MeV
plot_flux_spectrum(
    statepoint=statepoint,
    tally_name="tally_flux_photons",
    particle_name="Photon",
    figure_num=2,
    x_scale='linear',
    y_scale='log',
    energy_unit='MeV',
    savefig=True,
    ylim=(1e-7, 1e-1),
    figsize=(9, 6),
    sigma=2
)
