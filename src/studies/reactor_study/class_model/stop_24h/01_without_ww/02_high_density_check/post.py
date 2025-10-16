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
project_root = Path(__file__).resolve().parents[7]  
sys.path.append(str(project_root))

from parameters.parameters_paths import PATH_TO_CROSS_SECTIONS, IMAGE_PATH
from parameters.parameters_materials import *
from src.utils.pre_processing.pre_processing import *
from src.utils.post_preocessing.post_processing import *
from src.models.model_complete_reactor_class import *

os.environ["OPENMC_CROSS_SECTIONS"] = PATH_TO_CROSS_SECTIONS

material_dict["FUEL_UO2_MATERIAL"].remove_element("U")

statepoint = openmc.StatePoint(f"statepoint.100.h5")

bin_mesh_volume = get_mesh_volumes(lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0), thickness=20.0, bin_number=500)
photons_per_s = 2.5e15

my_reactor = Reactor_model(materials=material_dict, 
                           total_height_active_part=500.0, 
                           light_water_pool=False, 
                           slab_thickness=100,
                           concrete_wall_thickness=50,
                           calculation_sphere_coordinates=(-520, 0, 0),
                           calculation_sphere_radius=50)

model = my_reactor.model
model.export_to_xml()


mesh_tally_photons_xy = mesh_tally_data(statepoint, "flux_mesh_photons_xy", "XY", "photon")
mesh_tally_photons_xy.plot_dose_map(model=model, saving_figure=True, plot_error=True, model_geometry=False,
                                 particles_per_second=photons_per_s, radiological_area=False)   



mesh_tally_photons_xz = mesh_tally_data(statepoint, "flux_mesh_photons_xz", "XZ", "photon")
mesh_tally_photons_xz.plot_dose_map(model=model, saving_figure=True, plot_error=True, model_geometry=False,
                                 particles_per_second=photons_per_s, radiological_area=False)   

