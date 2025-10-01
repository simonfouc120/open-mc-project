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

os.environ["OPENMC_CROSS_SECTIONS"] = PATH_TO_CROSS_SECTIONS

material_dict["FUEL_UO2_MATERIAL"].remove_element("U")

statepoint = openmc.StatePoint(f"statepoint.010.h5")

bin_mesh_volume = get_mesh_volumes(lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0), thickness=20.0, bin_number=500)
photons_per_s = 2.5e15

my_reactor = Reactor_model(materials=material_dict, 
                           total_height_active_part=500.0, 
                           light_water_pool=False, 
                           slab_thickness=100,
                           concrete_wall_thickness=100,
                           calculation_sphere_coordinates=(-575, 0, 0),
                           calculation_sphere_radius=50)

MODEL = my_reactor.model
MODEL.export_to_xml()

dose_over_geometry(model=MODEL ,statepoint_file=statepoint, name_mesh_tally="flux_mesh_photons_xy", plane="xy", 
                saving_figure=True, bin_number=500, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0), 
                zoom_x=(-850, 850), zoom_y=(-850, 850), plot_error=True, particule_type="photon", 
                particles_per_second=photons_per_s, mesh_bin_volume=bin_mesh_volume, radiological_area=True)   

dose_over_geometry(model=MODEL ,statepoint_file=statepoint, name_mesh_tally="flux_mesh_photons_xz", plane="xz", 
                saving_figure=True, bin_number=500, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0), 
                zoom_x=(-800, 800), zoom_y=(-800, 800), plot_error=True, particule_type="photon", 
                particles_per_second=photons_per_s, mesh_bin_volume=bin_mesh_volume, pixels_model_geometry=1_000_000)  

volume_tally_sphere = Volume_cell(cell=my_reactor.calc_sphere_cell).get_volume()

dose_rate, dose_rate_error = compute_dose_rate_tally(
    statepoint=statepoint,
    tally_name="flux_tally_sphere_dose",
    particule_per_second=photons_per_s,
    volume=volume_tally_sphere,
    unit="mSv/h"
)

relative_error = (dose_rate_error / dose_rate) * 100 if dose_rate > 0 else 0
print(f"Dose rate in the calculation sphere: {dose_rate:.2f} mSv/h (relative error: {relative_error:.2f} %)")

save_tally_result_to_json(
    tally_name="dose_rate",
    value=dose_rate,
    error=dose_rate_error,
    unit="mSv/h",
    filename="results.json"
)

mesh_tally_photons = mesh_tally_data(statepoint, "flux_mesh_photons_xy", "xy", 500, (-850.0, -850.0), (850.0, 850.0))
mesh_tally_photons.plot_dose(axis_two_index=250, 
                             particles_per_second=photons_per_s, 
                             x_lim=(-650, 0),
                             y_lim=(1e2, 1e10),
                             mesh_bin_volume=bin_mesh_volume,
                             save_fig=True,
                             radiological_area=True,
                             fig_name="dose_plot_photons.png")