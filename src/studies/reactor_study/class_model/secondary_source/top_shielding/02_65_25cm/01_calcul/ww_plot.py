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

material_dict["FUEL_UO2_MATERIAL"].remove_element("U")

my_reactor = Reactor_model(materials=material_dict, 
                           total_height_active_part=500.0, 
                           light_water_pool=False, 
                           slab_thickness=100,
                           concrete_wall_thickness=100,
                           calculation_sphere_coordinates=(0, 0, 500), 
                           calculation_sphere_radius=50.0)

model = my_reactor.model
model.export_to_xml()

plot_geometry(materials = openmc.Materials(list(my_reactor.material.values())), 
              plane="xy", origin=(0,0,0), saving_figure=False, dpi=500, height=1700, width=1700,
              pixels=(700, 700))

plot_geometry(materials = openmc.Materials(list(my_reactor.material.values())), 
              plane="xz", origin=(0,0,0), saving_figure=False, dpi=500, height=1700, width=1700,
              pixels=(700, 700))

plot_geometry(materials = openmc.Materials(list(my_reactor.material.values())), 
              plane="xz", origin=(0,0,0), saving_figure=False, dpi=500, height=1700, width=1700,
              pixels=(700, 700), color_by="cell")


results_path = "simulation_results.json"
with open(results_path, "r") as f:
    results = json.load(f)
neutron_emission_rate = results["neutrons_emitted_per_second"]["value"]

tallys = openmc.Tallies()

# run the simulation

settings = openmc.Settings()
batches_number= 1000
settings.batches = batches_number
settings.particles = 50000
settings.source = openmc.FileSource('surface_source.h5')
settings.photon_transport = True
settings.run_mode = "fixed source"
settings.source.particles = ["neutron", "photon"]
settings.statepoint = {"batches": list(range(10, batches_number + 1, 10))}

ww = openmc.hdf5_to_wws("weight_windows5.h5")  

settings.weight_windows = apply_spherical_correction_to_weight_windows(ww, particule_type='photon', sphere_center=(0.0, 0.0, 500.0), sphere_radius=50.0)
size_ww = get_ww_size(ww)
plot_weight_window(weight_window=ww[0], index_coord=size_ww[1]//2, energy_index=0, saving_fig=True, plane="xz", particle_type='neutron')
plot_weight_window(weight_window=ww[1], index_coord=size_ww[1]//2, energy_index=0, saving_fig=True, plane="xz", particle_type='photon')
