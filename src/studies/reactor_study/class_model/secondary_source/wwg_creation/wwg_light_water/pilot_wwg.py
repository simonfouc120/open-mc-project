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
project_root = Path(__file__).resolve().parents[7]  
sys.path.append(str(project_root))

from parameters.parameters_paths import PATH_TO_CROSS_SECTIONS, IMAGE_PATH
from parameters.parameters_materials import *
from src.utils.pre_processing.pre_processing import *
from src.utils.post_preocessing.post_processing import *
from src.models.model_complete_reactor_class import *
from src.utils.weight_window.weight_window import *
from src.models.model_complete_reactor_class import material_dict



def plot_ww_slices(ww_list, mesh, fac_div, energy_index=0, particles=("neutron", "photon"), planes=("yz", "xy")):
    mid_idx = mesh.dimension[0] // 2
    for i, particle in enumerate(particles):
        if i >= len(ww_list):
            break
        for plane in planes:
            plot_weight_window(
                weight_window=ww_list[i],
                index_coord=mid_idx,
                energy_index=energy_index,
                saving_fig=True,
                plane=plane,
                particle_type=particle,
                suffix_fig=f"_factor_{fac_div}_reduced_density",
            )



# Define reactor
my_reactor = Reactor_model(materials=material_dict, 
                           total_height_active_part=500.0, 
                           light_water_pool=True, 
                           light_water_length=900.0, # cm
                           light_water_height=900.0, # cm
                           cavity=False,
                           top_shielding= False,
                           slab_thickness=100,
                           concrete_wall_thickness=100,
                           calculation_sphere_coordinates=(0, 0, 500), 
                           calculation_sphere_radius=50.0)

os.environ["OPENMC_CROSS_SECTIONS"] = PATH_TO_CROSS_SECTIONS


MATERIAL_DICT_ORIGINAL = deepcopy(material_dict)
new_material_dict = deepcopy(material_dict)

for material_name, material in new_material_dict.items():
    new_material_dict[material_name] = reducing_density(material, factor=10)

new_material_dict["FUEL_UO2_MATERIAL"].remove_element("U")

my_reactor.material = new_material_dict
MODEL = my_reactor.model

mesh, _ = setup_and_run_wwg(source='surface_source.h5',
                            model=MODEL,
                            mesh_dimension=(50, 50, 50),
                            mesh_size=850.0,
                            particle_types=("neutron", "photon"),
                            batches=5,
                            particles_per_batch=50_000,
                            max_history_splits=1_000)

ww = openmc.hdf5_to_wws("weight_windows.h5")  

plot_ww_slices(ww, mesh, 10)

for fac_div in [8, 6, 4]:
    MATERIAL_DICT_ORIGINAL = deepcopy(material_dict)
    new_material_dict = deepcopy(material_dict)

    for material_name, material in new_material_dict.items():
        new_material_dict[material_name] = reducing_density(material, factor=fac_div)

    new_material_dict["FUEL_UO2_MATERIAL"].remove_element("U")

    my_reactor.material = new_material_dict
    MODEL = my_reactor.model
    
    if fac_div != 8 : 
        ww = openmc.hdf5_to_wws("weight_windows.h5")  

    # remove previous weight windows
    if os.path.exists("weight_windows.h5"):
        os.remove("weight_windows.h5")

    my_settings = openmc.Settings()
    my_settings.weight_windows = ww

    mesh, _ = setup_and_run_wwg(source='surface_source.h5',
                                settings=my_settings,
                                model=MODEL,
                                mesh_dimension=(50, 50, 50),
                                mesh_size=850.0,
                                particle_types=("neutron", "photon"),
                                batches=5,
                                particles_per_batch=5_000,
                                max_history_splits=1_000)
    
    ww = openmc.hdf5_to_wws("weight_windows.h5")  

    # call the function
    plot_ww_slices(ww, mesh, fac_div)
