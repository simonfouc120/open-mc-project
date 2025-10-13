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



def plot_ww_slices(ww_list, mesh, fac_div:int=1, 
                   energy_index:int=0, 
                   particles:tuple=("neutron", "photon"), 
                   planes:tuple=("yz", "xy"),
                   bound_type:str='lower'):
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
                bound_type=bound_type,
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

factor = 0.1
MATERIAL_DICT_ORIGINAL = deepcopy(material_dict)
new_material_dict = deepcopy(material_dict)

for material_name, material in new_material_dict.items():
    new_material_dict[material_name] = reducing_density(material, factor=factor)

new_material_dict["FUEL_UO2_MATERIAL"].remove_element("U")

my_reactor.material = new_material_dict
MODEL = my_reactor.model


settings = openmc.Settings()
batches_number= 4
particles_per_batch = 50000
max_history_splits = 10_000
src = openmc.FileSource('surface_source.h5')
particle_types = ("neutron", "photon")
settings.weight_windows = openmc.hdf5_to_wws("weight_windows_f8.h5")  

settings.batches = batches_number
settings.particles = particles_per_batch
settings.run_mode = "fixed source"
src.particles = list(particle_types)  
settings.source = src
settings.photon_transport = True
# add random ray 


# Mesh (from model geometry)
mesh_dimension = (50, 50, 50)
mesh_size = 850.0
mesh = openmc.RegularMesh().from_domain(MODEL.geometry)
mesh.dimension = tuple(mesh_dimension)
mesh.lower_left = (-mesh_size, -mesh_size, -mesh_size)
mesh.upper_right = (mesh_size, mesh_size, mesh_size)

# Weight-window generators for each particle type
wwgs: List[openmc.WeightWindowGenerator] = []
for ptype in particle_types:
    wwg = openmc.WeightWindowGenerator(
        mesh=mesh,
        max_realizations=settings.batches,
        particle_type=ptype,
        method="magic",
        on_the_fly=True,
    )
    wwgs.append(wwg)

settings.max_history_splits = max_history_splits
settings.weight_window_generators = wwgs
# Attach settings to model and run
MODEL.settings = settings
MODEL.export_to_xml()

remove_previous_results(batches_number=batches_number)
# openmc.run()
MODEL.run()
ww= openmc.hdf5_to_wws("weight_windows.h5")  
plot_ww_slices(ww, mesh, factor)

