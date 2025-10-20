import openmc
import openmc.lib 
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
project_root = Path(__file__).resolve().parents[8]  
sys.path.append(str(project_root))

from parameters.parameters_paths import PATH_TO_CROSS_SECTIONS, IMAGE_PATH
from parameters.parameters_materials import *
from src.utils.pre_processing.pre_processing import *
from src.utils.post_preocessing.post_processing import *
from src.models.model_complete_reactor_class import *
from src.utils.weight_window.weight_window import *

my_reactor = Reactor_model(materials=material_dict, 
                           total_height_active_part=500.0, 
                           light_water_pool=False, 
                           slab_thickness=100,
                           concrete_wall_thickness=100,
                           calculation_sphere_coordinates=(0, 0, 510), 
                           calculation_sphere_radius=50.0,
                           thickness_lead_top_shielding=25.0, 
                           thickness_b4c_top_shielding=75.0)

model = my_reactor.model
model.export_to_xml()

os.environ["OPENMC_CROSS_SECTIONS"] = PATH_TO_CROSS_SECTIONS

factor = 4
new_material_dict = deepcopy(material_dict)

for material_name, material in new_material_dict.items():
    new_material_dict[material_name] = reducing_density(material, factor=factor)

new_material_dict["FUEL_UO2_MATERIAL"].remove_element("U")

my_reactor.material = new_material_dict
model = my_reactor.model
model.export_to_xml()

# Example usage:
create_weight_window(model, num_iterations=15, batches_number=50, particles_per_batch=100000, mesh_dimension=(150, 150, 150))
