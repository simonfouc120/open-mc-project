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
project_root = Path(__file__).resolve().parents[6]  
sys.path.append(str(project_root))

from parameters.parameters_paths import PATH_TO_CROSS_SECTIONS, IMAGE_PATH
from parameters.parameters_materials import *
from src.utils.pre_processing.pre_processing import *
from src.utils.post_preocessing.post_processing import *
from src.models.model_complete_reactor_class import *
from src.utils.weight_window.weight_window import *

for material in material_dict.values():
    material= reducing_density(material, factor=10)

os.environ["OPENMC_CROSS_SECTIONS"] = PATH_TO_CROSS_SECTIONS

material_dict["FUEL_UO2_MATERIAL"].remove_element("U")

my_reactor = Reactor_model(materials=material_dict, 
                           total_height_active_part=500.0, 
                           light_water_pool=False, 
                           slab_thickness=100,
                           concrete_wall_thickness=100,
                           calculation_sphere_coordinates=(-600, 0, 0), 
                           calculation_sphere_radius=50.0)

MODEL = my_reactor.model
MODEL.export_to_xml()

plot_geometry(materials = openmc.Materials(list(my_reactor.material.values())), 
              plane="xy", origin=(0,0,0), saving_figure=False, dpi=500, height=1700, width=1700,
              pixels=(700, 700))

# fonction material pas de fission

tallys = openmc.Tallies()

# run the simulation

settings = openmc.Settings()
batches_number= 100
settings.batches = batches_number
settings.particles = 500000
settings.source = openmc.FileSource('surface_source.h5')
settings.photon_transport = True
settings.run_mode = "fixed source"
settings.source.particles = ["neutron", "photon"]

mesh = openmc.RegularMesh().from_domain(MODEL.geometry)
mesh.dimension = (30, 30, 30)
mesh.lower_left = (-700.0, -700.0, -700.0)
mesh.upper_right = (700.0, 700.0, 700.0)

energy_bounds = np.linspace(0.0, 15e6, 10)  # 10 energy bins from 0 to 15 MeV

wwg_neutron = openmc.WeightWindowGenerator(
    mesh=mesh,  
    energy_bounds=energy_bounds,
    particle_type='neutron', 
    method="magic"
)

wwg_photon = deepcopy(wwg_neutron)
wwg_photon.particle_type = 'photon'

settings.max_history_splits = 1_000  
settings.weight_window_generators = [wwg_neutron, wwg_photon]


MODEL.settings = settings
# settings.export_to_xml()
MODEL.export_to_xml()

remove_previous_results(CWD)

openmc.run()

statepoint = openmc.StatePoint(f"statepoint.{batches_number}.h5")

ww = openmc.hdf5_to_wws("weight_windows.h5")  

plot_weight_window(weight_window=ww[0], index_coord=15, energy_index=0, saving_fig=True, plane="yz", particle_type='neutron')

plot_weight_window(weight_window=ww[1], index_coord=15, energy_index=0, saving_fig=True, plane="yz", particle_type='photon')

plot_weight_window(weight_window=ww[0], index_coord=15, energy_index=0, saving_fig=True, plane="xy", particle_type='neutron')

plot_weight_window(weight_window=ww[1], index_coord=15, energy_index=0, saving_fig=True, plane="xy", particle_type='photon')

