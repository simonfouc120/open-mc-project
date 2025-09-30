import openmc
import os 
import json
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
import sys 
from PIL import Image
import numpy as np
import copy


CWD = Path(__file__).parent.resolve() 
project_root = Path(__file__).resolve().parents[7]  
sys.path.append(str(project_root))

from parameters.parameters_paths import PATH_TO_CROSS_SECTIONS, IMAGE_PATH
from parameters.parameters_materials import *
from src.utils.pre_processing.pre_processing import *
from src.utils.post_preocessing.post_processing import *
from src.models.model_complete_reactor_class import *

os.environ["OPENMC_CROSS_SECTIONS"] = PATH_TO_CROSS_SECTIONS


for material in material_dict.values():
    material= reducing_density(material, factor=5)

material_dict["FUEL_UO2_MATERIAL"].remove_element("U")

my_reactor = Reactor_model(materials=material_dict, 
                           total_height_active_part=500.0, 
                           light_water_pool=False, 
                           slab_thickness=100,
                           concrete_wall_thickness=50,
                           calculation_sphere_coordinates=(-520, 0, 0),
                           calculation_sphere_radius=50)

MODEL = my_reactor.model

# fonction material pas de fission
# run the simulation
# read the data from gamma_spectrum_tabular.txt
data = np.loadtxt('gamma_spectrum_tabular.txt', comments='#')
energies_eV = data[:,0]*1e6  # Energy in MeV
pdf_list = data[:,1]  # PDF values
photons_per_s = 2.5e15

# -----------------------
settings = openmc.Settings()
batches_number= 100
settings.batches = batches_number
settings.inactive = 0
settings.particles = 10000
settings.statepoint_interval = 10
settings.photon_transport = True
settings.run_mode = "fixed source"
spatial_dist = openmc.stats.Box((-200., -200, -250.), (200., 200., 250.))
settings.source = openmc.IndependentSource(
    space=spatial_dist,
    constraints={'domains': [GRAPHITE_MATERIAL, HELIUM_MATERIAL, FUEL_UO2_MATERIAL]},  # contraints can be on cells, materials or universe (here both work fine)
)
settings.source.particles = ["photon"]
settings.source.energy = openmc.stats.Discrete(energies_eV, pdf_list)
settings.source.strength = photons_per_s

mesh = openmc.RegularMesh().from_domain(MODEL.geometry)
mesh.dimension = (65, 65, 65)
mesh.lower_left = (-700.0, -700.0, -700.0)
mesh.upper_right = (700.0, 700.0, 700.0)

wwg_photon = openmc.WeightWindowGenerator(
    mesh=mesh,  
    max_realizations=settings.particles, 
    particle_type='photon', 
    method="magic"
)

settings.max_history_splits = 10_000  
settings.weight_window_generators = [wwg_photon]

# rr_model.settings = settings
# rr_model.export_to_xml()

# ##
MODEL.settings = settings
MODEL.export_to_xml()


remove_previous_results()

openmc.run()

remove_intermediate_files()
