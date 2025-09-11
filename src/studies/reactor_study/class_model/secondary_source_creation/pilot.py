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
project_root = Path(__file__).resolve().parents[5]  
sys.path.append(str(project_root))

from parameters.parameters_paths import PATH_TO_CROSS_SECTIONS, IMAGE_PATH
from parameters.parameters_materials import *
from src.utils.pre_processing.pre_processing import *
from src.utils.post_preocessing.post_processing import *
from src.models.model_complete_reactor_class import *
os.environ["OPENMC_CROSS_SECTIONS"] = PATH_TO_CROSS_SECTIONS

# material_dict["FUEL_UO2_MATERIAL"].remove_element("U")

for material in material_dict.values():
    if material in [WATER_MATERIAL, HEAVY_WATER_MATERIAL, CONCRETE_MATERIAL]:
        material.set_density('g/cm3', EPSILON_DENSITY)

my_reactor = Reactor_model(materials=material_dict, 
                           total_height_active_part=500.0, 
                           light_water_pool=True, 
                           slab_thickness=100,
                           concrete_wall_thickness=50,
                           calculation_sphere_coordinates=(0, -350, 0))

MODEL = my_reactor.model
MODEL.export_to_xml()

plot_geometry(materials = openmc.Materials(list(my_reactor.material.values())), 
              plane="xy", origin=(0,0,0), saving_figure=True, dpi=500, height=1700, width=1700,
              suffix="_test_no_fission", pixels=(1700,1700))

# fonction material pas de fission

tallys = openmc.Tallies()

# flux_tally_neutron.filters = [openmc.ParticleFilter("neutron"), openmc.CellFilter(my_reactor.calc_sphere_cell)]

# run the simulation

settings = openmc.Settings()
batches_number= 50
settings.batches = batches_number
settings.inactive = 10
settings.particles = 500
settings.source = openmc.IndependentSource()
settings.source.space = openmc.stats.Point((0, 0, 0))
settings.source.particle = 'neutron'
settings.photon_transport = True
settings.source.angle = openmc.stats.Isotropic()  
settings.surf_source_write = {
    'cell': 999,
    'max_source_files': 3, 
    'max_particles' : 100000000,  # Nombre de particules par fichier
}
MODEL.settings = settings
settings.export_to_xml()
MODEL.export_to_xml()

openmc.run()

statepoint = openmc.StatePoint(f"statepoint.{batches_number}.h5")


