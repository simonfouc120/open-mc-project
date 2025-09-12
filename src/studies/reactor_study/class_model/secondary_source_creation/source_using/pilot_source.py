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
os.environ["OPENMC_CROSS_SECTIONS"] = PATH_TO_CROSS_SECTIONS

material_dict["FUEL_UO2_MATERIAL"].remove_element("U")

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
              suffix="_test_no_fission", pixels=(700, 700))

# fonction material pas de fission

tallys = openmc.Tallies()

# flux_tally_neutron.filters = [openmc.ParticleFilter("neutron"), openmc.CellFilter(my_reactor.calc_sphere_cell)]

# run the simulation

settings = openmc.Settings()
batches_number= 100
settings.batches = batches_number
settings.particles = 500000
settings.source = openmc.FileSource('surface_source.h5')
settings.photon_transport = True
settings.run_mode = "fixed source"
settings.source.particles = ["neutron", "photon"]
MODEL.settings = settings
settings.export_to_xml()
MODEL.export_to_xml()

tallys = openmc.Tallies()

mesh_tally_xy_neutrons = mesh_tally_plane(name_mesh_tally = "flux_mesh_xy_neutrons", particule_type='neutron', plane="xy",
                                      bin_number=500, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0),
                                      thickness= 20.0, coord_value=0.0)
tallys.append(mesh_tally_xy_neutrons)


mesh_tally_xy_photons = mesh_tally_plane(name_mesh_tally = "flux_mesh_photons_xy", particule_type='photon', plane="xy",
                                      bin_number=500, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0),
                                      thickness= 20.0, coord_value=0.0)
tallys.append(mesh_tally_xy_photons)

tallys.export_to_xml()

remove_previous_results(CWD)

openmc.run()

statepoint = openmc.StatePoint(f"statepoint.{batches_number}.h5")


load_mesh_tally(cwd=CWD, statepoint_file=statepoint, name_mesh_tally="flux_mesh_xy_neutrons", plane="xy", 
                saving_figure=False, bin_number=500, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0), 
                zoom_x=(-850, 850), zoom_y=(-850, 850), plot_error=True)

load_mesh_tally(cwd=CWD, statepoint_file=statepoint, name_mesh_tally="flux_mesh_photons_xy", plane="xy", 
                saving_figure=False, bin_number=500, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0), 
                zoom_x=(-850, 850), zoom_y=(-850, 850), plot_error=True, particule_type="photon") 
