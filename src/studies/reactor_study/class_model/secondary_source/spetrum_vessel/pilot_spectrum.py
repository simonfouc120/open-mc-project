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
                           calculation_sphere_coordinates=(-250, 0, 0),
                           calculation_sphere_radius=50)

MODEL = my_reactor.model
MODEL.export_to_xml()

plot_geometry(materials = openmc.Materials(list(my_reactor.material.values())), 
              plane="xy", origin=(0,0,0), saving_figure=True, dpi=500, height=1700, width=1700,
              suffix="_by_cell", pixels=(700, 700), color_by="cell")

settings = openmc.Settings()
batches_number= 100
settings.batches = batches_number
settings.particles = 50000
settings.source = openmc.FileSource('surface_source.h5')
settings.photon_transport = True
settings.run_mode = "fixed source"
settings.source.particles = ["neutron", "photon"]
MODEL.settings = settings
settings.export_to_xml()
MODEL.export_to_xml()

tallys = openmc.Tallies()

tally_flux_neutrons = openmc.Tally(name="tally_flux_neutrons")
tally_flux_neutrons.scores = ["flux"]
energy_limits_neutron_tally = [1e-2, 12e6]  # from 10 eV to 12 MeV
energy_bins = np.logspace(np.log10(energy_limits_neutron_tally[0]), np.log10(energy_limits_neutron_tally[1]), num=350)
tally_flux_neutrons.filters = [openmc.CellFilter(my_reactor.calc_sphere_cell),
                               openmc.EnergyFilter(energy_bins),
                               openmc.ParticleFilter(["neutron"])]
tallys.append(tally_flux_neutrons)

tally_flux_photons = openmc.Tally(name="tally_flux_photons")
tally_flux_photons.scores = ["flux"]
energy_bins_photons = np.linspace(1, 12e6, num=250)  # from 10 eV to 10 MeV
tally_flux_photons.filters = [openmc.CellFilter(my_reactor.calc_sphere_cell),
                              openmc.EnergyFilter(energy_bins_photons),
                              openmc.ParticleFilter(["photon"])]
tallys.append(tally_flux_photons)
tallys.export_to_xml()

remove_previous_results(batches_number=batches_number)
openmc.run()


