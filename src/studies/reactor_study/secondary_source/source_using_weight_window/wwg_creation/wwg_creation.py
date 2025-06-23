import openmc
import os 
import time
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
from parameters.parameters_paths import PATH_TO_CROSS_SECTIONS
from parameters.parameters_materials import FUEL_MATERIAL, HELIUM_MATERIAL, AIR_MATERIAL, CONCRETE_MATERIAL, GRAPHITE_MATERIAL, STEEL_MATERIAL, WATER_MATERIAL
from src.utils.pre_processing.pre_processing import (remove_previous_results, mesh_tally_plane, reducing_density)
from src.utils.post_preocessing.post_processing import load_mesh_tally, load_dammage_energy_tally, load_mesh_tally_dose
from src.utils.weight_window.weight_window import plot_weight_window
os.environ["OPENMC_CROSS_SECTIONS"] = PATH_TO_CROSS_SECTIONS

from src.models.model_complete_reactor import MODEL, GRAPHITE_CELL, CALCULATION_CELL

materials = MODEL.materials
for index,material in enumerate(materials):
    materials[index] = reducing_density(material, 10)  

MODEL.materials = materials
MODEL.export_to_xml()

geometry = MODEL.geometry

settings = openmc.Settings()
batches_number= 1
settings.batches = batches_number
settings.inactive = 0
settings.particles = 10000000 # try more
settings.source = openmc.FileSource('surface_source.h5')
settings.photon_transport = True

mesh = openmc.RegularMesh().from_domain(geometry)
mesh.dimension = (30, 30, 30)
mesh.lower_left = (-500.0, -500.0, -500.0)
mesh.upper_right = (500.0, 500.0, 500.0)

energy_bounds = np.linspace(0.0, 15e6, 10)  # 10 energy bins from 0 to 15 MeV

wwg_neutron = openmc.WeightWindowGenerator(
    mesh=mesh,  
    energy_bounds=energy_bounds,
    particle_type='neutron', 
)

wwg_photon = deepcopy(wwg_neutron)
wwg_photon.particle_type = 'photon'

settings.max_history_splits = 1_000  
settings.weight_window_generators = [wwg_neutron, wwg_photon]

mesh_tally_neutron_yz = mesh_tally_plane(name_mesh_tally = "flux_mesh_neutrons_yz", particule_type='neutron', 
                                      plane="yz", bin_number=500, lower_left=(-450.0, -450.0), upper_right=(450.0, 450.0),
                                      thickness= 10.0, coord_value=0.0)

# Neutron flux tally on the CALCULATION_CELL
flux_tally_neutron = openmc.Tally(name="flux_tally_neutron")
flux_tally_neutron.scores = ['flux']
flux_tally_neutron.filters = [openmc.ParticleFilter("neutron"), openmc.CellFilter(CALCULATION_CELL)]

tallies = openmc.Tallies([flux_tally_neutron, mesh_tally_neutron_yz])

settings.export_to_xml()
tallies.export_to_xml()

remove_previous_results(batches_number=batches_number)
start_time = time.time()
ww_statepoint_filename = openmc.run()

openmc.run(threads=4)
end_time = time.time()
calculation_time = end_time - start_time

# Save calculation time to a JSON file
output_json = CWD / "calculation_time.json"
with open(output_json, "w") as f:
    json.dump({"calculation_time_seconds": calculation_time}, f)

# Load the statepoint file
statepoint_file = openmc.StatePoint(f'statepoint.{batches_number}.h5')

# Load the neutron flux tally
flux_tally_neutron = statepoint_file.get_tally(name="flux_tally_neutron")

load_mesh_tally(cwd = CWD, statepoint_file = statepoint_file, 
                name_mesh_tally="flux_mesh_neutrons_yz", particule_type="neutron", 
                bin_number=500, lower_left=(-450.0, -450.0), upper_right=(450.0, 450.0), 
                zoom_x=(-450, 450), zoom_y=(-450, 450), plane="yz", saving_figure=False)

ww = openmc.hdf5_to_wws("weight_windows.h5")  

plot_weight_window(weight_window=ww[0], index_coord=15, energy_index=0, saving_fig=True, plane="yz")
