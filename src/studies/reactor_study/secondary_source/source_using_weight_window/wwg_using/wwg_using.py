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

CWD = Path(__file__).parent.resolve()

project_root = Path(__file__).resolve().parents[6]  
sys.path.append(str(project_root))
from parameters.parameters_paths import PATH_TO_CROSS_SECTIONS
from parameters.parameters_materials import FUEL_MATERIAL, HELIUM_MATERIAL, AIR_MATERIAL, CONCRETE_MATERIAL, GRAPHITE_MATERIAL, STEEL_MATERIAL, WATER_MATERIAL
from src.utils.pre_processing.pre_processing import (remove_previous_results, mesh_tally_plane)
from src.utils.post_preocessing.post_processing import load_mesh_tally, load_dammage_energy_tally, load_mesh_tally_dose
os.environ["OPENMC_CROSS_SECTIONS"] = PATH_TO_CROSS_SECTIONS

from src.models.model_complete_reactor import MODEL, GRAPHITE_CELL, CALCULATION_CELL

MODEL.export_to_xml()
material = MODEL.materials
graphite_cell = GRAPHITE_CELL
geometry = MODEL.geometry

settings = openmc.Settings()
batches_number= 1
settings.batches = batches_number
settings.inactive = 0
settings.particles = 4000000 # try more
settings.source = openmc.FileSource('surface_source.h5')
settings.photon_transport = True

#load and use the weight window wwg_creation.h5
mesh = openmc.RegularMesh().from_domain(geometry)
mesh.id = 999
mesh.dimension = (100, 100, 100)
mesh.lower_left = (-500.0, -500.0, -500.0)
mesh.upper_right = (500.0, 500.0, 500.0)

meshes = {mesh.id: mesh}

ww = openmc.hdf5_to_wws("weight_windows.h5")  
settings.weight_windows = ww

mesh_tally_neutron_yz = mesh_tally_plane(name_mesh_tally = "flux_mesh_neutrons_yz", particule_type='neutron', plane="yz",
                                      bin_number=800, lower_left=(-450.0, -450.0), upper_right=(450.0, 450.0),
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
os.environ["OMP_NUM_THREADS"] = "4"
openmc.run(threads=4)
end_time = time.time()
calculation_time = end_time - start_time

output_json = CWD / "calculation_time.json"
with open(output_json, "w") as f:
    json.dump({"calculation_time_seconds": calculation_time}, f)

# Load the statepoint file
statepoint_file = openmc.StatePoint(f'statepoint.{batches_number}.h5')

# Load the neutron flux tally
flux_tally_neutron = statepoint_file.get_tally(name="flux_tally_neutron")


load_mesh_tally(cwd = CWD, statepoint_file = statepoint_file, name_mesh_tally="flux_mesh_neutrons_yz",particule_type="neutron", bin_number=800,
                lower_left=(-450.0, -450.0), upper_right=(450.0, 450.0), zoom_x=(-450, 450), zoom_y=(-450, 450), plane="yz", saving_figure=True)

