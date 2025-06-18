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

project_root = Path(__file__).resolve().parents[4]  
sys.path.append(str(project_root))
from parameters.parameters_paths import PATH_TO_CROSS_SECTIONS
from parameters.parameters_materials import FUEL_MATERIAL, HELIUM_MATERIAL, AIR_MATERIAL, CONCRETE_MATERIAL, GRAPHITE_MATERIAL, STEEL_MATERIAL, WATER_MATERIAL
from src.utils.pre_processing.pre_processing import (remove_previous_results, remove_surface_source_files)
from src.utils.post_preocessing.post_processing import load_mesh_tally, load_dammage_energy_tally, load_mesh_tally_dose
os.environ["OPENMC_CROSS_SECTIONS"] = PATH_TO_CROSS_SECTIONS

from src.models.model_complete_reactor import MODEL, GRAPHITE_CELL

MODEL.export_to_xml()
material = MODEL.materials
graphite_cell = GRAPHITE_CELL

# Calcul de criticité simple 
settings = openmc.Settings()
batches_number= 100
settings.batches = batches_number
settings.inactive = 2
settings.particles = 50000
settings.source = openmc.Source()
settings.source.space = openmc.stats.Point((0, 0, 0))
settings.source.particle = 'neutron'
settings.photon_transport = True
settings.surf_source_write = {
    'cell': 11,
    'max_source_files': 5, 
    'max_particles' : 100000,  # Nombre de particules par fichier

}
# tally pour le flux dans le détecteur
tally = openmc.Tally(name="flux_tally")
tally.scores = ['flux']
tally.filters = [openmc.CellFilter(graphite_cell)]

# Tally for fission rate
fission_tally = openmc.Tally(name="fission_rate_tally")
fission_tally.scores = ['fission']
fission_tally.filters = [openmc.CellFilter(graphite_cell)]

# Tally for nu-fission (nu * fission rate)
nu_fission_tally = openmc.Tally(name="nu_fission_rate_tally")
nu_fission_tally.scores = ['nu-fission']
nu_fission_tally.filters = [openmc.CellFilter(graphite_cell)]

tallies = openmc.Tallies([tally, fission_tally, nu_fission_tally])

settings.export_to_xml()
tallies.export_to_xml()
remove_surface_source_files()
remove_previous_results(batches_number=batches_number)
start_time = time.time()
openmc.run(threads=4)
end_time = time.time()
calculation_time = end_time - start_time

# Save calculation time to a JSON file
output_json = CWD / "calculation_time.json"
with open(output_json, "w") as f:
    json.dump({"calculation_time_seconds": calculation_time}, f)
