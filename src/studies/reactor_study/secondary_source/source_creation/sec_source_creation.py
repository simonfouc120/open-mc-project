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

project_root = Path(__file__).resolve().parents[5]  
sys.path.append(str(project_root))
from parameters.parameters_paths import PATH_TO_CROSS_SECTIONS
from parameters.parameters_reactor import POWER
from src.utils.pre_processing.pre_processing import (remove_previous_results, remove_surface_source_files, estimate_fissions_and_neutrons)
os.environ["OPENMC_CROSS_SECTIONS"] = PATH_TO_CROSS_SECTIONS

from src.models.model_complete_reactor import MODEL, GRAPHITE_CELL

MODEL.export_to_xml()
material = MODEL.materials
graphite_cell = GRAPHITE_CELL

# Calcul de criticité simple 
settings = openmc.Settings()
batches_number= 450
settings.batches = batches_number
settings.inactive = 20
settings.particles = 50000
settings.source = openmc.Source()
settings.source.space = openmc.stats.Point((0, 0, 0))
settings.source.particle = 'neutron'
settings.photon_transport = True
settings.surf_source_write = {
    'cell': 11,
    'max_source_files': 3, 
    'max_particles' : 100000000,  # Nombre de particules par fichier
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
os.environ["OMP_NUM_THREADS"] = "4"
openmc.run(threads=1)
end_time = time.time()
calculation_time = end_time - start_time

# Save calculation time to a JSON file
output_json = CWD / "calculation_time.json"
with open(output_json, "w") as f:
    json.dump({"calculation_time_seconds": calculation_time}, f)

statepoint_file = openmc.StatePoint(f'statepoint.{batches_number}.h5')

tally = statepoint_file.get_tally(name="flux_tally")
mean_flux = tally.mean.flatten().tolist()
std_flux = tally.std_dev.flatten().tolist()
# Get fission rate and nu-bar
fission_tally_result = statepoint_file.get_tally(name="fission_rate_tally")
nu_fission_tally_result = statepoint_file.get_tally(name="nu_fission_rate_tally")

fission_rate = float(fission_tally_result.mean.flatten()[0])
nu_fission_rate = float(nu_fission_tally_result.mean.flatten()[0])
nu_bar = nu_fission_rate / fission_rate if fission_rate != 0 else float('nan')
# Estimate fissions per second (fission/s) for a 5 MW reactor

reactor_power = POWER  # Thermical Watts 


keff = {"mean": statepoint_file.keff._nominal_value,
        "std_dev": statepoint_file.keff._std_dev}

fissions_per_second, neutrons_emitted_per_second = estimate_fissions_and_neutrons(reactor_power, nu_bar)

results = {
    "mean_flux": {
        "value": mean_flux,
        "units": "neutrons/cm^2/source particle"
    },
    "fission_rate": {
        "value": fission_rate,
        "units": "fissions/source particle"
    },
    "fission_rate_per_second": {
        "value": fissions_per_second,
        "units": "fissions/s (for 5 MW reactor)"
    },
    "nu_fission_rate": {
        "value": nu_fission_rate,
        "units": "neutrons produced by fission/source particle"
    },
    "nu_bar": {
        "value": nu_bar,
        "units": "dimensionless (average neutrons per fission)"
    },
    "neutrons_emitted_per_second": {
        "value": neutrons_emitted_per_second,
        "units": "neutrons/s (for 5 MW reactor)"
    },
    "keff": {
        "mean": keff["mean"],
        "std_dev": keff["std_dev"],
        "units": "dimensionless (effective multiplication factor)"
    },
}

with open(CWD / "simulation_results.json", "w") as f:
    json.dump(results, f, indent=2)
