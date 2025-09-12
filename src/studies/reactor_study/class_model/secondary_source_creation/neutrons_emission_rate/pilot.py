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

my_reactor = Reactor_model(materials=material_dict, 
                           total_height_active_part=500.0, 
                           light_water_pool=True, 
                           slab_thickness=100,
                           concrete_wall_thickness=50,
                           calculation_sphere_coordinates=(0, -350, 0))

MODEL = my_reactor.model
MODEL.export_to_xml()

# run the simulation

settings = openmc.Settings()
batches_number= 100
settings.batches = batches_number
settings.inactive = 10
settings.particles = 4000
settings.source = openmc.IndependentSource()
settings.source.space = openmc.stats.Point((0, 0, 0))
settings.source.particle = 'neutron'
settings.photon_transport = True
MODEL.settings = settings
settings.export_to_xml()
MODEL.export_to_xml()

tallys = openmc.Tallies()

nu_fission_tally = openmc.Tally(name="nu_fission_rate_tally")
nu_fission_tally.scores = ['nu-fission']
nu_fission_tally.filters = [openmc.CellFilter(my_reactor.graphite_assembly_main_cell)]
tallys.append(nu_fission_tally)

fission_tally = openmc.Tally(name="fission_rate_tally")
fission_tally.scores = ['fission']
fission_tally.filters = [openmc.CellFilter(my_reactor.graphite_assembly_main_cell)]
tallys.append(fission_tally)
tallys.export_to_xml()
remove_previous_results(CWD)

openmc.run()

statepoint = openmc.StatePoint(f"statepoint.{batches_number}.h5")

fission_tally_result = statepoint.get_tally(name="fission_rate_tally")
nu_fission_tally_result = statepoint.get_tally(name="nu_fission_rate_tally")

fission_rate = float(fission_tally_result.mean.flatten()[0])
fission_rate_error = float(fission_tally_result.std_dev.flatten()[0])
nu_fission_rate = float(nu_fission_tally_result.mean.flatten()[0])
nu_fission_rate_error = float(nu_fission_tally_result.std_dev.flatten()[0])

nu_bar = nu_fission_rate / fission_rate if fission_rate != 0 else float('nan')
fissions_per_second, neutrons_emitted_per_second = estimate_fissions_and_neutrons(POWER, nu_bar)

keff = {"mean": statepoint.keff._nominal_value,
        "std_dev": statepoint.keff._std_dev}

results = {
    "fission_rate": {
        "value": fission_rate,
        "error": fission_rate_error,
        "units": "fissions/source particle"
    },
    "fission_rate_per_second": {
        "value": fissions_per_second,
        "units": f"fissions/s (for {POWER:.2e} W reactor)"
    },
    "nu_fission_rate": {
        "value": nu_fission_rate,
        "error": nu_fission_rate_error,
        "units": "neutrons produced by fission/source particle"
    },
    "nu_bar": {
        "value": nu_bar,
        "units": "dimensionless (average neutrons per fission)"
    },
    "neutrons_emitted_per_second": {
        "value": neutrons_emitted_per_second,
        "units": f"neutrons/s (for {POWER:.2e} W reactor)"
    },
    "keff": {
        "mean": keff["mean"],
        "std_dev": keff["std_dev"],
        "units": "dimensionless (effective multiplication factor)"
    },
}

with open("simulation_results.json", "w") as f:
    json.dump(results, f, indent=2)
