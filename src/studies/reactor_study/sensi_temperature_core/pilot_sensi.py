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
from src.utils.pre_processing.pre_processing import (remove_previous_results, remove_surface_source_files, estimate_fissions_and_neutrons)
os.environ["OPENMC_CROSS_SECTIONS"] = PATH_TO_CROSS_SECTIONS

from src.models.model_complete_reactor import MODEL, GRAPHITE_CELL

# MODEL.export_to_xml()
material = MODEL.materials
geometry = MODEL.geometry
graphite_cell = GRAPHITE_CELL

# Variable definition
flux_array = []
fission_rate_array = []
nu_fission_rate_array= []
keff_array = []
temperature = [450 ,600, 750, 900, 1050, 1200, 1350 ,1500]  # K

for temp in temperature:
    for index, mat in enumerate(material):
        if mat.name == "Graphite":
            mat.temperature = temp  # Set the temperature to 300 K for all materials
        elif mat.name == "Fuel":
            mat.temperature = temp
        elif mat.name == "borated_steel":
            mat.temperature = temp - 100
        elif mat.name == "Beryllium":
            mat.temperature = temp - 120
        else: 
            mat.temperature = 300

    material.export_to_xml()
    geometry.export_to_xml()



    # Calcul de criticité simple 
    settings = openmc.Settings()
    batches_number= 100
    settings.batches = batches_number
    settings.inactive = 15
    settings.particles = 20000
    settings.source = openmc.Source()
    settings.source.space = openmc.stats.Point((0, 0, 0))
    settings.source.particle = 'neutron'
    settings.photon_transport = True
    settings.temperature = {'method': 'interpolation', 'range': (100, 1500)}  # Temperature range for sensitivity analysis

    # tally pour le flux dans le détecteur
    flux_tally = openmc.Tally(name="flux_tally")
    flux_tally.scores = ['flux']
    flux_tally.filters = [openmc.CellFilter(graphite_cell), openmc.ParticleFilter("neutron")]

    # Tally for fission rate
    fission_tally = openmc.Tally(name="fission_rate_tally")
    fission_tally.scores = ['fission']
    fission_tally.filters = [openmc.CellFilter(graphite_cell)]

    nu_fission_tally = openmc.Tally(name="nu_fission_rate_tally")
    nu_fission_tally.scores = ['nu-fission']
    nu_fission_tally.filters = [openmc.CellFilter(graphite_cell), openmc.ParticleFilter("neutron")]


    tallies = openmc.Tallies([flux_tally, fission_tally, nu_fission_tally])

    settings.export_to_xml()

    tallies.export_to_xml()

    remove_surface_source_files()
    remove_previous_results(batches_number=batches_number)
    start_time = time.time()
    openmc.run()
    end_time = time.time()  
    calculation_time = end_time - start_time

    output_json = CWD / "calculation_time.json"
    with open(output_json, "w") as f:
        json.dump({"calculation_time_seconds": calculation_time}, f)

    statepoint_file = openmc.StatePoint(f'statepoint.{batches_number}.h5')

    # load fission rate and error into a tuple
    flux_tally_results = statepoint_file.get_tally(name="flux_tally")
    flux_tuple = (flux_tally_results.mean.flatten(), flux_tally_results.std_dev.flatten())
    flux_array.append(flux_tuple)

    fission_rate_results = statepoint_file.get_tally(name="fission_rate_tally")
    fission_rate_tuple = (fission_rate_results.mean.flatten(), fission_rate_results.std_dev.flatten())
    fission_rate_array.append(fission_rate_tuple)


    nu_fission_rate_results = statepoint_file.get_tally(name="nu_fission_rate_tally")
    nu_fission_rate_tuple = (nu_fission_rate_results.mean.flatten(), nu_fission_rate_results.std_dev.flatten())
    nu_fission_rate_array.append(nu_fission_rate_tuple)


    keff_results = statepoint_file.keff
    keff_tuple = (keff_results.nominal_value, keff_results.std_dev)
    keff_array.append(keff_tuple)

# Save results to a JSON file
results = {
    "flux": flux_array,
    "fission_rate": fission_rate_array,
    "nu_fission_rate": nu_fission_rate_array,
    "keff": keff_array,
    "temperature": temperature
}
output_json = CWD / "sensitivity_temperature_results.json"
with open(output_json, "w") as f:
    json.dump(results, f, indent=4)

# plot results 
plt.figure(figsize=(10, 6))
plt.errorbar(temperature, [x[0] for x in keff_array], 
             yerr=[x[1] for x in keff_array], fmt='o-', color='orange')
plt.xlabel('Temperature [K]')
plt.ylabel('Value')
plt.title('Sensitivity Keff Results')
plt.grid()
plt.savefig(CWD / "sensitivity_keff_temp.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.errorbar(temperature, [x[0][0] for x in flux_array], 
            yerr=[x[1][0] for x in flux_array], fmt='o-', color='blue')
plt.xlabel('Temperature [K]')
plt.ylabel('Flux [neutrons/cm²/source particle]')
plt.title('Sensitivity Neutron Flux Results')
plt.grid()
plt.savefig(CWD / "sensitivity_flux_temp.png")
plt.show()


plt.figure(figsize=(10, 6))
plt.errorbar(temperature, [x[0][0] for x in fission_rate_array], 
            yerr=[x[1][0] for x in fission_rate_array], fmt='o-', color='red')
plt.xlabel('Temperature [K]')
plt.ylabel('Fission Rate [fissions/source particle]')
plt.title('Sensitivity Fission Rate Results')
plt.grid()
plt.savefig(CWD / "sensitivity_fission_rate_temp.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.errorbar(temperature, [x[0][0] for x in nu_fission_rate_array], 
            yerr=[x[1][0] for x in nu_fission_rate_array], fmt='o-', color='green')
plt.xlabel('Temperature [K]')
plt.ylabel('Nu Fission Rate [neutrons produced by fission/source particle]')
plt.title('Sensitivity Nu Fission Rate Results')
plt.grid()
plt.savefig(CWD / "sensitivity_nu_fission_rate_temp.png")
plt.show()