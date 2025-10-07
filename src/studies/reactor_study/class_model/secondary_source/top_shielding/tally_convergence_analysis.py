import openmc
import os 
import json
from pathlib import Path
import sys 
import numpy as np
import matplotlib.pyplot as plt

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

results_path = "simulation_results.json"
with open(results_path, "r") as f:
    results = json.load(f)
neutron_emission_rate = results["neutrons_emitted_per_second"]["value"]

my_reactor = Reactor_model(materials=material_dict, 
                           total_height_active_part=500.0, 
                           light_water_pool=False, 
                           slab_thickness=100,
                           concrete_wall_thickness=100,
                           calculation_sphere_coordinates=(0, 0, 500), 
                           calculation_sphere_radius=50.0)

statepoint_files = sorted(Path(".").glob("statepoint.*.h5"), key=lambda f: int(f.stem.split(".")[-1]))

batch_numbers = []
dose_rate_values_photon = []
dose_rate_errors_photon = []
dose_rate_values_neutron = []
dose_rate_errors_neutron = []

for statepoint_file in statepoint_files:
    batch_number = int(statepoint_file.stem.split(".")[-1])
    batch_numbers.append(batch_number)
    statepoint = openmc.StatePoint(str(statepoint_file))
    volume = Volume_cell(cell=my_reactor.calc_sphere_cell).get_volume()

    # Photon dose rate
    photon_value, photon_error = compute_dose_rate_tally(
        statepoint=statepoint,
        tally_name="flux_tally_sphere_dose_photon",
        particule_per_second=neutron_emission_rate,
        volume=volume,
        unit="mSv/h"
    )
    dose_rate_values_photon.append(photon_value)
    dose_rate_errors_photon.append(photon_error)

    # Neutron dose rate
    neutron_value, neutron_error = compute_dose_rate_tally(
        statepoint=statepoint,
        tally_name="flux_tally_sphere_dose_neutron",
        particule_per_second=neutron_emission_rate,
        volume=volume,
        unit="mSv/h"
    )
    dose_rate_values_neutron.append(neutron_value)
    dose_rate_errors_neutron.append(neutron_error)

dose_rate_values_photon_array = np.array(dose_rate_values_photon)
dose_rate_errors_photon_array = np.array(dose_rate_errors_photon)
dose_rate_values_neutron_array = np.array(dose_rate_values_neutron)
dose_rate_errors_neutron_array = np.array(dose_rate_errors_neutron)

relative_errors_photon = dose_rate_errors_photon_array / dose_rate_values_photon_array
relative_errors_neutron = dose_rate_errors_neutron_array / dose_rate_values_neutron_array

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Relative Error - Neutrons
axs[0, 0].scatter(batch_numbers, relative_errors_neutron, label="Neutrons", color='blue', marker='x')
axs[0, 0].set_title("Relative Error (Neutrons)")
axs[0, 0].set_xlabel("Batch Number")
axs[0, 0].set_ylabel("Relative Error")
axs[0, 0].legend()
axs[0, 0].grid()

# Relative Error - Photons
axs[0, 1].scatter(batch_numbers, relative_errors_photon, label="Photons", color='orange', marker='x')
axs[0, 1].set_title("Relative Error (Photons)")
axs[0, 1].set_xlabel("Batch Number")
axs[0, 1].set_ylabel("Relative Error")
axs[0, 1].legend()
axs[0, 1].grid()

# Dose Rate - Neutrons
axs[1, 0].scatter(batch_numbers, dose_rate_values_neutron_array, label="Neutrons", color='blue', marker='o')
axs[1, 0].set_title("Dose Rate (Neutrons) [mSv/h]")
axs[1, 0].set_xlabel("Batch Number")
axs[1, 0].set_ylabel("Dose Rate [mSv/h]")
axs[1, 0].set_yscale("log")
axs[1, 0].legend()
axs[1, 0].grid()

# Dose Rate - Photons
axs[1, 1].scatter(batch_numbers, dose_rate_values_photon_array, label="Photons", color='orange', marker='o')
axs[1, 1].set_title("Dose Rate (Photons) [mSv/h]")
axs[1, 1].set_xlabel("Batch Number")
axs[1, 1].set_ylabel("Dose Rate [mSv/h]")
axs[1, 1].set_yscale("log")
axs[1, 1].legend()
axs[1, 1].grid()

plt.tight_layout()
plt.savefig("tally_convergence_analysis.png")
plt.show()