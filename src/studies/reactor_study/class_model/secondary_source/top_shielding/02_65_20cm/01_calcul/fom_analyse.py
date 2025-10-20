import openmc
import os 
import json
from pathlib import Path
import sys 
import numpy as np
from pilot_source import my_reactor
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

statepoint_files = sorted(Path(".").glob("statepoint.*.h5"), key=lambda f: int(f.stem.split(".")[-1]))


model = my_reactor.model

fom_photon_list = []
fom_neutron_list = []
batch = []



for statepoint in statepoint_files:
    batch_number = int(statepoint.stem.split(".")[-1])

    statepoint = openmc.StatePoint(str(statepoint))

    dose_rate_photon, dose_rate_error_photon, fom_photon = compute_dose_rate_tally(
        statepoint=statepoint,
        tally_name="flux_tally_sphere_dose_photon",
        particule_per_second=1,
        volume=Volume_cell(cell=my_reactor.calc_sphere_cell).get_volume(),
        unit="mSv/h"
    )

    dose_rate_neutron, dose_rate_error_neutron, fom_neutron = compute_dose_rate_tally(
        statepoint=statepoint,
        tally_name="flux_tally_sphere_dose_neutron",
        particule_per_second=1,
        volume=Volume_cell(cell=my_reactor.calc_sphere_cell).get_volume(),
        unit="mSv/h"
    )
    fom_photon_list.append(fom_photon)
    fom_neutron_list.append(fom_neutron)
    batch.append(batch_number)

plt.plot(batch, fom_photon_list, label="photon FOM", marker='o')
plt.plot(batch, fom_neutron_list, label="neutron FOM", marker='o')
plt.yscale("log")
plt.xlabel("Batch number")
plt.ylabel("FOM (log scale)")
plt.title("FOM vs Batch number")
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.savefig("fom_vs_batch_number.png", dpi=500) 
plt.show()