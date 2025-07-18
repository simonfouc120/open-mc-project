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

project_root = Path(__file__).resolve().parents[7]  
sys.path.append(str(project_root))
from parameters.parameters_paths import PATH_TO_CROSS_SECTIONS
from parameters.parameters_materials import FUEL_MATERIAL, HELIUM_MATERIAL, AIR_MATERIAL, CONCRETE_MATERIAL, GRAPHITE_MATERIAL, STEEL_MATERIAL, WATER_MATERIAL
from src.utils.pre_processing.pre_processing import (remove_previous_results, mesh_tally_plane, reducing_density)
from src.utils.post_preocessing.post_processing import load_mesh_tally, load_dammage_energy_tally, load_mesh_tally_dose
from src.utils.weight_window.weight_window import plot_weight_window, create_and_apply_correction_ww_tally, get_ww_size, remove_zeros_from_ww
os.environ["OPENMC_CROSS_SECTIONS"] = PATH_TO_CROSS_SECTIONS

from src.models.model_complete_reactor import MODEL, GRAPHITE_CELL, CALCULATION_CELL

divide_factor_list = [10, 8, 6, 4, 2]
divide_factor_list = [10, 8, 6, 4]

start_time = time.time()

for itteration_number, divide_factor in enumerate(divide_factor_list):

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
    settings.particles = 1000000 # try more
    settings.source = openmc.FileSource('surface_source.h5')
    settings.photon_transport = True

    if itteration_number != 0:
        ww = openmc.hdf5_to_wws("weight_windows.h5")  
        size = get_ww_size(ww)
        ww_with_zeros_removed = remove_zeros_from_ww(weight_windows=ww)
        ww_corrected = create_and_apply_correction_ww_tally(ww=ww_with_zeros_removed, target=np.array([0.0, 400.0, -300.0]), nx=size[0], ny=size[1], nz=size[2])
        settings.weight_windows = ww_corrected
    else:
        pass

    mesh = openmc.RegularMesh().from_domain(geometry)
    mesh.dimension = (25, 25, 25)
    mesh.lower_left = (-500.0, -500.0, -500.0)
    mesh.upper_right = (500.0, 500.0, 500.0)

    energy_bounds = np.linspace(0.0, 15e6, 2) 

    wwg_neutron = openmc.WeightWindowGenerator(
        mesh=mesh,  
        energy_bounds=energy_bounds,
        particle_type='neutron', 
        method="magic"
    )

    wwg_photon = deepcopy(wwg_neutron)
    wwg_photon.particle_type = 'photon'

    settings.max_history_splits = 1_000  
    settings.weight_window_generators = [wwg_neutron, wwg_photon]

    settings.export_to_xml()

    remove_previous_results(batches_number=batches_number)
    
    time.sleep(1)
    openmc.run()

end_time = time.time()
calculation_time = end_time - start_time

# Save calculation time to a JSON file
output_json = CWD / "calculation_time.json"
with open(output_json, "w") as f:
    json.dump({"calculation_time_seconds": calculation_time}, f)

# Load the statepoint file
statepoint_file = openmc.StatePoint(f'statepoint.{batches_number}.h5')

ww = openmc.hdf5_to_wws("weight_windows.h5")  

plot_weight_window(weight_window=ww[0], index_coord=15, energy_index=0, saving_fig=True, plane="yz", particle_type='neutron')

plot_weight_window(weight_window=ww[1], index_coord=15, energy_index=0, saving_fig=True, plane="yz", particle_type='photon')