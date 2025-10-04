import openmc
import os 
import json
from pathlib import Path
import sys 
import numpy as np

CWD = Path(__file__).parent.resolve() 
project_root = Path(__file__).resolve().parents[7]  
sys.path.append(str(project_root))

from parameters.parameters_paths import PATH_TO_CROSS_SECTIONS, IMAGE_PATH
from parameters.parameters_materials import *
from src.utils.pre_processing.pre_processing import *
from src.utils.post_preocessing.post_processing import *
from src.models.model_complete_reactor_class import *

os.environ["OPENMC_CROSS_SECTIONS"] = PATH_TO_CROSS_SECTIONS

material_dict["FUEL_UO2_MATERIAL"].remove_element("U")

statepoint = openmc.StatePoint(f"statepoint.050.h5")

results_path = "simulation_results.json"
with open(results_path, "r") as f:
    results = json.load(f)
neutron_emission_rate = results["neutrons_emitted_per_second"]["value"]


my_reactor = Reactor_model(materials=material_dict, 
                           total_height_active_part=500.0, 
                           light_water_pool=False, 
                           slab_thickness=100,
                           concrete_wall_thickness=150,
                           calculation_sphere_coordinates=(-650, 0, 0),
                           calculation_sphere_radius=50)

MODEL = my_reactor.model
MODEL.export_to_xml()

dose_rate_photon, dose_rate_error_photon = compute_dose_rate_tally(
    statepoint=statepoint,
    tally_name="flux_tally_sphere_dose_photon",
    particule_per_second=neutron_emission_rate,
    volume=Volume_cell(cell=my_reactor.calc_sphere_cell).get_volume(),
    unit="mSv/h"
)

dose_rate_neutron, dose_rate_error_neutron = compute_dose_rate_tally(
    statepoint=statepoint,
    tally_name="flux_tally_sphere_dose_neutron",
    particule_per_second=neutron_emission_rate,
    volume=Volume_cell(cell=my_reactor.calc_sphere_cell).get_volume(),
    unit="mSv/h"
)


save_tally_result_to_json(
    tally_name="dose_rate_photon",
    value=dose_rate_photon,
    error=dose_rate_error_photon,
    unit="mSv/h",
    filename="results.json"
)

save_tally_result_to_json(
    tally_name="dose_rate_neutron",
    value=dose_rate_neutron,
    error=dose_rate_error_neutron,
    unit="mSv/h",
    filename="results.json"
)

mesh_tally_photons = mesh_tally_data(statepoint, "flux_mesh_photons_xy", "XY", "photon")
mesh_tally_photons.plot_dose(axis_two_index=250, 
                             particles_per_second=neutron_emission_rate, 
                             x_lim=(-650, 0),
                             y_lim=(1e3, 1e14),
                             save_fig=True,
                             radiological_area=True,
                             geometrical_limit=[(-400, 'Concrete wall start'), (-550, 'Concrete wall end')],
                             fig_name="dose_plot_photons.png")
mesh_tally_photons.plot_dose_map(model=MODEL, saving_figure=True, plot_error=True, 
                                 particles_per_second=neutron_emission_rate, radiological_area=False)   

mesh_tally_neutrons = mesh_tally_data(statepoint, "flux_mesh_neutrons_xy", "XY", "neutron")
mesh_tally_neutrons.plot_dose(axis_two_index=250, 
                             particles_per_second=neutron_emission_rate, 
                             x_lim=(-650, 0),
                             y_lim=(1e3, 1e14),
                             save_fig=True,
                             radiological_area=True,
                             geometrical_limit=[(-400, 'Concrete wall start'), (-550, 'Concrete wall end')],
                             fig_name="dose_plot_neutrons.png")
mesh_tally_neutrons.plot_dose_map(model=MODEL, saving_figure=True, plot_error=True, 
                                 particles_per_second=neutron_emission_rate, radiological_area=False)   
