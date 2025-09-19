
import openmc
import os 
import json
from pathlib import Path
import sys 


CWD = Path(__file__).parent.resolve() 
project_root = Path(__file__).resolve().parents[7]  
sys.path.append(str(project_root))

from parameters.parameters_paths import PATH_TO_CROSS_SECTIONS, IMAGE_PATH
from parameters.parameters_materials import *
from src.utils.pre_processing.pre_processing import *
from src.utils.post_preocessing.post_processing import *
from src.models.model_complete_reactor_class import *
from src.utils.weight_window.weight_window import *

os.environ["OPENMC_CROSS_SECTIONS"] = PATH_TO_CROSS_SECTIONS

statepoint = openmc.StatePoint(f"statepoint.100.h5")

results_path = "simulation_results.json"
with open(results_path, "r") as f:
    results = json.load(f)
neutron_emission_rate = results["neutrons_emitted_per_second"]["value"]

my_reactor = Reactor_model(materials=material_dict, 
                           total_height_active_part=500.0, 
                           light_water_pool=False, 
                           slab_thickness=100,
                           concrete_wall_thickness=100,
                           calculation_sphere_coordinates=(-600, 200, 0), 
                           calculation_sphere_radius=50.0)

volume_tally_sphere = Volume_cell(cell=my_reactor.calc_sphere_cell).get_volume()

bin_mesh_volume = get_mesh_volumes(lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0), thickness=20.0, bin_number=500)

load_mesh_tally_dose(statepoint_file=statepoint, name_mesh_tally="flux_mesh_xy_neutrons", plane="xy", 
                saving_figure=True, bin_number=500, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0), 
                zoom_x=(-850, 850), zoom_y=(-850, 850), plot_error=True, particles_per_second=neutron_emission_rate,
                mesh_bin_volume=bin_mesh_volume, particule_type="neutron")

load_mesh_tally_dose(statepoint_file=statepoint, name_mesh_tally="flux_mesh_photons_xy", plane="xy", 
                saving_figure=True, bin_number=500, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0), 
                zoom_x=(-850, 850), zoom_y=(-850, 850), plot_error=True, particule_type="photon", 
                particles_per_second=neutron_emission_rate, mesh_bin_volume=bin_mesh_volume) 

# load flux tally neutron
flux_tally_neutron = statepoint.get_tally(name="flux_tally_sphere_dose")
dose_rate = flux_tally_neutron.mean.flatten()[0] * neutron_emission_rate * 1e-6 * 3600 / volume_tally_sphere
dose_rate_error = flux_tally_neutron.std_dev.flatten()[0] * neutron_emission_rate * 1e-6 * 3600 / volume_tally_sphere
print(f"Neutron dose rate in calculation sphere: {dose_rate:.3e} μSv/h ± {dose_rate_error:.3e} μSv/h ({dose_rate_error/dose_rate*100:.2f} %)")
output_json = "dose_rate_results.json"

with open(output_json, "w") as f:
    json.dump({"dose_rate": {"value": dose_rate, 
                             "error": dose_rate_error, 
                             "relative_error": dose_rate_error/dose_rate*100,
                             "unit": "microSv/h"}}, f)
