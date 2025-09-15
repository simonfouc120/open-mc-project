import openmc
import os 
import json
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
import sys 
from PIL import Image
import numpy as np
from copy import deepcopy

CWD = Path(__file__).parent.resolve() 
project_root = Path(__file__).resolve().parents[6]  
sys.path.append(str(project_root))

from parameters.parameters_paths import PATH_TO_CROSS_SECTIONS, IMAGE_PATH
from parameters.parameters_materials import *
from src.utils.pre_processing.pre_processing import *
from src.utils.post_preocessing.post_processing import *
from src.models.model_complete_reactor_class import *
from src.utils.weight_window.weight_window import *

for material in material_dict.values():
    material= reducing_density(material, factor=10)

os.environ["OPENMC_CROSS_SECTIONS"] = PATH_TO_CROSS_SECTIONS

material_dict["FUEL_UO2_MATERIAL"].remove_element("U")

my_reactor = Reactor_model(materials=material_dict, 
                           total_height_active_part=500.0, 
                           light_water_pool=False, 
                           slab_thickness=100,
                           concrete_wall_thickness=100,
                           calculation_sphere_coordinates=(-600, 0, 0), 
                           calculation_sphere_radius=50.0)

MODEL = my_reactor.model
MODEL.export_to_xml()

plot_geometry(materials = openmc.Materials(list(my_reactor.material.values())), 
              plane="xy", origin=(0,0,0), saving_figure=False, dpi=500, height=1700, width=1700,
              pixels=(700, 700))


results_path = "simulation_results.json"
with open(results_path, "r") as f:
    results = json.load(f)
neutron_emission_rate = results["neutrons_emitted_per_second"]["value"]

# fonction material pas de fission

tallys = openmc.Tallies()

# run the simulation

settings = openmc.Settings()
batches_number= 100
settings.batches = batches_number
settings.particles = 50000
settings.source = openmc.FileSource('surface_source.h5')
settings.photon_transport = True
settings.run_mode = "fixed source"
settings.source.particles = ["neutron", "photon"]

ww = openmc.hdf5_to_wws("weight_windows.h5")  
shape_ww = get_ww_size(weight_windows=ww)
ww_corrected = apply_correction_ww(ww=ww, correction_weight_window=create_correction_ww_tally(nx=30, ny=30, nz=30,
                                                                                              lower_left=(-700, -700, -700),
                                                                                              upper_right=(700, 700, 700),
                                                                                              target=np.array([-600, 0, 0])))
settings.weight_windows = ww_corrected

plot_weight_window(weight_window=ww[0], index_coord=15, energy_index=0, saving_fig=True, plane="xy", particle_type='neutron')
plot_weight_window(weight_window=ww[1], index_coord=15, energy_index=0, saving_fig=True, plane="xy", particle_type='photon')

MODEL.settings = settings
MODEL.export_to_xml()


tallys = openmc.Tallies()

mesh_tally_xy_neutrons = mesh_tally_dose_plane(name_mesh_tally = "flux_mesh_xy_neutrons", particule_type='neutron', plane="xy",
                                      bin_number=500, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0),
                                      thickness= 20.0, coord_value=0.0)
tallys.append(mesh_tally_xy_neutrons)


mesh_tally_xy_photons = mesh_tally_dose_plane(name_mesh_tally = "flux_mesh_photons_xy", particule_type='photon', plane="xy",
                                      bin_number=500, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0),
                                      thickness= 20.0, coord_value=0.0)
tallys.append(mesh_tally_xy_photons)

tallys.export_to_xml()

tally_sphere_dose = openmc.Tally(name="flux_tally_sphere_dose")
tally_sphere_dose.scores = ['flux']
energy_bins, dose_coeffs = openmc.data.dose_coefficients(particle="neutron", geometry='ISO')
tally_sphere_dose.filters = [openmc.CellFilter(my_reactor.calc_sphere_cell), openmc.EnergyFunctionFilter(energy_bins, dose_coeffs, interpolation='cubic')]
tallys.append(tally_sphere_dose)

volume_tally_sphere = Volume_cell(cell=my_reactor.calc_sphere_cell).get_volume()

tallys.export_to_xml()

remove_previous_results(CWD)

openmc.run()

statepoint = openmc.StatePoint(f"statepoint.{batches_number}.h5")

bin_mesh_volume = get_mesh_volumes(lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0), thickness=20.0, bin_number=500)

load_mesh_tally_dose(cwd=CWD, statepoint_file=statepoint, name_mesh_tally="flux_mesh_xy_neutrons", plane="xy", 
                saving_figure=False, bin_number=500, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0), 
                zoom_x=(-850, 850), zoom_y=(-850, 850), plot_error=True, particles_per_second=neutron_emission_rate,
                mesh_bin_volume=bin_mesh_volume, particule_type="neutron")

load_mesh_tally_dose(cwd=CWD, statepoint_file=statepoint, name_mesh_tally="flux_mesh_photons_xy", plane="xy", 
                saving_figure=False, bin_number=500, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0), 
                zoom_x=(-850, 850), zoom_y=(-850, 850), plot_error=True, particule_type="photon", 
                particles_per_second=neutron_emission_rate, mesh_bin_volume=bin_mesh_volume) 

# load flux tally neutron
flux_tally_neutron = statepoint.get_tally(name="flux_tally_sphere_dose")
dose_rate = flux_tally_neutron.mean.flatten()[0] * neutron_emission_rate * 1e-6 * 3600 / volume_tally_sphere
dose_rate_error = flux_tally_neutron.std_dev.flatten()[0] * neutron_emission_rate * 1e-6 * 3600 / volume_tally_sphere
print(f"Neutron dose rate in calculation sphere: {dose_rate:.3e} Sv/s ± {dose_rate_error:.3e} Sv/s")
output_json = CWD / "dose_rate_results.json"
with open(output_json, "w") as f:
    json.dump({"dose_rate_sievert_per_hour": dose_rate, "dose_rate_error_sievert_per_hour": dose_rate_error}, f)