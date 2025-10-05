import openmc
import os 
import json
from pathlib import Path
import sys 
from copy import deepcopy

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

material_dict["FUEL_UO2_MATERIAL"].remove_element("U")

my_reactor = Reactor_model(materials=material_dict, 
                           total_height_active_part=500.0, 
                           light_water_pool=False, 
                           slab_thickness=100,
                           concrete_wall_thickness=150,
                           calculation_sphere_coordinates=(-650, 0, 0), 
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
settings.particles = 2000000
settings.source = openmc.FileSource('surface_source.h5')
settings.photon_transport = True
settings.run_mode = "fixed source"
settings.source.particles = ["neutron", "photon"]
settings.statepoint = {"batches": list(range(10, batches_number + 1, 10))}

ww = openmc.hdf5_to_wws("weight_windows.h5")  

shape_ww = get_ww_size(weight_windows=ww, particule_type='photon')

lower_left = ww[0].mesh.bounding_box.lower_left
upper_right = ww[0].mesh.bounding_box.upper_right
mesh_bounds = ((lower_left[0], upper_right[0]), (lower_left[1], upper_right[1]), (lower_left[2], upper_right[2]))  # mm or cm (cohérent avec tes unités)

shape = (shape_ww[0] , shape_ww[1], shape_ww[2])  # number of voxels in each direction
sphere_center = (-650.0, 0.0, 0.0)
sphere_radius = 50.0
correction_matrix = make_oriented_importance(mesh_bounds, shape, sphere_center, sphere_radius,
                                        I_min=1e-5, I_max=1e-1, lambda_radial=0.005,
                                        beam_dir=None, angular_power=4.0, alpha=0.02)
# sauvegarde
ww_corrected = apply_correction_ww(ww=ww, correction_weight_window=correction_matrix)
settings.weight_windows = ww_corrected

plot_weight_window(weight_window=ww[0], index_coord=15, energy_index=0, saving_fig=True, plane="xy", particle_type='neutron')
plot_weight_window(weight_window=ww[1], index_coord=15, energy_index=0, saving_fig=True, plane="xy", particle_type='photon')

MODEL.settings = settings
MODEL.export_to_xml()

tallys = openmc.Tallies()

mesh_tally_xy_neutrons = mesh_tally_dose_plane(name_mesh_tally = "flux_mesh_neutrons_xy", particule_type='neutron', plane="xy",
                                      bin_number=250, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0),
                                      thickness= 20.0, coord_value=0.0)
tallys.append(mesh_tally_xy_neutrons)

mesh_tally_xy_photons = mesh_tally_dose_plane(name_mesh_tally = "flux_mesh_photons_xy", particule_type='photon', plane="xy",
                                      bin_number=250, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0),
                                      thickness= 20.0, coord_value=0.0)
tallys.append(mesh_tally_xy_photons)

tally_sphere_dose_photon = create_dose_rate_tally(
    name="flux_tally_sphere_dose_photon",
    particle_type="photon",
    cell=my_reactor.calc_sphere_cell
)
tallys.append(tally_sphere_dose_photon)

tally_sphere_dose_neutron = create_dose_rate_tally(
    name="flux_tally_sphere_dose_neutron",
    particle_type="neutron",
    cell=my_reactor.calc_sphere_cell
)
tallys.append(tally_sphere_dose_neutron)

tallys.export_to_xml()

remove_previous_results(batches_number=batches_number)

openmc.run()

remove_intermediate_files()