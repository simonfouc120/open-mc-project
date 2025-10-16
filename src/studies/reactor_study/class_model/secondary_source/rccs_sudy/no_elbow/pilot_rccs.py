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
                           concrete_wall_thickness=100,
                           calculation_sphere_coordinates=(-600, 200, 0), 
                           calculation_sphere_radius=50.0)

my_reactor.add_cell(surface=openmc.model.RectangularParallelepiped(-500, -400, 190, 210, -10, 10), material_name="AIR_MATERIAL",
                    cells_to_be_excluded_by=[my_reactor.concrete_walls_cells[i] for i in range(len(my_reactor.concrete_walls_cells))] + [my_reactor.air_main_cell])

model = my_reactor.model
model.export_to_xml()

plot_geometry(materials = openmc.Materials(list(my_reactor.material.values())), 
              plane="xy", origin=(0,0,0), saving_figure=False, dpi=500, height=1700, width=1700,
              pixels=(700, 700), color_by="cell")

plot_geometry(materials = openmc.Materials(list(my_reactor.material.values())), 
              plane="xy", origin=(0,0,0), saving_figure=False, dpi=500, height=1700, width=1700,
              pixels=(700, 700), color_by="material")

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
settings.particles = 500
settings.source = openmc.FileSource('surface_source.h5')
settings.photon_transport = True
settings.run_mode = "fixed source"
settings.source.particles = ["neutron", "photon"]

ww = openmc.hdf5_to_wws("weight_windows.h5")  
shape_ww = get_ww_size(weight_windows=ww)
ww_corrected = apply_correction_ww(ww=ww, correction_weight_window=create_correction_ww_tally(nx=30, ny=30, nz=30,
                                                                                              lower_left=(-700, -700, -700),
                                                                                              upper_right=(700, 700, 700),
                                                                                              target=np.array([-600, 200, 0]),
                                                                                              factor_div=1e6))
settings.weight_windows = ww_corrected

plot_weight_window(weight_window=ww[0], index_coord=15, energy_index=0, saving_fig=True, plane="xy", particle_type='neutron')
plot_weight_window(weight_window=ww[1], index_coord=15, energy_index=0, saving_fig=True, plane="xy", particle_type='photon')

model.settings = settings
model.export_to_xml()


tallys = openmc.Tallies()

### Tally dose
# XY plane
dose_mesh_xy_neutrons = mesh_tally_dose_plane(name_mesh_tally = "dose_mesh_xy_neutrons", particule_type='neutron', plane="xy",
                                      bin_number=500, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0),
                                      thickness= 20.0, coord_value=0.0)
tallys.append(dose_mesh_xy_neutrons)


dose_mesh_xy_photons = mesh_tally_dose_plane(name_mesh_tally = "dose_mesh_xy_photons", particule_type='photon', plane="xy",
                                      bin_number=500, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0),
                                      thickness= 20.0, coord_value=0.0)
tallys.append(dose_mesh_xy_photons)

# YZ plane
dose_mesh_yz_photons = mesh_tally_dose_plane(name_mesh_tally = "dose_mesh_yz_photons", particule_type='photon', plane="yz",
                                      bin_number=500, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0),
                                      thickness= 20.0, coord_value=0.0)
tallys.append(dose_mesh_yz_photons)

dose_mesh_yz_neutrons = mesh_tally_dose_plane(name_mesh_tally = "dose_mesh_yz_neutrons", particule_type='neutron', plane="yz",
                                      bin_number=500, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0),
                                      thickness= 20.0, coord_value=0.0)
tallys.append(dose_mesh_yz_neutrons)

### Tally flux
# XY plane
flux_mesh_xy_neutrons = mesh_tally_plane(name_mesh_tally = "flux_mesh_xy_neutrons", particule_type='neutron', plane="xy",
                                      bin_number=500, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0),
                                      thickness= 20.0, coord_value=0.0)

tallys.append(flux_mesh_xy_neutrons)

flux_mesh_xy_photons = mesh_tally_plane(name_mesh_tally = "flux_mesh_photons_xy", particule_type='photon', plane="xy",
                                      bin_number=500, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0),
                                      thickness= 20.0, coord_value=0.0)
tallys.append(flux_mesh_xy_photons)

# YZ plane
flux_mesh_yz_photons = mesh_tally_plane(name_mesh_tally = "flux_mesh_yz_photons", particule_type='photon', plane="yz",
                                      bin_number=500, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0),
                                      thickness= 20.0, coord_value=0.0)
tallys.append(flux_mesh_yz_photons)

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

# retracer dose en fonction de x