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
                           light_water_pool=True, 
                           light_water_length=900.0, # cm
                           light_water_height=900.0, # cm
                           cavity=False,
                           top_shielding= False,
                           slab_thickness=100,
                           concrete_wall_thickness=100,
                           calculation_sphere_coordinates=(0, 0, 550), 
                           calculation_sphere_radius=50.0)

sphere_coords = [(220, 0, 0), (290, 0, 0), (350, 0, 0), (400, 0, 0)]
excluded_cells = [my_reactor.light_water_main_cell, my_reactor.light_water_liner_main_cell]

spheres = []
for coord in sphere_coords:
    sphere = my_reactor.add_calculation_sphere(
        coordinates=coord,
        radius=10.0,
        material_name="WATER_MATERIAL",
        cells_to_be_excluded_by=excluded_cells
    )
    spheres.append(sphere)

P1, P2, P3, P4 = spheres

model = my_reactor.model
model.export_to_xml()

if __name__ == "__main__":
        
    plot_geometry(materials = openmc.Materials(list(my_reactor.material.values())), 
                plane="xz", origin=(0,0,0), saving_figure=False, dpi=500, height=1700, width=1700,
                pixels=(700, 700), color_by="cell")


    settings = openmc.Settings()
    batches_number= 1000
    settings.batches = batches_number
    settings.particles = 50000
    settings.source = openmc.FileSource('surface_source.h5')
    settings.photon_transport = True
    settings.run_mode = "fixed source"
    settings.source.particles = ["neutron", "photon"]
    settings.statepoint = {"batches": list(range(10, batches_number + 1, 10))}
    ww = openmc.hdf5_to_wws("weight_windows14.h5")  
    ww = remove_zeros_from_ww(ww)
    settings.weight_windows = apply_spherical_correction_to_weight_windows(ww, particule_type='photon', sphere_center=(400.0, 0.0, 0.0), sphere_radius=50.0)
    size_ww = get_ww_size(ww)
    plot_weight_window(weight_window=ww[0], index_coord=size_ww[1]//2, energy_index=0, saving_fig=True, plane="xz", particle_type='neutron')
    plot_weight_window(weight_window=ww[1], index_coord=size_ww[1]//2, energy_index=0, saving_fig=True, plane="xz", particle_type='photon')
    plot_weight_window(weight_window=ww[0], index_coord=size_ww[1]//2, energy_index=0, saving_fig=True, plane="xy", particle_type='neutron')
    plot_weight_window(weight_window=ww[1], index_coord=size_ww[1]//2, energy_index=0, saving_fig=True, plane="xy", particle_type='photon')

    model.settings = settings
    model.export_to_xml()

    tallys = openmc.Tallies()

    # Tallies for P1
    # Define the spheres and their names
    spheres = [P1, P2, P3, P4]
    sphere_names = ["P1", "P2", "P3", "P4"]

    for sphere, name in zip(spheres, sphere_names):
        for particle in ["photon", "neutron"]:
            tally = create_dose_rate_tally(
                name=f"flux_tally_sphere_dose_{particle}_{name}",
                particle_type=particle,
                cell=sphere
            )
            tallys.append(tally)

    mesh_tally_xy_neutrons = mesh_tally_dose_plane(name_mesh_tally = "flux_mesh_neutrons_xy", particule_type='neutron', plane="xy",
                                        bin_number=300, lower_left=(-500.0, -500.0), upper_right=(500.0, 500.0),
                                        thickness= 20.0, coord_value=0.0)
    tallys.append(mesh_tally_xy_neutrons)

    mesh_tally_xy_photons = mesh_tally_dose_plane(name_mesh_tally = "flux_mesh_photons_xy", particule_type='photon', plane="xy",
                                        bin_number=300, lower_left=(-500.0, -500.0), upper_right=(500.0, 500.0),
                                        thickness= 20.0, coord_value=0.0)
    tallys.append(mesh_tally_xy_photons)

    tallys.export_to_xml()

    remove_previous_results(batches_number=batches_number)

    openmc.run()

    remove_intermediate_files()