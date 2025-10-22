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
project_root = Path(__file__).resolve().parents[8]  
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
                           calculation_sphere_coordinates=(0, 0, 510), 
                           calculation_sphere_radius=50.0,
                           thickness_lead_top_shielding=25.0, 
                           thickness_b4c_top_shielding=75.0)

model = my_reactor.model
if __name__ == "__main__":
    model.export_to_xml()

    plot_geometry(materials = openmc.Materials(list(my_reactor.material.values())), 
                plane="xy", origin=(0,0,0), saving_figure=False, dpi=500, height=1700, width=1700,
                pixels=(700, 700))

    plot_geometry(materials = openmc.Materials(list(my_reactor.material.values())), 
                plane="xz", origin=(0,0,0), saving_figure=False, dpi=500, height=1700, width=1700,
                pixels=(700, 700))

    plot_geometry(materials = openmc.Materials(list(my_reactor.material.values())), 
                plane="xz", origin=(0,0,0), saving_figure=False, dpi=500, height=1700, width=1700,
                pixels=(700, 700), color_by="cell")


    results_path = "simulation_results.json"
    with open(results_path, "r") as f:
        results = json.load(f)
    neutron_emission_rate = results["neutrons_emitted_per_second"]["value"]

    tallys = openmc.Tallies()

    # run the simulation

    settings = openmc.Settings()
    batches_number= 1000
    settings.batches = batches_number
    settings.particles = 50000
    settings.source = openmc.FileSource('surface_source.h5')
    settings.photon_transport = True
    settings.run_mode = "fixed source"
    settings.source.particles = ["neutron", "photon"]
    settings.statepoint = {"batches": list(range(10, batches_number + 1, 10))}

    ww = openmc.hdf5_to_wws("weight_windows4.h5")  

    settings.weight_windows = apply_spherical_correction_to_weight_windows(ww, particule_type='photon', sphere_center=(0.0, 0.0, 500.0), sphere_radius=50.0)
    size_ww = get_ww_size(ww)
    plot_weight_window(weight_window=ww[0], index_coord=size_ww[1]//2, energy_index=0, saving_fig=True, plane="xz", particle_type='neutron')
    plot_weight_window(weight_window=ww[1], index_coord=size_ww[1]//2, energy_index=0, saving_fig=True, plane="xz", particle_type='photon')

    model.settings = settings
    model.export_to_xml()

    tallys = openmc.Tallies()

    mesh_tally_xy_neutrons = mesh_tally_dose_plane(name_mesh_tally = "flux_mesh_neutrons_xy", particule_type='neutron', plane="xy",
                                        bin_number=500, lower_left=(-600.0, -600.0), upper_right=(600.0, 600.0),
                                        thickness= 20.0, coord_value=455.0)
    tallys.append(mesh_tally_xy_neutrons)


    mesh_tally_xy_photons = mesh_tally_dose_plane(name_mesh_tally = "flux_mesh_photons_xy", particule_type='photon', plane="xy",
                                        bin_number=250, lower_left=(-600.0, -600.0), upper_right=(600.0, 600.0),
                                        thickness= 20.0, coord_value=455.0)
    tallys.append(mesh_tally_xy_photons)

    mesh_tally_xz_neutrons = mesh_tally_dose_plane(name_mesh_tally = "flux_mesh_neutrons_xz", particule_type='neutron', plane="xz",
                                        bin_number=250, lower_left=(-600.0, -600.0), upper_right=(600.0, 600.0),
                                        thickness= 10.0, coord_value=0.0)
    tallys.append(mesh_tally_xz_neutrons)


    mesh_tally_xz_photons = mesh_tally_dose_plane(name_mesh_tally = "flux_mesh_photons_xz", particule_type='photon', plane="xz",
                                        bin_number=250, lower_left=(-600.0, -600.0), upper_right=(600.0, 600.0),
                                        thickness= 10.0, coord_value=0.0)
    tallys.append(mesh_tally_xz_photons)

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