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
                           calculation_sphere_coordinates=(0, 0, 500), 
                           calculation_sphere_radius=50.0,
                           thickness_lead_top_shielding=20.0, 
                           thickness_b4c_top_shielding=65.0)

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

    itterations_number = []
    fom_neutron_list = []
    fom_photon_list = []


    for itteration in range(1, 9):

        settings = openmc.Settings()
        batches_number= 100
        settings.batches = batches_number
        settings.particles = 50000
        settings.source = openmc.FileSource('surface_source.h5')
        settings.photon_transport = True
        settings.run_mode = "fixed source"
        settings.source.particles = ["neutron", "photon"]

        ww = openmc.hdf5_to_wws(f"weight_windows{itteration}.h5")

        settings.weight_windows = apply_spherical_correction_to_weight_windows(ww, sphere_center=(0.0, 0.0, 500.0), sphere_radius=50.0)
        size_ww = get_ww_size(ww)

        model.settings = settings
        model.export_to_xml()

        tallys = openmc.Tallies()

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

        statepoint = openmc.StatePoint(f"statepoint.{batches_number}.h5")

        dose_rate_photon, dose_rate_error_photon, fom_photon = compute_dose_rate_tally(
            statepoint=statepoint,
            tally_name="flux_tally_sphere_dose_photon",
            particule_per_second=neutron_emission_rate,
            volume=Volume_cell(cell=my_reactor.calc_sphere_cell).get_volume(),
            unit="mSv/h"
        )

        dose_rate_neutron, dose_rate_error_neutron, fom_neutron = compute_dose_rate_tally(
            statepoint=statepoint,
            tally_name="flux_tally_sphere_dose_neutron",
            particule_per_second=neutron_emission_rate,
            volume=Volume_cell(cell=my_reactor.calc_sphere_cell).get_volume(),
            unit="mSv/h"
        )
        itterations_number.append(itteration)
        fom_neutron_list.append(fom_neutron)
        fom_photon_list.append(fom_photon)

    plt.figure(figsize=(8,6))
    plt.plot(itterations_number, fom_neutron_list, marker='o', label='Neutrons FOM')
    plt.plot(itterations_number, fom_photon_list, marker='o', label='Photons FOM')
    plt.yscale('log')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Figure of Merit')
    plt.title('FOM vs Number of Iterations')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig("fom_vs_iterations.png", dpi=300)
    plt.show()