import openmc
import os 
from pathlib import Path
import sys 
import numpy as np


CWD = Path(__file__).parent.resolve() 
project_root = Path(__file__).resolve().parents[2]  
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

MODEL = my_reactor.model
MODEL.export_to_xml()

plot_geometry(materials = openmc.Materials(list(my_reactor.material.values())), 
              plane="xy", origin=(0,0,0), saving_figure=True, dpi=500, height=1700, width=1700,
              pixels=(700, 700), color_by="cell")

plot_geometry(materials = openmc.Materials(list(my_reactor.material.values())), 
              plane="xy", origin=(0,0,0), saving_figure=True, dpi=500, height=1700, width=1700,
              pixels=(700, 700), color_by="material", suffix="_by_material")

statepoint = openmc.StatePoint(f"statepoint.100_2.h5")
neutron_emission_rate = 1.0e18
bin_mesh_volume = get_mesh_volumes(lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0), thickness=20.0, bin_number=500)
# flux_over_geometry(statepoint_file=statepoint, name_mesh_tally="flux_mesh_photons_xy", plane="xy", 
#                 saving_figure=True, bin_number=500, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0), 
#                 zoom_x=(-850, 850), zoom_y=(-850, 850), plot_error=True, particule_type="photon", 
#                 model=MODEL) 

# flux_over_geometry(statepoint_file=statepoint, name_mesh_tally="flux_mesh_photons_xy", plane="xy", 
#                 saving_figure=True, bin_number=500, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0), 
#                 zoom_x=(-850, 850), zoom_y=(-850, 850), plot_error=True, particule_type="photon", 
#                 model=MODEL) 

# flux_over_geometry(statepoint_file=statepoint, name_mesh_tally="flux_mesh_photons_xy", plane="xy", 
#                 saving_figure=True, bin_number=500, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0), 
#                 zoom_x=(-600, -350), zoom_y=(100, 350), plot_error=True, particule_type="photon", 
#                 model=MODEL, pixels_model_geometry=1_000_000, suffix_saving="_zoom") 

# flux_over_geometry(statepoint_file=statepoint, name_mesh_tally="flux_mesh_xy_neutrons", plane="xy", 
#                 saving_figure=True, bin_number=500, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0), 
#                 zoom_x=(-850, 850), zoom_y=(-850, 850), plot_error=True, particule_type="neutron", 
#                 model=MODEL, pixels_model_geometry=1_000_000) 

# flux_over_geometry(statepoint_file=statepoint, name_mesh_tally="flux_mesh_xy_neutrons", plane="xy", 
#                 saving_figure=True, bin_number=500, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0), 
#                 zoom_x=(-600, -350), zoom_y=(100, 350), plot_error=True, particule_type="neutron", 
#                 model=MODEL, pixels_model_geometry=1_000_000, suffix_saving="_zoom") 

dose_over_geometry(statepoint_file=statepoint, name_mesh_tally="flux_mesh_photons_xy", plane="xy", 
                saving_figure=True, bin_number=500, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0), 
                zoom_x=(-850, 850), zoom_y=(-850, 850), plot_error=True, particule_type="photon", 
                model=MODEL, particles_per_second=neutron_emission_rate)

dose_over_geometry(statepoint_file=statepoint, name_mesh_tally="flux_mesh_xy_neutrons", plane="xy", 
                saving_figure=True, bin_number=500, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0), 
                zoom_x=(-850, 850), zoom_y=(-850, 850), plot_error=True, particule_type="neutron", 
                model=MODEL, particles_per_second=neutron_emission_rate, suffix_saving="_zoom")

dose_over_geometry(statepoint_file=statepoint, name_mesh_tally="flux_mesh_yz_photons", plane="yz",
                saving_figure=True, bin_number=500, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0), 
                zoom_x=(-850, 850), zoom_y=(-850, 850), plot_error=True, particule_type="photon", 
                model=MODEL, particles_per_second=neutron_emission_rate)
