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
project_root = Path(__file__).resolve().parents[5]  
sys.path.append(str(project_root))

from parameters.parameters_paths import PATH_TO_CROSS_SECTIONS, IMAGE_PATH
from parameters.parameters_materials import *
from src.utils.pre_processing.pre_processing import *
from src.utils.post_preocessing.post_processing import *
from src.models.model_complete_reactor_class import *
os.environ["OPENMC_CROSS_SECTIONS"] = PATH_TO_CROSS_SECTIONS

# material_dict["FUEL_UO2_MATERIAL"].remove_element("U")

for material in material_dict.values():
    if material in [WATER_MATERIAL, HEAVY_WATER_MATERIAL, CONCRETE_MATERIAL]:
        material.set_density('g/cm3', EPSILON_DENSITY)

my_reactor = Reactor_model(materials=material_dict, 
                           total_height_active_part=500.0, 
                           light_water_pool=True, 
                           slab_thickness=100,
                           concrete_wall_thickness=50,
                           calculation_sphere_coordinates=(0, -350, 0), 
                           r_pin_fuel=1.5)

# my_reactor.add_cell(surface=openmc.Sphere(y0=230, z0=-200, r=15.0),
#                     material_name="STEEL_MATERIAL",
#                     cells_to_be_excluded_by=[my_reactor.air_main_cell, my_reactor.light_water_main_cell, my_reactor.light_water_liner_main_cell])

# my_reactor.add_cell(surface=openmc.model.RectangularParallelepiped(xmin=-450, xmax=-400, ymin=-15, ymax=15, zmin=-15, zmax=15),
#                     material_name="AIR_MATERIAL",
#                     cells_to_be_excluded_by=[my_reactor.concrete_walls_cells[i] for i in range(len(my_reactor.concrete_walls_cells))] + [my_reactor.air_main_cell])


# my_reactor.add_wall_concrete(concrete_wall_coordinates=(0,0,0), dx=800.0, dy=800.0, dz=1000.0, 
#                              cells_to_exclude=[my_reactor.light_water_main_cell, 
#                                                my_reactor.light_water_liner_main_cell, 
#                                                my_reactor.steel_liner_main_cell])

P1 = my_reactor.add_calculation_sphere(coordinates=(0, 350, 0), radius=10.0)
P2 = my_reactor.add_calculation_sphere(coordinates=(0, 400, 0), radius=10.0)


MODEL = my_reactor.model
MODEL.export_to_xml()

# plot_geometry(materials = openmc.Materials(list(my_reactor.material.values())), plane="yz", saving_figure=True, dpi=500, height=1000, width=1000)
# plot_geometry(materials = openmc.Materials(list(my_reactor.material.values())), plane="xy", saving_figure=True, dpi=500, height=400, width=400)

# plot_geometry(materials = openmc.Materials(list(my_reactor.material.values())), plane="xy", origin=(0,0,-200), saving_figure=True, dpi=500, height=700, width=700)
plot_geometry(materials = openmc.Materials(list(my_reactor.material.values())), 
              plane="xy", origin=(0,0,0), saving_figure=True, dpi=500, height=1700, width=1700,
              suffix="_test_no_fission", pixels=(1700,1700))

# fonction material pas de fission

tallys = openmc.Tallies()

mesh_tally_xy_neutrons = mesh_tally_plane(name_mesh_tally = "flux_mesh_xy_neutrons", particule_type='neutron', plane="xy",
                                      bin_number=500, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0),
                                      thickness= 20.0, coord_value=0.0)
tallys.append(mesh_tally_xy_neutrons)


mesh_tally_xy_photons = mesh_tally_plane(name_mesh_tally = "flux_mesh_photons_xy", particule_type='photon', plane="xy",
                                      bin_number=500, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0),
                                      thickness= 20.0, coord_value=0.0)
tallys.append(mesh_tally_xy_photons)


flux_tally_neutron = openmc.Tally(name="flux_tally_neutron")
flux_tally_neutron.scores = ['flux']
flux_tally_neutron.filters = [openmc.ParticleFilter("neutron"), openmc.CellFilter(P1)]

# flux_tally_neutron.filters = [openmc.ParticleFilter("neutron"), openmc.CellFilter(my_reactor.calc_sphere_cell)]

tallys.append(flux_tally_neutron)

tallys.export_to_xml()

# run the simulation

settings = openmc.Settings()
batches_number= 50
settings.batches = batches_number
settings.inactive = 10
settings.particles = 500
settings.source = openmc.IndependentSource()
settings.source.space = openmc.stats.Point((0, 0, 0))
settings.source.particle = 'neutron'
settings.photon_transport = True
settings.source.angle = openmc.stats.Isotropic()  
MODEL.settings = settings
settings.export_to_xml()
MODEL.export_to_xml()

openmc.run()

statepoint = openmc.StatePoint(f"statepoint.{batches_number}.h5")

load_mesh_tally(cwd=CWD, statepoint_file=statepoint, name_mesh_tally="flux_mesh_xy_neutrons", plane="xy", 
                saving_figure=False, bin_number=500, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0), 
                zoom_x=(-850, 850), zoom_y=(-850, 850), plot_error=True)

load_mesh_tally(cwd=CWD, statepoint_file=statepoint, name_mesh_tally="flux_mesh_photons_xy", plane="xy", 
                saving_figure=False, bin_number=500, lower_left=(-850.0, -850.0), upper_right=(850.0, 850.0), 
                zoom_x=(-850, 850), zoom_y=(-850, 850), plot_error=True, particule_type="photon") 


# load flux tally neutron
flux_tally_neutron = statepoint.get_tally(name="flux_tally_neutron")
print(f"Neutron flux tally: {flux_tally_neutron.mean[0][0][0]} particles/cm^2/p-source")