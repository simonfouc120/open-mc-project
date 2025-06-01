import openmc
import os 
import json
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
import sys 
from PIL import Image
import numpy as np
import openmc.universe

CWD = Path(__file__).parent.resolve()

project_root = Path(__file__).resolve().parents[3]  
sys.path.append(str(project_root))
from parameters.parameters_paths import PATH_TO_CROSS_SECTIONS
from parameters.parameters_materials import FUEL_MATERIAL, HELIUM_MATERIAL, AIR_MATERIAL, CONCRETE_MATERIAL, GRAPHITE_MATERIAL, STEEL_MATERIAL
from src.utils.pre_processing.pre_processing import remove_previous_results, parallelepiped, plot_geometry, mesh_tally
from src.utils.post_preocessing.post_processing import load_mesh_tally
os.environ["OPENMC_CROSS_SECTIONS"] = PATH_TO_CROSS_SECTIONS

material = openmc.Materials([FUEL_MATERIAL, HELIUM_MATERIAL, AIR_MATERIAL, CONCRETE_MATERIAL, GRAPHITE_MATERIAL, STEEL_MATERIAL])


r_pin = openmc.ZCylinder(r=0.3)
fuel_cell = openmc.Cell(fill=FUEL_MATERIAL, region=-r_pin)
graphite_cell_2 = openmc.Cell(fill=GRAPHITE_MATERIAL, region=+r_pin)
pin_universe = openmc.Universe(cells=(fuel_cell, graphite_cell_2))

r_big_pin = openmc.ZCylinder(r=0.5)
fuel2_cell = openmc.Cell(fill=FUEL_MATERIAL, region=-r_big_pin)
graphite_cell_2 = openmc.Cell(fill=GRAPHITE_MATERIAL, region=+r_big_pin)
big_pin_universe = openmc.Universe(cells=(fuel2_cell, graphite_cell_2))

all_graphite_cell = openmc.Cell(fill=GRAPHITE_MATERIAL)
outer_universe = openmc.Universe(cells=(all_graphite_cell,))

lat = openmc.HexLattice()
lat.center = (0., 0.)
lat.pitch = (1.25,)   # pitch en cm
lat.outer = outer_universe

lat.universes = [
    [big_pin_universe] * 18,    # 1st ring
    [pin_universe] * 12,    # 2nd ring
    [big_pin_universe] * 6, # 3rd ring
    [big_pin_universe]      # 4th ring
]

outer_surface = openmc.ZCylinder(r=5.5)
height_top = openmc.ZPlane(z0=30.0)
height_bottom = openmc.ZPlane(z0=-30.0)
main_cell = openmc.Cell(
    fill=lat,
    region=(-outer_surface & -height_top & +height_bottom)
)

outer_sphere = openmc.Sphere(r=100.0, boundary_type='vacuum')

# Air region above the cylinder
air_region_above = -outer_sphere & -height_top
air_cell_above = openmc.Cell(fill=AIR_MATERIAL, region=air_region_above)

# Air region below the cylinder
air_region_below = -outer_sphere & +height_bottom
air_cell_below = openmc.Cell(fill=AIR_MATERIAL, region=air_region_below)

# Air region surrounding the cylinder (radially outside)
air_region_side = -outer_sphere & +outer_surface & -height_top & +height_bottom
air_cell_side = openmc.Cell(fill=AIR_MATERIAL, region=air_region_side)

geometry = openmc.Geometry([main_cell, air_cell_above, air_cell_below, air_cell_side])

plot_geometry(materials = material, plane="xy", width=12, height=12)

plot_geometry(materials = material, plane="xz", width=12, height=12)

plot_geometry(materials = material, plane="yz", width=70, height=70)

# Calcul de criticité simple 
settings = openmc.Settings()
batches_number= 100
settings.batches = batches_number
settings.inactive = 20
settings.particles = 10000
settings.source = openmc.Source()
settings.source.space = openmc.stats.Point((0, 0, 0))
settings.source.particle = 'neutron'
settings.photon_transport = True
settings.source.angle = openmc.stats.Isotropic()

# tally pour le flux dans le détecteur
tally = openmc.Tally(name="flux_tally")
tally.scores = ['flux']
tally.filters = [openmc.CellFilter(main_cell)]
tallies = openmc.Tallies([tally])

# Mesh tally for neutron flux in a specific region
mesh_tally_neutron = mesh_tally(name_mesh_tally = "flux_mesh_neutrons", particule_type='neutron', bin_number=400, lower_left=(-10.0, -10.0), upper_right=(10.0, 10.0))
tallies.append(mesh_tally_neutron)

mesh_tally_photon = mesh_tally(name_mesh_tally = "flux_mesh_photons", particule_type='photon', bin_number=400, lower_left=(-10.0, -10.0), upper_right=(10.0, 10.0))
tallies.append(mesh_tally_photon)


geometry.export_to_xml()
material.export_to_xml()
settings.export_to_xml()
tallies.export_to_xml()

# Run the simulation
remove_previous_results(batches_number=batches_number)
openmc.run()

# Load the statepoint file
statepoint_file = openmc.StatePoint(f'statepoint.{batches_number}.h5')
tally = statepoint_file.get_tally(name="flux_tally")
mean_flux = tally.mean.flatten()
std_flux = tally.std_dev.flatten()
print(f"Mean flux: {mean_flux}")
print(f"Standard deviation of flux: {std_flux}")


### mesh tallty ####
# Récupérer le tally du maillage

load_mesh_tally(cwd = CWD, statepoint_file = statepoint_file, name_mesh_tally="flux_mesh_neutrons")

load_mesh_tally(cwd = CWD, statepoint_file = statepoint_file, name_mesh_tally="flux_mesh_photons")