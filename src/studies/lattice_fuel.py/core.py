import openmc
import os 
import json
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
import sys 
from PIL import Image
import numpy as np
import openmc.tally_derivative
import openmc.universe

CWD = Path(__file__).parent.resolve()

project_root = Path(__file__).resolve().parents[3]  
sys.path.append(str(project_root))
from parameters.parameters_paths import PATH_TO_CROSS_SECTIONS
from parameters.parameters_materials import FUEL_MATERIAL, HELIUM_MATERIAL, AIR_MATERIAL, CONCRETE_MATERIAL, GRAPHITE_MATERIAL, STEEL_MATERIAL
from src.utils.pre_processing.pre_processing import (remove_previous_results, parallelepiped, plot_geometry, mesh_tally_xy, mesh_tally_yz, 
                                                     dammage_energy_mesh_xy, dammage_energy_mesh_yz)
from src.utils.post_preocessing.post_processing import load_mesh_tally, load_dammage_energy_tally
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
steel_outer_surface = openmc.ZCylinder(r=6.5)  # 1 cm thickness around graphite
height_top_active_part = openmc.ZPlane(z0=30.0)
height_bottom_active_part = openmc.ZPlane(z0=-30.0)

# Main cell: lattice inside graphite cylinder
main_cell = openmc.Cell(
    fill=lat,
    region=(-outer_surface & -height_top_active_part & +height_bottom_active_part)
)

# Steel shell cell: between graphite and steel cylinder
steel_shell_region = (+outer_surface & -steel_outer_surface & -height_top_active_part & +height_bottom_active_part)
steel_shell_cell = openmc.Cell(fill=STEEL_MATERIAL, region=steel_shell_region)

outer_sphere = openmc.Sphere(r=100.0, boundary_type='vacuum')

# Air region above the cylinder
air_region_above = -outer_sphere & -height_top_active_part
air_cell_above = openmc.Cell(fill=AIR_MATERIAL, region=air_region_above)

# Air region below the cylinder
air_region_below = -outer_sphere & +height_bottom_active_part
air_cell_below = openmc.Cell(fill=AIR_MATERIAL, region=air_region_below)

# Air region surrounding the steel cylinder (radially outside)
air_region_side = -outer_sphere & +steel_outer_surface & -height_top_active_part & +height_bottom_active_part
air_cell_side = openmc.Cell(fill=AIR_MATERIAL, region=air_region_side)

geometry = openmc.Geometry([main_cell, steel_shell_cell, air_cell_above, air_cell_below, air_cell_side])

plot_geometry(materials = material, plane="xy", width=15, height=15)

plot_geometry(materials = material, plane="xz", width=15, height=15)

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

# Tally for fission rate
fission_tally = openmc.Tally(name="fission_rate_tally")
fission_tally.scores = ['fission']
fission_tally.filters = [openmc.CellFilter(main_cell)]

# Tally for nu-fission (nu * fission rate)
nu_fission_tally = openmc.Tally(name="nu_fission_rate_tally")
nu_fission_tally.scores = ['nu-fission']
nu_fission_tally.filters = [openmc.CellFilter(main_cell)]

tallies = openmc.Tallies([tally, fission_tally, nu_fission_tally])

# Mesh tally for neutron flux in a specific region
mesh_tally_neutron_xy = mesh_tally_xy(name_mesh_tally = "flux_mesh_neutrons_xy", particule_type='neutron', 
                                      bin_number=400, lower_left=(-10.0, -10.0), upper_right=(10.0, 10.0),
                                      z_thickness= 4.0, z_value=0.0)
tallies.append(mesh_tally_neutron_xy)

mesh_tally_photon_xy = mesh_tally_xy(name_mesh_tally = "flux_mesh_photons_xy", particule_type='photon', 
                                     bin_number=400, lower_left=(-10.0, -10.0), upper_right=(10.0, 10.0),
                                     z_thickness= 4.0, z_value=0.0)
tallies.append(mesh_tally_photon_xy)

mesh_tally_neutron_yz = mesh_tally_yz(name_mesh_tally = "flux_mesh_neutrons_yz", particule_type='neutron', 
                                      bin_number=1000, lower_left=(-10.0, -10.0), upper_right=(10.0, 10.0))
tallies.append(mesh_tally_neutron_yz)

mesh_tally_photon_yz = mesh_tally_yz(name_mesh_tally = "flux_mesh_photons_yz", particule_type='photon', 
                                     bin_number=1000, lower_left=(-10.0, -10.0), upper_right=(10.0, 10.0))
tallies.append(mesh_tally_photon_yz)

dammage_energy_tally_xy = dammage_energy_mesh_xy(name_mesh_tally="dammage_energy_mesh_xy", bin_number=500, lower_left=(-10.0, -10.0), upper_right=(10.0, 10.0), z_value=0.0, z_thickness=4.0)
tallies.append(dammage_energy_tally_xy)

dommage_energy_tally_yz = dammage_energy_mesh_yz(name_mesh_tally="dammage_energy_mesh_yz", bin_number=500, lower_left=(-10.0, -10.0), upper_right=(10.0, 10.0), x_value=0.0, x_thickness=1.0)
tallies.append(dommage_energy_tally_yz)

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
mean_flux = tally.mean.flatten().tolist()
std_flux = tally.std_dev.flatten().tolist()

# Get fission rate and nu-bar
fission_tally_result = statepoint_file.get_tally(name="fission_rate_tally")
nu_fission_tally_result = statepoint_file.get_tally(name="nu_fission_rate_tally")

fission_rate = float(fission_tally_result.mean.flatten()[0])
nu_fission_rate = float(nu_fission_tally_result.mean.flatten()[0])
nu_bar = nu_fission_rate / fission_rate if fission_rate != 0 else float('nan')

results = {
    "mean_flux": mean_flux,
    "std_flux": std_flux,
    "fission_rate": fission_rate,
    "nu_fission_rate": nu_fission_rate,
    "nu_bar": nu_bar
}

with open(CWD / "simulation_results.json", "w") as f:
    json.dump(results, f, indent=2)

### mesh tallty ####

load_mesh_tally(cwd = CWD, statepoint_file = statepoint_file, name_mesh_tally="flux_mesh_neutrons_xy")

load_mesh_tally(cwd = CWD, statepoint_file = statepoint_file, name_mesh_tally="flux_mesh_photons_xy")

load_mesh_tally(cwd = CWD, statepoint_file = statepoint_file, name_mesh_tally="flux_mesh_neutrons_yz", bin_number=1000,
                lower_left=(-40.0, -40.0), upper_right=(40.0, 40.0), zoom_x=(-40, 40), zoom_y=(-40, 40), plane="yz")

load_mesh_tally(cwd = CWD, statepoint_file = statepoint_file, name_mesh_tally="flux_mesh_photons_yz", bin_number=1000,
                lower_left=(-40.0, -40.0), upper_right=(40.0, 40.0), zoom_x=(-40, 40), zoom_y=(-40, 40), plane="yz")

load_dammage_energy_tally(cwd = CWD, statepoint_file = statepoint_file, name_mesh_tally="dammage_energy_mesh_xy",
                         bin_number=500, lower_left=(-10.0, -10.0), upper_right=(10.0, 10.0), 
                         zoom_x=(-10, 10), zoom_y=(-10.0, 10.0), plane="xy")

load_dammage_energy_tally(cwd = CWD, statepoint_file = statepoint_file, name_mesh_tally="dammage_energy_mesh_yz",
                         bin_number=500, lower_left=(-10.0, -10.0), upper_right=(10.0, 10.0), 
                         zoom_x=(-10, 10), zoom_y=(-10.0, 10.0), plane="yz")