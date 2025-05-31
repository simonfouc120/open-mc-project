
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

project_root = Path(__file__).resolve().parents[3]  # remonte de src/studies/simulation_cs_137
sys.path.append(str(project_root))
from parameters.parameters_paths import PATH_TO_CROSS_SECTIONS
from parameters.parameters_materials import FUEL_MATERIAL, CDTE_MATERIAL, AIR_MATERIAL, CONCRETE_MATERIAL, GRAPHITE_MATERIAL, URANIUM_MATERIAL
from src.utils.pre_processing.pre_processing import remove_previous_results, parallelepiped
os.environ["OPENMC_CROSS_SECTIONS"] = PATH_TO_CROSS_SECTIONS

materials = openmc.Materials([FUEL_MATERIAL, CDTE_MATERIAL, AIR_MATERIAL, CONCRETE_MATERIAL, GRAPHITE_MATERIAL])

fuel = openmc.Material(name='fuel')
fuel.add_nuclide('U235', 1.0)
fuel.set_density('g/cm3', 10.0)


water = openmc.Material(name='water')
water.add_nuclide('H1', 2.0)
water.add_nuclide('O16', 1.0)
water.set_density('g/cm3', 1.0)

mats = openmc.Materials((FUEL_MATERIAL, GRAPHITE_MATERIAL))
mats.export_to_xml()

r_pin = openmc.ZCylinder(r=0.3)
fuel_cell = openmc.Cell(fill=FUEL_MATERIAL, region=-r_pin)
water_cell = openmc.Cell(fill=GRAPHITE_MATERIAL, region=+r_pin)
pin_universe = openmc.Universe(cells=(fuel_cell, water_cell))

r_big_pin = openmc.ZCylinder(r=0.5)
fuel2_cell = openmc.Cell(fill=FUEL_MATERIAL, region=-r_big_pin)
water2_cell = openmc.Cell(fill=GRAPHITE_MATERIAL, region=+r_big_pin)
big_pin_universe = openmc.Universe(cells=(fuel2_cell, water2_cell))

all_water_cell = openmc.Cell(fill=GRAPHITE_MATERIAL)
outer_universe = openmc.Universe(cells=(all_water_cell,))

lat = openmc.HexLattice()

lat.center = (0., 0.)
lat.pitch = (1.25,)
lat.outer = outer_universe

outer_ring = [big_pin_universe] + [pin_universe]*11
middle_ring = [big_pin_universe] + [pin_universe]*5
inner_ring = [big_pin_universe]
lat.universes = [outer_ring, middle_ring, inner_ring]

outer_surface = openmc.ZCylinder(r=4.0, boundary_type='vacuum')
main_cell = openmc.Cell(fill=lat, region=-outer_surface)
geom = openmc.Geometry([main_cell])
geom.export_to_xml()

plot = openmc.Plot()

WIDTH = 10  # cm
HEIGHT = 10  # cm

plot.origin = (0, 0, 0)
plot.width = (WIDTH, HEIGHT)
plot.pixels = (600, 600)

X_VALUES = np.linspace(-WIDTH//2, WIDTH//2, 600)
Y_VALUES = np.linspace(-HEIGHT//2, HEIGHT//2, 600)

X, Y = np.meshgrid(X_VALUES, Y_VALUES)

# Define the colors for each material
default_colors = ['red', 'green', 'lightblue', 'gray', 'brown', 'orange', 'purple', 'yellow', 'pink', 'cyan']
# Extend colors if needed
colors = (default_colors * ((len(materials) + len(default_colors) - 1) // len(default_colors)))[:len(materials)]
plot.colors = {mat: color for mat, color in zip(materials, colors)}

plot.color_by = 'material'
plot.basis = 'xy'
plot.filename = "plot_xy.png"
plots = openmc.Plots([plot])
plots.export_to_xml()
openmc.plot_geometry()

img = Image.open("plot_xy.png")
plt.figure(figsize=(8, 6))
plt.imshow(img)
plt.title("OpenMC Geometry Plot - XY Plane")
plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
plt.xticks(np.linspace(0, img.size[0], num=7), np.linspace(np.min(X_VALUES), np.max(X_VALUES), num=7, dtype=int))
plt.yticks(np.linspace(0, img.size[1], num=7), np.linspace(np.min(Y_VALUES), np.max(Y_VALUES), num=7, dtype=int))
plt.xlabel("X (cm)")
plt.ylabel("Y (cm)")

# Add color/material legend at the right of the figure
import matplotlib.patches as mpatches
legend_patches = [mpatches.Patch(color=color, label=mat.name if hasattr(mat, 'name') else str(mat)) for mat, color in zip(materials, colors)]
plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.tight_layout()
plt.show()