
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
from parameters.parameters_materials import FUEL_MATERIAL, CDTE_MATERIAL, AIR_MATERIAL, CONCRETE_MATERIAL, GRAPHITE_MATERIAL
from src.utils.pre_processing.pre_processing import remove_previous_results, parallelepiped
os.environ["OPENMC_CROSS_SECTIONS"] = PATH_TO_CROSS_SECTIONS

materials = openmc.Materials([FUEL_MATERIAL, CDTE_MATERIAL, AIR_MATERIAL, CONCRETE_MATERIAL, GRAPHITE_MATERIAL])

outer_boundary = openmc.Sphere(r=200.0, surface_id=3, boundary_type='vacuum')  # Limite du monde
graphite_surface = openmc.ZCylinder(r=45.0, surface_id=4)

graphite = openmc.Cell(name="graphite_cell")
graphite.fill = GRAPHITE_MATERIAL
graphite.region = -graphite_surface 

lat = openmc.HexLattice()
lat.center = (0., 0.)
lat.pitch = (1.25,)

geometry = openmc.Geometry(-graphite_surface )

materials.export_to_xml()
geometry.export_to_xml()


# Plot geometry in XY and display the image
plot = openmc.Plot()

WIDTH = 120  # cm
HEIGHT = 120  # cm

plot.origin = (0, 0, 0)
plot.width = (WIDTH, HEIGHT)
plot.pixels = (600, 600)

X_VALUES = np.linspace(-WIDTH//2, WIDTH//2, 600)
Y_VALUES = np.linspace(-HEIGHT//2, HEIGHT//2, 600)

X, Y = np.meshgrid(X_VALUES, Y_VALUES)

# Define the colors for each material
plot.colors = {
    AIR_MATERIAL : 'lightblue',
    FUEL_MATERIAL : 'red'  ,
    CDTE_MATERIAL : 'green',
    GRAPHITE_MATERIAL : 'brown'
}

plot.color_by = 'material'
plot.basis = 'xy'
plot.filename = "plot_xy.png"
plots = openmc.Plots([plot])
plots.export_to_xml()
openmc.plot_geometry()

img = Image.open("plot_xy.png")
plt.imshow(img)
plt.title("OpenMC Geometry Plot - XY Plane")
plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
plt.xticks(np.linspace(0, img.size[0], num=7), np.linspace(np.min(X_VALUES), np.max(X_VALUES), num=7, dtype=int))
plt.yticks(np.linspace(0, img.size[1], num=7), np.linspace(np.min(Y_VALUES), np.max(Y_VALUES), num=7, dtype=int))
plt.xlabel("X (cm)")
plt.ylabel("Y (cm)")
plt.legend()
plt.show()