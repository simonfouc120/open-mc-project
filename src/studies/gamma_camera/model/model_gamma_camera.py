import openmc
from pathlib import Path
import sys
import os 
import numpy as np

CELL_SIZE = 0.08 # cm
LENGTH_DETECTOR = 1.48 # cm



CWD = Path.cwd().resolve()
project_root = Path.cwd().parents[3]
sys.path.append(str(project_root))
from parameters.parameters_materials import AIR_MATERIAL, CONCRETE_MATERIAL, GRAPHITE_MATERIAL, STEEL_MATERIAL, CDTE_MATERIAL, VOID_MATERIAL

material = openmc.Materials([AIR_MATERIAL, CDTE_MATERIAL, CONCRETE_MATERIAL, GRAPHITE_MATERIAL, STEEL_MATERIAL, VOID_MATERIAL])

from src.utils.pre_processing.pre_processing import plot_geometry, parallelepiped

material.export_to_xml()
MATERIAL = material

# === Paramètres géométriques ===
pixel_size = CELL_SIZE  #  cm
pixel_thickness = 2.0 * 0.1 # 2 mm → cm (en x)
frame_thickness = 1.0 * 0.1 # 1 mm → cm

n_y = 16  # Nombre de pixels selon y
n_z = 16  # Nombre de pixels selon z

# === Dimensions de la matrice ===
matrix_width_y = n_y * pixel_size
matrix_width_z = n_z * pixel_size
total_width_y = matrix_width_y + 2 * frame_thickness
total_width_z = matrix_width_z + 2 * frame_thickness

# === Plans de délimitation du cadre ===
x_half = pixel_thickness / 2
y_half = total_width_y / 2
z_half = total_width_z / 2

xmin = openmc.XPlane(x0=-x_half)
xmax = openmc.XPlane(x0=+x_half)
ymin = openmc.YPlane(y0=-y_half)  
ymax = openmc.YPlane(y0=+y_half)  
zmin = openmc.ZPlane(z0=-z_half)  
zmax = openmc.ZPlane(z0=+z_half)  

cadre_region = +xmin & -xmax & +ymin & -ymax & +zmin & -zmax
cadre_cell = openmc.Cell(region=cadre_region, fill=CDTE_MATERIAL)

# === Pixels ===
pixel_cells = []

for i in range(n_y):
    for j in range(n_z):
        idx = i * n_z + j

        y0 = -matrix_width_y / 2 + i * pixel_size
        y1 = y0 + pixel_size
        z0 = -matrix_width_z / 2 + j * pixel_size
        z1 = z0 + pixel_size

        pixel_region = +xmin & -xmax & \
                       +openmc.YPlane(y0 = y0) & -openmc.YPlane(y0 = y1) & \
                       +openmc.ZPlane(z0 = z0) & -openmc.ZPlane(z0 = z1)

        pixel_cell = openmc.Cell(region=pixel_region, fill=CDTE_MATERIAL)
        pixel_cells.append(pixel_cell)



# === Sphère d'air autour ===
bounding_sphere = openmc.Sphere(r=100.0, boundary_type='vacuum')  # rayon en cm
air_region = -bounding_sphere & ~cadre_region
air_cell = openmc.Cell(region=air_region, fill=VOID_MATERIAL)

# === Assemblage final ===
root_universe = openmc.Universe(cells=[cadre_cell, air_cell] + [pixel_cell for pixel_cell in pixel_cells])
geometry = openmc.Geometry(root_universe)
PIXEL_CELLS = pixel_cells
GEOMETRY = geometry