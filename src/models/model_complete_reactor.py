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
project_root = Path(__file__).resolve().parents[2]  
sys.path.append(str(project_root))

from parameters.parameters_paths import PATH_TO_CROSS_SECTIONS
from parameters.parameters_materials import (FUEL_MATERIAL, HELIUM_MATERIAL, AIR_MATERIAL, 
                                             CONCRETE_MATERIAL, GRAPHITE_MATERIAL, STEEL_MATERIAL, 
                                             WATER_MATERIAL, HEAVY_WATER_MATERIAL, BERYLLIUM_MATERIAL, 
                                             BORATED_STEEL_MATERIAL)
from src.utils.pre_processing.pre_processing import (plot_geometry)
os.environ["OPENMC_CROSS_SECTIONS"] = PATH_TO_CROSS_SECTIONS

material = openmc.Materials([FUEL_MATERIAL, HELIUM_MATERIAL, AIR_MATERIAL, 
                             CONCRETE_MATERIAL, GRAPHITE_MATERIAL, STEEL_MATERIAL, 
                             WATER_MATERIAL, HEAVY_WATER_MATERIAL, BERYLLIUM_MATERIAL, 
                             BORATED_STEEL_MATERIAL])
# material.export_to_xml()


FUEL_MATERIAL.temperature = 900 # K
HELIUM_MATERIAL.temperature = 900 # K

AIR_MATERIAL.temperature = 600 # K
CONCRETE_MATERIAL.temperature = 600 # K

r_pin_fuel = openmc.ZCylinder(r=1.)
pin_fuel_fuel_cell = openmc.Cell(fill=FUEL_MATERIAL, region=-r_pin_fuel)
graphite_cell = openmc.Cell(fill=GRAPHITE_MATERIAL, region=+r_pin_fuel)
pin_fuel_universe = openmc.Universe(cells=(pin_fuel_fuel_cell, graphite_cell))

pin_helium_cell = openmc.ZCylinder(r=1.)
helium_cell = openmc.Cell(fill=HELIUM_MATERIAL, region=-pin_helium_cell)
graphite_cell = openmc.Cell(fill=GRAPHITE_MATERIAL, region=+pin_helium_cell)
pin_helium_universe = openmc.Universe(cells=(helium_cell, graphite_cell))

all_graphite_cell = openmc.Cell(fill=GRAPHITE_MATERIAL)
outer_universe = openmc.Universe(cells=(all_graphite_cell,))

lat_he_fuel = openmc.HexLattice()
lat_he_fuel.center = (0., 0.)
lat_he_fuel.pitch = (4.5,)   # pitch en cm
lat_he_fuel.outer = outer_universe

lat_he_fuel.universes = [
    [pin_helium_universe, pin_fuel_universe] * 12, # 5th ring
    [pin_helium_universe, pin_fuel_universe] * 9,    # 4th ring    
    [pin_helium_universe, pin_fuel_universe] * 6,        # 3rd ring
    [pin_helium_universe, pin_fuel_universe] * 3,     # 2nd ring
    [pin_fuel_universe]          # 1st ring
]   

outer_surface = openmc.ZCylinder(r=22.)
height_top_active_part = openmc.ZPlane(z0=250.0)
height_bottom_active_part = openmc.ZPlane(z0=-250.0)


# Main cell: lattice inside graphite cylinder
graphite_cell = openmc.Cell(
    fill=lat_he_fuel,
    region=(-outer_surface & -height_top_active_part & +height_bottom_active_part)
)

hw_region = -openmc.ZCylinder(r=45.) & ~graphite_cell.region
hw_cell = openmc.Cell(fill=HEAVY_WATER_MATERIAL, region=hw_region)

bloc_universe = openmc.Universe(cells=(graphite_cell, hw_cell))

# Create a lattice of graphite universes
graphite_lattice = openmc.HexLattice()
graphite_lattice.center = (0., 0.)
graphite_lattice.pitch = (55.0,)  # pitch in cm
graphite_lattice.outer = bloc_universe
graphite_lattice.universes = [
    [bloc_universe]*6,                # center (1 universe)
    [bloc_universe]               # first ring (6 universes)
] 

graphite_assembly_cell = openmc.Cell(
    name="graphite_assembly_cell",
    fill=graphite_lattice,
    region=(-openmc.model.HexagonalPrism(edge_length=194.0, orientation='y', origin=(0.0, 0.0))
 & -height_top_active_part & +height_bottom_active_part)
)


beryllium_above_assembly_cell = openmc.Cell(
    fill=BERYLLIUM_MATERIAL,
    region=(-openmc.model.HexagonalPrism(edge_length=194.0, orientation='y', origin=(0.0, 0.0))
 & ~graphite_assembly_cell.region & -openmc.ZPlane(z0=280.0) & +openmc.ZPlane(z0=250.0)) 
)

beryllium_below_assembly_cell = openmc.Cell(
    fill=BERYLLIUM_MATERIAL,
    region=(-openmc.model.HexagonalPrism(edge_length=194.0, orientation='y', origin=(0.0, 0.0))
 & ~graphite_assembly_cell.region & +openmc.ZPlane(z0=-280.0) & -openmc.ZPlane(z0=-250.0))
)

steel_liner_cell = openmc.Cell(
    fill=BORATED_STEEL_MATERIAL,
    region=(-openmc.model.HexagonalPrism(edge_length=200.0, orientation='y', origin=(0.0, 0.0))
 & ~graphite_assembly_cell.region & ~beryllium_above_assembly_cell.region &  ~beryllium_below_assembly_cell.region
 & -openmc.ZPlane(z0=290.0) & +openmc.ZPlane(z0=-290.0)) 
 )

light_water_cell = openmc.Cell(
    fill=WATER_MATERIAL,
    region=(-openmc.model.RectangularParallelepiped(xmin=-315.0, xmax=315.0, ymin=-315.0, ymax=315.0, zmin=-350.0, zmax=-90.0)
    & ~graphite_assembly_cell.region & ~beryllium_above_assembly_cell.region &  ~beryllium_below_assembly_cell.region & ~steel_liner_cell.region
))

light_water_liner_cell = openmc.Cell(
    fill=STEEL_MATERIAL,
    region=(-openmc.model.RectangularParallelepiped(xmin=-320.0, xmax=320.0, ymin=-320.0, ymax=320.0, zmin=-355.0, zmax=-90.0)
    & ~graphite_assembly_cell .region & ~beryllium_above_assembly_cell.region &  ~beryllium_below_assembly_cell.region & ~steel_liner_cell.region
    & ~light_water_cell.region)
)

# add a slab of concrete below the light water
concrete_cell = openmc.Cell(
    fill=CONCRETE_MATERIAL,
    region=(-openmc.model.RectangularParallelepiped(xmin=-400.0, xmax=400.0, ymin=-400.0, ymax=400.0, zmax=-355.0, zmin=-400.0)
    & ~graphite_assembly_cell.region & ~beryllium_above_assembly_cell.region &  ~beryllium_below_assembly_cell.region & ~steel_liner_cell.region
    & ~light_water_cell.region & ~light_water_liner_cell.region)
)

sphere_calculation = openmc.Cell(
    fill = AIR_MATERIAL,
    region=-(openmc.Sphere(x0=0.0, y0=400.0, z0=-300.0, r=10.0)) 
)

outer_sphere = openmc.Sphere(r=1000.0, boundary_type='vacuum')

air_region = -outer_sphere & ~steel_liner_cell.region & ~light_water_cell.region & ~concrete_cell.region & ~sphere_calculation.region
air_cell = openmc.Cell(fill=AIR_MATERIAL, region=air_region)

geometry = openmc.Geometry([steel_liner_cell, graphite_assembly_cell, sphere_calculation,
                            beryllium_above_assembly_cell, beryllium_below_assembly_cell,
                            light_water_cell, light_water_liner_cell, concrete_cell, air_cell])

model = openmc.Model(
        geometry=openmc.Geometry(
            [graphite_assembly_cell, beryllium_above_assembly_cell, beryllium_below_assembly_cell, steel_liner_cell, 
             light_water_cell, light_water_liner_cell, concrete_cell, sphere_calculation, air_cell]
        ),
        settings=openmc.Settings(),
        materials=material
)
MODEL = model
GRAPHITE_CELL = graphite_assembly_cell
CALCULATION_CELL = sphere_calculation
# Export the model to XML files
model.export_to_xml()
