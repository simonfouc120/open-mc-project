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
project_root = Path(__file__).resolve().parents[4]  
sys.path.append(str(project_root))

from parameters.parameters_paths import PATH_TO_CROSS_SECTIONS, IMAGE_PATH
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

# Plot the geometry

# plot_geometry(materials = material, plane="yz", saving_figure=True, dpi=500, height=800, width=800)

# plot_geometry(materials = material, plane="xy", saving_figure=True, dpi=500, height=400, width=400)


class reactor_model:
    def __init__(self, 
                 fuel_temperature:int=900, 
                 helium_temperature:int=900, 
                 air_temperature:int=600, 
                 concrete_temperature:int=600,
                 r_pin_fuel:float=1.0,
                 pin_helium_fuel:float=1.0,
                 pitch_lattice:float=4.5,
                 radius_graphite_cylinder:float=22.0,
                 total_height_active_part:float=500.0,
                 pitch_graphite_assembly:float=55.0,
                 berryllium_thickness:float=30.0,
                 edge_length_vessel:float=194.0):

        FUEL_MATERIAL.temperature = fuel_temperature # K
        HELIUM_MATERIAL.temperature = helium_temperature # K

        AIR_MATERIAL.temperature = air_temperature # K
        CONCRETE_MATERIAL.temperature = concrete_temperature # K

        fuel_cyl = openmc.ZCylinder(r=r_pin_fuel)
        fuel_cell = openmc.Cell(fill=FUEL_MATERIAL, region=-fuel_cyl)
        graphite_fuel_cell = openmc.Cell(fill=GRAPHITE_MATERIAL, region=+fuel_cyl)
        fuel_universe = openmc.Universe(cells=(fuel_cell, graphite_fuel_cell))

        helium_cyl = openmc.ZCylinder(r=pin_helium_fuel)
        helium_main_cell = openmc.Cell(fill=HELIUM_MATERIAL, region=-helium_cyl)
        graphite_helium_cell = openmc.Cell(fill=GRAPHITE_MATERIAL, region=+helium_cyl)
        helium_universe = openmc.Universe(cells=(helium_main_cell, graphite_helium_cell))

        all_graphite_main_cell = openmc.Cell(fill=GRAPHITE_MATERIAL)
        graphite_outer_universe = openmc.Universe(cells=(all_graphite_main_cell,))

        hex_lat = openmc.HexLattice()
        hex_lat.center = (0., 0.)
        hex_lat.pitch = (pitch_lattice,)   # pitch en cm
        hex_lat.outer = graphite_outer_universe

        hex_lat.universes = [
            [helium_universe, fuel_universe] * 12, # 5th ring
            [helium_universe, fuel_universe] * 9,    # 4th ring    
            [helium_universe, fuel_universe] * 6,        # 3rd ring
            [helium_universe, fuel_universe] * 3,     # 2nd ring
            [fuel_universe]          # 1st ring
        ]   

        outer_cyl = openmc.ZCylinder(r=radius_graphite_cylinder)
        top_plane = openmc.ZPlane(z0=total_height_active_part/2)
        bottom_plane = openmc.ZPlane(z0=-total_height_active_part/2)

        # Main cell: lattice inside graphite cylinder
        main_graphite_cell = openmc.Cell(
            fill=hex_lat,
            region=(-outer_cyl & -top_plane & +bottom_plane)
        )

        hw_region = -openmc.ZCylinder(r=radius_graphite_cylinder*3) & ~main_graphite_cell.region
        hw_main_cell = openmc.Cell(fill=HEAVY_WATER_MATERIAL, region=hw_region)

        block_universe = openmc.Universe(cells=(main_graphite_cell, hw_main_cell))

        # Create a lattice of graphite universes
        graphite_lat = openmc.HexLattice()
        graphite_lat.center = (0., 0.)
        graphite_lat.pitch = (pitch_graphite_assembly,)  # pitch in cm
        graphite_lat.outer = block_universe
        graphite_lat.universes = [
            [block_universe]*6,            # center (1 universe)
            [block_universe]               # first ring (6 universes)
        ] 

        graphite_assembly_main_cell = openmc.Cell(
            name="graphite_assembly_cell",
            fill=graphite_lat,
            region=(-openmc.model.HexagonalPrism(edge_length=edge_length_vessel, orientation='y', origin=(0.0, 0.0))
        & -top_plane & +bottom_plane)
        )

        beryllium_above_cell = openmc.Cell(
            fill=BERYLLIUM_MATERIAL,
            region=(-openmc.model.HexagonalPrism(edge_length=edge_length_vessel, orientation='y', origin=(0.0, 0.0))
        & ~graphite_assembly_main_cell.region & -openmc.ZPlane(z0=total_height_active_part/2 + berryllium_thickness) 
        & +openmc.ZPlane(z0=total_height_active_part/2)) 
        )

        beryllium_below_cell = openmc.Cell(
            fill=BERYLLIUM_MATERIAL,
            region=(-openmc.model.HexagonalPrism(edge_length=edge_length_vessel, orientation='y', origin=(0.0, 0.0))
        & ~graphite_assembly_main_cell.region & +openmc.ZPlane(z0=-total_height_active_part/2 - berryllium_thickness) 
        & -openmc.ZPlane(z0=-total_height_active_part/2))
        )

        steel_liner_main_cell = openmc.Cell(
            fill=BORATED_STEEL_MATERIAL,
            region=(-openmc.model.HexagonalPrism(edge_length=200.0, orientation='y', origin=(0.0, 0.0))
        & ~graphite_assembly_main_cell.region & ~beryllium_above_cell.region &  ~beryllium_below_cell.region
        & -openmc.ZPlane(z0=286.0) & +openmc.ZPlane(z0=-286.0)) 
        )

        light_water_main_cell = openmc.Cell(
            fill=WATER_MATERIAL,
            region=(-openmc.model.RectangularParallelepiped(xmin=-315.0, xmax=315.0, ymin=-315.0, ymax=315.0, zmin=-350.0, zmax=-90.0)
            & ~graphite_assembly_main_cell.region & ~beryllium_above_cell.region 
            &  ~beryllium_below_cell.region & ~steel_liner_main_cell.region
        ))

        light_water_liner_main_cell = openmc.Cell(
            fill=STEEL_MATERIAL,
            region=(-openmc.model.RectangularParallelepiped(xmin=-320.0, xmax=320.0, ymin=-320.0, ymax=320.0, zmin=-355.0, zmax=-90.0)
            & ~graphite_assembly_main_cell.region & ~beryllium_above_cell.region &  ~beryllium_below_cell.region 
            & ~steel_liner_main_cell.region & ~light_water_main_cell.region)
        )

        # add a slab of concrete below the light water
        concrete_main_cell = openmc.Cell(
            fill=CONCRETE_MATERIAL,
            region=(-openmc.model.RectangularParallelepiped(xmin=-400.0, xmax=400.0, ymin=-400.0, ymax=400.0, zmax=-355.0, zmin=-400.0)
            & ~graphite_assembly_main_cell.region & ~beryllium_above_cell.region &  ~beryllium_below_cell.region 
            & ~steel_liner_main_cell.region & ~light_water_main_cell.region & ~light_water_liner_main_cell.region)
        )

        calc_sphere_cell = openmc.Cell(
            fill = AIR_MATERIAL,
            region=-(openmc.Sphere(x0=0.0, y0=400.0, z0=-300.0, r=10.0)) 
        )

        outer_sphere_main = openmc.Sphere(r=1000.0, boundary_type='vacuum')

        air_main_region = -outer_sphere_main & ~steel_liner_main_cell.region & ~light_water_main_cell.region & ~concrete_main_cell.region & ~calc_sphere_cell.region
        air_main_cell = openmc.Cell(fill=AIR_MATERIAL, region=air_main_region)

        geometry = openmc.Geometry([steel_liner_main_cell, graphite_assembly_main_cell, calc_sphere_cell,
                                    beryllium_above_cell, beryllium_below_cell,
                                    light_water_main_cell, light_water_liner_main_cell, concrete_main_cell, air_main_cell])

        model = openmc.Model(
                geometry=openmc.Geometry(
                    [graphite_assembly_main_cell, beryllium_above_cell, beryllium_below_cell, steel_liner_main_cell, 
                    light_water_main_cell, light_water_liner_main_cell, concrete_main_cell, calc_sphere_cell, air_main_cell]
                ),
                settings=openmc.Settings(),
                materials=material
        )
        self.model = model
        self.graphite_cell = main_graphite_cell
        self.calculation_cell = calc_sphere_cell

    def build_model(self):
        return self.model, self.graphite_cell, self.calculation_cell

    def export_to_xml(self):
        self.model.export_to_xml()


    def materials(self):
        return self.model.materials


my_reactor = reactor_model(total_height_active_part=500.0)
MODEL = my_reactor.build_model()[0]
my_reactor.export_to_xml()
plot_geometry(materials = MODEL.materials, plane="yz", saving_figure=True, dpi=500, height=800, width=800)
plot_geometry(materials = my_reactor.materials(), plane="xy", saving_figure=True, dpi=500, height=400, width=400)



# fonction material pas de fission