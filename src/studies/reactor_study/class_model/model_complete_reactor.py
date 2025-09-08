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


### A INCLURE DANS CLASSE MODEL ###
material_list = openmc.Materials([FUEL_MATERIAL, HELIUM_MATERIAL, AIR_MATERIAL, 
                             CONCRETE_MATERIAL, GRAPHITE_MATERIAL, STEEL_MATERIAL, 
                             WATER_MATERIAL, HEAVY_WATER_MATERIAL, BERYLLIUM_MATERIAL, 
                             BORATED_STEEL_MATERIAL])

material_dict = materials = {
    "FUEL_MATERIAL": FUEL_MATERIAL,
    "HELIUM_MATERIAL": HELIUM_MATERIAL,
    "AIR_MATERIAL": AIR_MATERIAL,
    "CONCRETE_MATERIAL": CONCRETE_MATERIAL,
    "GRAPHITE_MATERIAL": GRAPHITE_MATERIAL,
    "STEEL_MATERIAL": STEEL_MATERIAL,
    "WATER_MATERIAL": WATER_MATERIAL,
    "HEAVY_WATER_MATERIAL": HEAVY_WATER_MATERIAL,
    "BERYLLIUM_MATERIAL": BERYLLIUM_MATERIAL,
    "BORATED_STEEL_MATERIAL": BORATED_STEEL_MATERIAL
}

EPSILON_DENSITY = 1e-6

class reactor_model:
    """Class to create a reactor model with various components and materials."""
    def __init__(self, 
                 materials:dict,
                 fuel_temperature:int=900, 
                 helium_temperature:int=900, 
                 air_temperature:int=600, 
                 concrete_temperature:int=600,
                 graphite_temperature:int=600,
                 r_pin_fuel:float=1.0,
                 pin_helium_fuel:float=1.0,
                 pitch_lattice:float=4.5,
                 radius_graphite_cylinder:float=22.0,
                 total_height_active_part:float=500.0,
                 pitch_graphite_assembly:float=55.0,
                 berryllium_thickness:float=30.0,
                 edge_length_vessel:float=194.0,
                 thickness_steel_liner:float=6.0, 
                 calculation_sphere_coordinates:tuple=(0.0, 400.0, -300.0),
                 calculation_sphere_radius:float=10.0):
        
        """Initialize the reactor model with specified parameters.

        Parameters:
        - fuel_temperature (int): Temperature of the fuel in Kelvin.
        - helium_temperature (int): Temperature of the helium in Kelvin.
        - air_temperature (int): Temperature of the air in Kelvin.
        - concrete_temperature (int): Temperature of the concrete in Kelvin.
        - graphite_temperature (int): Temperature of the graphite in Kelvin.
        - r_pin_fuel (float): Radius of the fuel pin in cm.
        - pin_helium_fuel (float): Radius of the helium pin in cm.
        - pitch_lattice (float): Pitch of the lattice in cm.
        - radius_graphite_cylinder (float): Radius of the graphite cylinder in cm.
        - total_height_active_part (float): Total height of the active part in cm.
        - pitch_graphite_assembly (float): Pitch of the graphite assembly in cm.
        - berryllium_thickness (float): Thickness of the beryllium layer in cm.
        - edge_length_vessel (float): Edge length of the vessel in cm.
        - thickness_steel_liner (float): Thickness of the steel liner in cm.
        - calculation_sphere_coordinates (tuple): Coordinates of the calculation sphere center (x, y, z).
        - calculation_sphere_radius (float): Radius of the calculation sphere in cm.
        """

        self.material = materials


        self.material = materials

        self.material["FUEL_MATERIAL"].temperature = fuel_temperature # K
        self.material["HELIUM_MATERIAL"].temperature = helium_temperature # K
        self.material["AIR_MATERIAL"].temperature = air_temperature # K
        self.material["CONCRETE_MATERIAL"].temperature = concrete_temperature # K
        self.material["GRAPHITE_MATERIAL"].temperature = graphite_temperature # K


        fuel_cyl = openmc.ZCylinder(r=r_pin_fuel)
        fuel_cell = openmc.Cell(fill=self.material["FUEL_MATERIAL"], region=-fuel_cyl)
        graphite_fuel_cell = openmc.Cell(fill=self.material["GRAPHITE_MATERIAL"], region=+fuel_cyl)
        fuel_universe = openmc.Universe(cells=(fuel_cell, graphite_fuel_cell))

        helium_cyl = openmc.ZCylinder(r=pin_helium_fuel)
        helium_main_cell = openmc.Cell(fill=self.material["HELIUM_MATERIAL"], region=-helium_cyl)
        graphite_helium_cell = openmc.Cell(fill=self.material["GRAPHITE_MATERIAL"], region=+helium_cyl)
        helium_universe = openmc.Universe(cells=(helium_main_cell, graphite_helium_cell))

        all_graphite_main_cell = openmc.Cell(fill=self.material["GRAPHITE_MATERIAL"])
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
        self.main_graphite_cell = openmc.Cell(
            fill=hex_lat,
            region=(-outer_cyl & -top_plane & +bottom_plane)
        )

        hw_region = -openmc.ZCylinder(r=radius_graphite_cylinder*3) & ~self.main_graphite_cell.region
        hw_main_cell = openmc.Cell(fill=self.material["HEAVY_WATER_MATERIAL"], region=hw_region)

        block_universe = openmc.Universe(cells=(self.main_graphite_cell, hw_main_cell))

        # Create a lattice of graphite universes
        graphite_lat = openmc.HexLattice()
        graphite_lat.center = (0., 0.)
        graphite_lat.pitch = (pitch_graphite_assembly,)  # pitch in cm
        graphite_lat.outer = block_universe
        graphite_lat.universes = [
            [block_universe]*6,            # center (1 universe)
            [block_universe]               # first ring (6 universes)
        ] 

        self.graphite_assembly_main_cell = openmc.Cell(
            name="graphite_assembly_cell",
            fill=graphite_lat,
            region=(-openmc.model.HexagonalPrism(edge_length=edge_length_vessel, orientation='y', origin=(0.0, 0.0))
        & -top_plane & +bottom_plane)
        )

        self.beryllium_above_cell = openmc.Cell(
            fill=self.material["BERYLLIUM_MATERIAL"],
            region=(-openmc.model.HexagonalPrism(edge_length=edge_length_vessel, orientation='y', origin=(0.0, 0.0))
        & ~self.graphite_assembly_main_cell.region & -openmc.ZPlane(z0=total_height_active_part/2 + berryllium_thickness) 
        & +openmc.ZPlane(z0=total_height_active_part/2)) 
        )

        self.beryllium_below_cell = openmc.Cell(
            fill=self.material["BERYLLIUM_MATERIAL"],
            region=(-openmc.model.HexagonalPrism(edge_length=edge_length_vessel, orientation='y', origin=(0.0, 0.0))
        & ~self.graphite_assembly_main_cell.region & +openmc.ZPlane(z0=-total_height_active_part/2 - berryllium_thickness) 
        & -openmc.ZPlane(z0=-total_height_active_part/2))
        )

        self.steel_liner_main_cell = openmc.Cell(
            fill=self.material["BORATED_STEEL_MATERIAL"],
            region=(-openmc.model.HexagonalPrism(edge_length=edge_length_vessel+thickness_steel_liner, orientation='y', origin=(0.0, 0.0))
        & ~self.graphite_assembly_main_cell.region & ~self.beryllium_above_cell.region &  ~self.beryllium_below_cell.region
        & -openmc.ZPlane(z0=total_height_active_part/2 + berryllium_thickness + thickness_steel_liner) 
        & +openmc.ZPlane(z0=-total_height_active_part/2 - berryllium_thickness - thickness_steel_liner)) 
        )

        self.reactor_cells = [self.graphite_assembly_main_cell, self.beryllium_above_cell, 
                               self.beryllium_below_cell, self.steel_liner_main_cell]
        
        self.other_cells = []

        self.light_water_main_cell = openmc.Cell(
            fill=self.material["WATER_MATERIAL"],
            region=(-openmc.model.RectangularParallelepiped(xmin=-315.0, xmax=315.0, ymin=-315.0, ymax=315.0, zmin=-350.0, zmax=-90.0)
            & ~self.graphite_assembly_main_cell.region & ~self.beryllium_above_cell.region 
            & ~self.beryllium_below_cell.region & ~self.steel_liner_main_cell.region
        ))

        self.other_cells.append(self.light_water_main_cell)

        self.light_water_liner_main_cell = openmc.Cell(
            fill=self.material["STEEL_MATERIAL"],
            region=(-openmc.model.RectangularParallelepiped(xmin=-320.0, xmax=320.0, ymin=-320.0, ymax=320.0, zmin=-355.0, zmax=-90.0)
            & ~self.graphite_assembly_main_cell.region & ~self.beryllium_above_cell.region &  ~self.beryllium_below_cell.region 
            & ~self.steel_liner_main_cell.region & ~self.light_water_main_cell.region)
        )

        self.other_cells.append(self.light_water_liner_main_cell)

        # add a slab of concrete below the light water
        self.concrete_slab_cell = openmc.Cell(
            fill=self.material["CONCRETE_MATERIAL"],
            region=(-openmc.model.RectangularParallelepiped(xmin=-400.0, xmax=400.0, ymin=-400.0, ymax=400.0, zmax=-355.0, zmin=-400.0)
            & ~self.graphite_assembly_main_cell.region & ~self.beryllium_above_cell.region &  ~self.beryllium_below_cell.region 
            & ~self.steel_liner_main_cell.region & ~self.light_water_main_cell.region & ~self.light_water_liner_main_cell.region)
        )

        self.other_cells.append(self.concrete_slab_cell)

        self.calc_sphere_cell = openmc.Cell(
            fill=self.material["AIR_MATERIAL"],
            region=-(openmc.Sphere(x0=calculation_sphere_coordinates[0], y0=calculation_sphere_coordinates[1], z0=calculation_sphere_coordinates[2], r=calculation_sphere_radius)) 
        )

        self.other_cells.append(self.calc_sphere_cell)

        self.outer_sphere_main = openmc.Sphere(r=1000.0, boundary_type='vacuum')

        self.air_main_region = -self.outer_sphere_main & ~self.steel_liner_main_cell.region
        if self.other_cells:
            for cell in self.other_cells:
                self.air_main_region &= ~cell.region

        self.air_main_cell = openmc.Cell(fill=self.material["AIR_MATERIAL"], region=self.air_main_region)


    @property
    def geometry(self):
        return openmc.Geometry(self.reactor_cells + [self.air_main_cell] + self.other_cells)

    @property
    def model(self):
        return openmc.Model(
            geometry=self.geometry,
            settings=openmc.Settings(),
            materials=openmc.Materials(list(self.material.values()))
        )

    # def build_geometry(self):
    #     self.geometry = openmc.Geometry(self.reactor_cells + [self.air_main_cell] + self.other_cells)

    # def build_model(self):
    #     self.build_geometry()
    #     self.model = openmc.Model(
    #             geometry=self.geometry,
    #             settings=openmc.Settings(),
    #             materials=openmc.Materials(list(self.material.values()))
    #     )

    def rebuild_universe(self):
        self.air_main_region = -self.outer_sphere_main 
        if self.other_cells:
            for cell in self.other_cells:
                self.air_main_region &= ~cell.region
        self.air_main_cell = openmc.Cell(fill=self.material["AIR_MATERIAL"], region=self.air_main_region)
        # self.build_model()


    def add_wall_concrete(self, concrete_wall_coordinates = (0, 0, 0), 
                          dx:float=50.0, dy:float=50.0, dz:float=1000.0, 
                          cells_to_exclude:list=[], 
                          cells_to_be_excluded_by:list=[]):
        x0, y0, z0 = concrete_wall_coordinates
        concrete_wall_region = -openmc.model.RectangularParallelepiped(
            xmin=x0 - dx/2, xmax=x0 + dx/2, 
            ymin=y0 - dy/2, ymax=y0 + dy/2, 
            zmin=z0 - dz/2, zmax=z0 + dz/2
        )
        if cells_to_exclude:
            for cell in cells_to_exclude:
                concrete_wall_region &= ~cell.region
        concrete_wall_cell = openmc.Cell(fill=self.material["CONCRETE_MATERIAL"], region=concrete_wall_region)

        cells = []
        if cells_to_be_excluded_by:
            for cell in cells_to_be_excluded_by:
                cell.region &= ~concrete_wall_cell.region
                cells.append(openmc.Cell(fill=cell.fill, region=cell.region))
        print(len(self.other_cells))
        for cell in cells_to_be_excluded_by:
            if cell in self.other_cells:
                self.other_cells.remove(cell)

        print(len(self.other_cells))
        for cell in cells:
            self.other_cells.append(cell)
        print(len(self.other_cells))

        self.other_cells.append(concrete_wall_cell)
        print(len(self.other_cells))

        self.rebuild_universe()

    def add_cell(self, cell:openmc.Cell, excluding_cells:list=[]):
        if excluding_cells:
            for ex_cell in excluding_cells:
                cell.region &= ~ex_cell.region
        self.other_cells.append(cell)
        self.rebuild_universe()

    def export_to_xml(self):
        self.model.export_to_xml()


for material in material_dict.values():
    if material in [WATER_MATERIAL, HEAVY_WATER_MATERIAL, CONCRETE_MATERIAL]:
        material.set_density('g/cm3', EPSILON_DENSITY)


my_reactor = reactor_model(materials=material_dict,total_height_active_part=500.0)
MODEL = my_reactor.model
my_reactor.add_wall_concrete(concrete_wall_coordinates=(0,250,-200), dy=50.0, dz=150.0, 
                             cells_to_be_excluded_by=[my_reactor.light_water_main_cell, my_reactor.light_water_liner_main_cell],)
# my_reactor.add_wall_concrete(concrete_wall_coordinates=(0,250,0), dy=50.0, dz=1000.0, cells_to_exclude=[my_reactor.light_water_main_cell])
my_reactor.export_to_xml()

plot_geometry(materials = openmc.Materials(list(my_reactor.material.values())), plane="yz", saving_figure=True, dpi=500, height=1000, width=1000)
plot_geometry(materials = openmc.Materials(list(my_reactor.material.values())), plane="xy", saving_figure=True, dpi=500, height=400, width=400)

plot_geometry(materials = openmc.Materials(list(my_reactor.material.values())), plane="xy", origin=(0,0,-200), saving_figure=True, dpi=500, height=700, width=700)

# fonction material pas de fission

# run the simulation

settings = openmc.Settings()
batches_number= 100
settings.batches = batches_number
settings.inactive = 20
settings.particles = 500
settings.source = openmc.IndependentSource()
settings.source.space = openmc.stats.Point((0, 0, 0))
settings.source.particle = 'neutron'
settings.photon_transport = True
settings.source.angle = openmc.stats.Isotropic()  
settings.export_to_xml()
MODEL.settings = settings
settings.export_to_xml()
MODEL.export_to_xml()

openmc.run()
