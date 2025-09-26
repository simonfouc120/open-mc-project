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
from parameters.parameters_materials import *
from src.utils.pre_processing.pre_processing import *
from src.utils.post_preocessing.post_processing import *
os.environ["OPENMC_CROSS_SECTIONS"] = PATH_TO_CROSS_SECTIONS


### A INCLURE DANS CLASSE MODEL ###
material_list = openmc.Materials([FUEL_UO2_MATERIAL, HELIUM_MATERIAL, AIR_MATERIAL, 
                             CONCRETE_MATERIAL, GRAPHITE_MATERIAL, STEEL_MATERIAL, 
                             WATER_MATERIAL, HEAVY_WATER_MATERIAL, BERYLLIUM_MATERIAL, 
                             BORATED_STEEL_MATERIAL, HEAVY_CONCRETE_HEMATITE_MATERIAL])

material_dict = materials = {
    "FUEL_UO2_MATERIAL": FUEL_UO2_MATERIAL,
    "HELIUM_MATERIAL": HELIUM_MATERIAL,
    "AIR_MATERIAL": AIR_MATERIAL,
    "CONCRETE_MATERIAL": CONCRETE_MATERIAL,
    "GRAPHITE_MATERIAL": GRAPHITE_MATERIAL,
    "STEEL_MATERIAL": STEEL_MATERIAL,
    "WATER_MATERIAL": WATER_MATERIAL,
    "HEAVY_WATER_MATERIAL": HEAVY_WATER_MATERIAL,
    "BERYLLIUM_MATERIAL": BERYLLIUM_MATERIAL,
    "BORATED_STEEL_MATERIAL": BORATED_STEEL_MATERIAL,
    "HEAVY_CONCRETE_HEMATITE_MATERIAL": HEAVY_CONCRETE_HEMATITE_MATERIAL,
}

EPSILON_DENSITY = 1e-6
POWER = 50e6
class Reactor_model:
    """Class to create a reactor model with various components and materials."""
    def __init__(
        self,
        materials: dict,
        fuel_temperature: int = 900, # K
        helium_temperature: int = 900, # K
        air_temperature: int = 600, # K
        concrete_temperature: int = 600, # K
        graphite_temperature: int = 600, # K
        r_pin_fuel: float = 1.0, # cm
        pin_helium_fuel: float = 1.0, # cm
        pitch_lattice: float = 4.5, # cm
        radius_graphite_cylinder: float = 22.0, # cm
        total_height_active_part: float = 500.0, # cm
        pitch_graphite_assembly: float = 55.0, # cm
        berryllium_thickness: float = 30.0, # cm
        edge_length_vessel: float = 194.0, # cm
        thickness_steel_liner: float = 6.0, # cm
        calculation_sphere_coordinates: tuple = (0.0, 350.0, -300.0), # cm
        calculation_sphere_radius: float = 10.0, # cm
        light_water_pool: bool = True,
        width_cavity: float = 400.0, # cm
        slab_thickness: float = 40.0, # cm
        cavity: bool = True, 
        concrete_wall_thickness: float = 30.0, # cm
        universe_boundary: str = "vacuum",
    ):
        """Initialize the reactor model with specified parameters."""

        self.material = materials.copy()
        self.other_cells = []

        # Set material temperatures
        temp_map = {
            "FUEL_UO2_MATERIAL": fuel_temperature,
            "HELIUM_MATERIAL": helium_temperature,
            "AIR_MATERIAL": air_temperature,
            "CONCRETE_MATERIAL": concrete_temperature,
            "GRAPHITE_MATERIAL": graphite_temperature,
        }
        for key, temp in temp_map.items():
            if key in self.material:
                self.material[key].temperature = temp

        # Fuel pin
        fuel_cyl = openmc.ZCylinder(r=r_pin_fuel)
        fuel_cell = openmc.Cell(fill=self.material["FUEL_UO2_MATERIAL"], region=-fuel_cyl)
        graphite_fuel_cell = openmc.Cell(fill=self.material["GRAPHITE_MATERIAL"], region=+fuel_cyl)
        fuel_universe = openmc.Universe(cells=[fuel_cell, graphite_fuel_cell])

        # Helium pin
        helium_cyl = openmc.ZCylinder(r=pin_helium_fuel)
        helium_main_cell = openmc.Cell(fill=self.material["HELIUM_MATERIAL"], region=-helium_cyl)
        graphite_helium_cell = openmc.Cell(fill=self.material["GRAPHITE_MATERIAL"], region=+helium_cyl)
        helium_universe = openmc.Universe(cells=[helium_main_cell, graphite_helium_cell])

        # Outer graphite
        all_graphite_main_cell = openmc.Cell(fill=self.material["GRAPHITE_MATERIAL"])
        graphite_outer_universe = openmc.Universe(cells=[all_graphite_main_cell])

        # Fuel lattice
        hex_lat = openmc.HexLattice()
        hex_lat.center = (0.0, 0.0)
        hex_lat.pitch = (pitch_lattice,)
        hex_lat.outer = graphite_outer_universe
        # Example: 5 rings, alternating helium/fuel, innermost is fuel
        hex_lat.universes = [
            [helium_universe, fuel_universe] * 12,  # 5th ring
            [helium_universe, fuel_universe] * 9,   # 4th ring
            [helium_universe, fuel_universe] * 6,   # 3rd ring
            [helium_universe, fuel_universe] * 3,   # 2nd ring
            [fuel_universe],                        # 1st ring
        ]

        outer_cyl = openmc.ZCylinder(r=radius_graphite_cylinder)
        top_plane = openmc.ZPlane(z0=total_height_active_part / 2)
        bottom_plane = openmc.ZPlane(z0=-total_height_active_part / 2)

        # Main cell: lattice inside graphite cylinder
        self.main_graphite_cell = openmc.Cell(
            fill=hex_lat,
            region=(-outer_cyl & -top_plane & +bottom_plane),
        )

        # Heavy water region
        hw_region = -openmc.ZCylinder(r=radius_graphite_cylinder * 3) & ~self.main_graphite_cell.region
        hw_main_cell = openmc.Cell(fill=self.material["HEAVY_WATER_MATERIAL"], region=hw_region)
        block_universe = openmc.Universe(cells=[self.main_graphite_cell, hw_main_cell])

        # Graphite assembly lattice
        graphite_lat = openmc.HexLattice()
        graphite_lat.center = (0.0, 0.0)
        graphite_lat.pitch = (pitch_graphite_assembly,)
        graphite_lat.outer = block_universe
        graphite_lat.universes = [
            [block_universe] * 18,  # 3rd ring
            [block_universe] * 12,  # 2nd ring
            [block_universe] * 6,   # 1st ring
            [block_universe],       # Center
        ]

        hex_prism = openmc.model.HexagonalPrism(
            edge_length=edge_length_vessel, orientation="y", origin=(0.0, 0.0)
        )

        self.graphite_assembly_main_cell = openmc.Cell(
            name="graphite_assembly_cell",
            fill=graphite_lat,
            region=(-hex_prism & -top_plane & +bottom_plane),
        )

        # Beryllium above
        hex_prism_above = openmc.model.HexagonalPrism(
            edge_length=edge_length_vessel, orientation="y", origin=(0.0, 0.0)
        )
        self.beryllium_above_cell = openmc.Cell(
            fill=self.material["BERYLLIUM_MATERIAL"],
            region=(
                -hex_prism_above
                & ~self.graphite_assembly_main_cell.region
                & -openmc.ZPlane(z0=total_height_active_part / 2 + berryllium_thickness)
                & +openmc.ZPlane(z0=total_height_active_part / 2)
            ),
        )

        # Beryllium below
        hex_prism_below = openmc.model.HexagonalPrism(
            edge_length=edge_length_vessel, orientation="y", origin=(0.0, 0.0)
        )
        self.beryllium_below_cell = openmc.Cell(
            fill=self.material["BERYLLIUM_MATERIAL"],
            region=(
                -hex_prism_below
                & ~self.graphite_assembly_main_cell.region
                & +openmc.ZPlane(z0=-total_height_active_part / 2 - berryllium_thickness)
                & -openmc.ZPlane(z0=-total_height_active_part / 2)
            ),
        )

        # Steel liner
        hex_prism_liner = openmc.model.HexagonalPrism(
            edge_length=edge_length_vessel + thickness_steel_liner, orientation="y", origin=(0.0, 0.0)
        )
        self.steel_liner_main_cell = openmc.Cell(
            fill=self.material["BORATED_STEEL_MATERIAL"],
            region=(
                -hex_prism_liner
                & ~self.graphite_assembly_main_cell.region
                & ~self.beryllium_above_cell.region
                & ~self.beryllium_below_cell.region
                & -openmc.ZPlane(z0=total_height_active_part / 2 + berryllium_thickness + thickness_steel_liner)
                & +openmc.ZPlane(z0=-total_height_active_part / 2 - berryllium_thickness - thickness_steel_liner)
            ),
            cell_id=999,
        )

        self.reactor_cells = [
            self.graphite_assembly_main_cell,
            self.beryllium_above_cell,
            self.beryllium_below_cell,
            self.steel_liner_main_cell,
        ]

        if light_water_pool:
            self.light_water_main_cell = openmc.Cell(
                fill=self.material["WATER_MATERIAL"],
                region=(
                    -openmc.model.RectangularParallelepiped(
                        xmin=-315.0, xmax=315.0, ymin=-315.0, ymax=315.0, zmin=-350.0, zmax=-90.0
                    )
                    & ~self.graphite_assembly_main_cell.region
                    & ~self.beryllium_above_cell.region
                    & ~self.beryllium_below_cell.region
                    & ~self.steel_liner_main_cell.region
                ),
            )
            self.other_cells.append(self.light_water_main_cell)

            # Steel liner for light water
            self.light_water_liner_main_cell = openmc.Cell(
                fill=self.material["STEEL_MATERIAL"],
                region=(
                    -openmc.model.RectangularParallelepiped(
                        xmin=-320.0, xmax=320.0, ymin=-320.0, ymax=320.0, zmin=-355.0, zmax=-90.0
                    )
                    & ~self.graphite_assembly_main_cell.region
                    & ~self.beryllium_above_cell.region
                    & ~self.beryllium_below_cell.region
                    & ~self.steel_liner_main_cell.region
                    & ~self.light_water_main_cell.region
                ),
            )
            self.other_cells.append(self.light_water_liner_main_cell)
        else:
            pass


        if cavity:
            # Concrete slab below light water
            self.concrete_slab_cell = openmc.Cell(
                fill=self.material["CONCRETE_MATERIAL"],
                region=(
                    -openmc.model.RectangularParallelepiped(
                        xmin=-width_cavity, xmax=width_cavity, ymin=-width_cavity, ymax=width_cavity, zmax=-400.0, zmin=-400.0 - slab_thickness
                    )
                ),
            )
            self.other_cells.append(self.concrete_slab_cell)

            self.concrete_walls_cells = []
            # Concrete walls around cavity
            for x_sign in [-1, 1]:
                if x_sign == -1:
                    xmin = -width_cavity - concrete_wall_thickness
                    xmax = -width_cavity
                else:
                    xmin = width_cavity
                    xmax = width_cavity + concrete_wall_thickness
                wall_cell = openmc.Cell(
                    fill=self.material["HEAVY_CONCRETE_HEMATITE_MATERIAL"],
                    region=(
                        -openmc.model.RectangularParallelepiped(
                            xmin=xmin, xmax=xmax, ymin=-width_cavity, ymax=width_cavity + concrete_wall_thickness, zmin=-400.0 - slab_thickness, zmax=600.0
                        )
                        & ~self.graphite_assembly_main_cell.region
                        & ~self.beryllium_above_cell.region
                        & ~self.beryllium_below_cell.region
                        & ~self.steel_liner_main_cell.region
                    ),
                )
                self.concrete_walls_cells.append(wall_cell)
                self.other_cells.append(wall_cell)
            for y_sign in [-1, 1]:
                if y_sign == -1:
                    ymin = -width_cavity - concrete_wall_thickness
                    ymax = -width_cavity
                else:
                    ymin = width_cavity
                    ymax = width_cavity + concrete_wall_thickness
                wall_cell = openmc.Cell(
                    fill=self.material["HEAVY_CONCRETE_HEMATITE_MATERIAL"],
                    region=(
                        -openmc.model.RectangularParallelepiped(
                            xmin=-width_cavity - concrete_wall_thickness, xmax=width_cavity + concrete_wall_thickness, ymin=ymin, ymax=ymax, zmin=-400.0 - slab_thickness, zmax=600.0
                        )
                        & ~self.graphite_assembly_main_cell.region
                        & ~self.beryllium_above_cell.region
                        & ~self.beryllium_below_cell.region
                        & ~self.steel_liner_main_cell.region
                    ),
                )
                self.concrete_walls_cells.append(wall_cell)
                self.other_cells.append(wall_cell)
        else:
            pass

        # Calculation sphere
        self.calc_sphere_cell = openmc.Cell(
            fill=self.material["AIR_MATERIAL"],
            region=-openmc.Sphere(
                x0=calculation_sphere_coordinates[0],
                y0=calculation_sphere_coordinates[1],
                z0=calculation_sphere_coordinates[2],
                r=calculation_sphere_radius,
            ),
        )
        self.other_cells.append(self.calc_sphere_cell)

        # Outer sphere (vacuum boundary)
        self.outer_sphere_main = openmc.Sphere(r=1500.0, boundary_type=universe_boundary)

        # Air region outside everything else
        self.air_main_region = -self.outer_sphere_main & ~self.steel_liner_main_cell.region
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

    def export_to_xml(self):
        self.model.export_to_xml()

    def rebuild_universe(self):
        self.air_main_region = -self.outer_sphere_main 
        if self.other_cells:
            for cell in self.other_cells:
                self.air_main_region &= ~cell.region
        self.air_main_cell = openmc.Cell(fill=self.material["AIR_MATERIAL"], region=self.air_main_region)
        # self.build_model()

    def add_cell(self, surface:openmc.Surface, 
                  material_name:str="CONCRETE_MATERIAL",
                  cells_to_exclude:list=[], 
                  cells_to_be_excluded_by:list=[]) -> openmc.Cell:
        """
        Adds a new cell to the reactor model.
        Parameters:
        - surface: An OpenMC surface defining the boundary of the new cell.
        - cells_to_exclude: List of existing cells to exclude from the new cell's region.
        - cells_to_be_excluded_by: List of existing cells that should exclude the new cell's region.
        """
        cell_region = -surface
        if cells_to_exclude:
            for cell in cells_to_exclude:
                cell.region &= ~cell.region
        self.new_cell = openmc.Cell(fill=self.material[material_name], region=cell_region)

        cells = []
        if cells_to_be_excluded_by:
            for cell in cells_to_be_excluded_by:
                cell.region &= ~self.new_cell.region
                cells.append(openmc.Cell(fill=cell.fill, region=cell.region))
        for cell in cells_to_be_excluded_by:
            if cell in self.other_cells:
                self.other_cells.remove(cell)
        for cell in cells:
            self.other_cells.append(cell)
        self.other_cells.append(self.new_cell)
        self.rebuild_universe()
        return self.new_cell

    # from add cell create a method add wall concrete
    def add_wall_concrete(self, concrete_wall_coordinates=(0, 0, 0),
                                        dx: float = 50.0, dy: float = 50.0, dz: float = 1000.0,
                                        cells_to_exclude: list = [],
                                        cells_to_be_excluded_by: list = []):
        """
        Adds a concrete wall cell using the add_cell method.
        Parameters:
        - concrete_wall_coordinates: Center coordinates (x0, y0, z0) of the wall.
        - dx, dy, dz: Dimensions of the wall.
        - cells_to_exclude: List of cells to exclude from the wall region.
        - cells_to_be_excluded_by: List of cells that should exclude the wall region.
        """
        x0, y0, z0 = concrete_wall_coordinates
        wall_surface = openmc.model.RectangularParallelepiped(
            xmin=x0 - dx / 2, xmax=x0 + dx / 2,
            ymin=y0 - dy / 2, ymax=y0 + dy / 2,
            zmin=z0 - dz / 2, zmax=z0 + dz / 2
        )
        self.add_cell(
            surface=wall_surface,
            material_name="CONCRETE_MATERIAL",
            cells_to_exclude=cells_to_exclude,
            cells_to_be_excluded_by=cells_to_be_excluded_by
        )

    def add_calculation_sphere(self, coordinates: tuple = (0.0, 350.0, -300.0), radius: float = 10.0,
                               cells_to_exclude: list = [],
                               cells_to_be_excluded_by: list = []) -> openmc.Cell:
        """
        Adds a calculation sphere cell using the add_cell method.
        Parameters:
        - coordinates: Center coordinates (x0, y0, z0) of the sphere.
        - radius: Radius of the sphere.
        - cells_to_exclude: List of cells to exclude from the sphere region.
        - cells_to_be_excluded_by: List of cells that should exclude the sphere region.
        """
        x0, y0, z0 = coordinates
        sphere_surface = openmc.Sphere(x0=x0, y0=y0, z0=z0, r=radius)
        sphere_cell = self.add_cell(
            surface=sphere_surface,
            material_name="AIR_MATERIAL",
            cells_to_exclude=cells_to_exclude,
            cells_to_be_excluded_by=cells_to_be_excluded_by
        )
        return sphere_cell
    

def set_low_density_for_materials(material_dict, low_density_material_types:list=None, density_value:float=EPSILON_DENSITY):
    """
    Sets a very low density for specified material types in the material dictionary.
    
    Parameters:
    -----------
    material_dict : dict
        Dictionary containing all materials
    low_density_material_types : list, optional
        List of materials to set to low density, defaults to [WATER_MATERIAL, HEAVY_WATER_MATERIAL, CONCRETE_MATERIAL]
    density_value : float, optional
        Low density value to set (in g/cm3), defaults to EPSILON_DENSITY
    """
    if low_density_material_types is None:
        low_density_material_types = [WATER_MATERIAL, HEAVY_WATER_MATERIAL, CONCRETE_MATERIAL]
        
    for material in material_dict.values():
        if material in low_density_material_types:
            material.set_density('g/cm3', density_value)
    return material_dict