import os
import openmc
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def remove_previous_results(batches_number):
    summary_path = "summary.h5"
    statepoint_path = f"statepoint.{batches_number}.h5"
    if os.path.exists(summary_path):
        os.remove(summary_path)
    if os.path.exists(statepoint_path):
        os.remove(statepoint_path)
        

def parallelepiped(xmin, xmax, ymin, ymax, zmin, zmax, surface_id_start=10):
    """
    Creates a rectangular parallelepiped region using OpenMC planes.

    Parameters:
        xmin (float): Minimum x-coordinate.
        xmax (float): Maximum x-coordinate.
        ymin (float): Minimum y-coordinate.
        ymax (float): Maximum y-coordinate.
        zmin (float): Minimum z-coordinate.
        zmax (float): Maximum z-coordinate.
        surface_id_start (int, optional): Starting surface ID for the planes (default is 10).

    Returns:
        openmc.Region: The OpenMC region representing the parallelepiped.
    """
    wall_xmin = openmc.XPlane(x0=xmin, surface_id=surface_id_start)
    wall_xmax = openmc.XPlane(x0=xmax, surface_id=surface_id_start+1)
    wall_ymin = openmc.YPlane(y0=ymin, surface_id=surface_id_start+2)
    wall_ymax = openmc.YPlane(y0=ymax, surface_id=surface_id_start+3)
    wall_zmin = openmc.ZPlane(z0=zmin, surface_id=surface_id_start+4)
    wall_zmax = openmc.ZPlane(z0=zmax, surface_id=surface_id_start+5)
    region = +wall_xmin & -wall_xmax & +wall_ymin & -wall_ymax & +wall_zmin & -wall_zmax
    return region


def plot_geometry(
    materials: list,
    plane: str = "xy",
    width: float = 10.0,
    height: float = 10.0,
    origin: tuple = (0, 0, 0),
    pixels: tuple = (600, 600),
    dpi: int = 300,
    color_by: str = 'material',
    saving_figure: bool = True
):
    """
    Plots the OpenMC geometry in a specified plane.
    """
    # Set up plot object
    plot = openmc.Plot()
    plot.origin = origin
    plot.width = (width, height)
    plot.pixels = pixels
    plot.color_by = color_by
    plot.basis = plane
    plot.filename = f"plot_openmc_{plane}.png"

    # Assign colors to materials
    default_colors = [
        'red', 'green', 'lightblue', 'gray', 'brown',
        'orange', 'purple', 'yellow', 'pink', 'cyan'
    ]
    colors = (default_colors * ((len(materials) + len(default_colors) - 1) // len(default_colors)))[:len(materials)]
    plot.colors = {mat: color for mat, color in zip(materials, colors)}

    # Plane basis and axis values
    if plane == "xy":
        x_vals = np.linspace(origin[0] - width / 2, origin[0] + width / 2, pixels[0])
        y_vals = np.linspace(origin[1] - height / 2, origin[1] + height / 2, pixels[1])
        xlabel, ylabel = "X (cm)", "Y (cm)"
    elif plane == "xz":
        x_vals = np.linspace(origin[0] - width / 2, origin[0] + width / 2, pixels[0])
        y_vals = np.linspace(origin[2] - height / 2, origin[2] + height / 2, pixels[1])
        xlabel, ylabel = "X (cm)", "Z (cm)"
    elif plane == "yz":
        x_vals = np.linspace(origin[1] - width / 2, origin[1] + width / 2, pixels[0])
        y_vals = np.linspace(origin[2] - height / 2, origin[2] + height / 2, pixels[1])
        xlabel, ylabel = "Y (cm)", "Z (cm)"
    else:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")

    # Export and plot geometry
    openmc.Plots([plot]).export_to_xml()
    openmc.plot_geometry()

    # Load and display image
    img = Image.open(plot.filename)
    plt.imshow(img)
    plt.title(f"OpenMC Geometry Plot - {plane.upper()} Plane")
    plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.xticks(
        np.linspace(0, img.size[0], num=7),
        [f"{x:.2f}" for x in np.linspace(np.min(x_vals), np.max(x_vals), num=7)]
    )
    plt.yticks(
        np.linspace(0, img.size[1], num=7),
        [f"{y:.2f}" for y in np.linspace(np.max(y_vals), np.min(y_vals), num=7)]
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if saving_figure:
        plt.savefig(f"plot_{plane}.png", dpi=dpi, bbox_inches='tight')
    os.remove(plot.filename)
    plt.tight_layout()
    plt.show()

def mesh_tally_xy(
    name_mesh_tally="flux_mesh",
    particule_type='neutrons',
    bin_number=400,
    lower_left=(-50.0, -50.0),
    upper_right=(50.0, 50.0),
    z_value:float = 0.0,
    z_thickness:float = 1.0
    )-> object:
    """
    Create a mesh tally in the XY plane at a specified Z value.

    Parameters:
        name_mesh_tally (str): Name of the tally.
        particule_type (str): Particle type.
        bin_number (int): Number of bins in each direction.
        lower_left (tuple): Lower left corner of the mesh (x, y).
        upper_right (tuple): Upper right corner of the mesh (x, y).
        z_value (float): Z coordinate of the mesh center.
        z_thickness (float): Thickness of the mesh in the Z direction.

    Returns:
        openmc.Tally: The mesh tally object.
    """
    mesh = openmc.RegularMesh()
    mesh.dimension = [bin_number, bin_number, 1]
    mesh.lower_left = (lower_left[0], lower_left[1], z_value - z_thickness / 2)
    mesh.upper_right = (upper_right[0], upper_right[1], z_value + z_thickness / 2)

    mesh_filter = openmc.MeshFilter(mesh)
    mesh_tally = openmc.Tally(name=name_mesh_tally)
    particle_filter = openmc.ParticleFilter([particule_type])
    mesh_tally.filters = [mesh_filter, particle_filter]
    mesh_tally.scores = ['flux']
    return mesh_tally

def mesh_tally_xz(
    name_mesh_tally="flux_mesh",
    particule_type='neutrons',
    bin_number=400,
    lower_left=(-50.0, -50.0),
    upper_right=(50.0, 50.0),
    y_value:float = 0.0,
    y_thickness:float = 1.0
    )-> object:
    """
    Create a mesh tally in the XZ plane at a specified Y value.

    Parameters:
        name_mesh_tally (str): Name of the tally.
        particule_type (str): Particle type.
        bin_number (int): Number of bins in each direction.
        lower_left (tuple): Lower left corner of the mesh (x, z).
        upper_right (tuple): Upper right corner of the mesh (x, z).
        y_value (float): Y coordinate of the mesh center.
        y_thickness (float): Thickness of the mesh in the Y direction.

    Returns:
        openmc.Tally: The mesh tally object.
    """
    mesh = openmc.RegularMesh()
    mesh.dimension = [bin_number, 1, bin_number]
    mesh.lower_left = (lower_left[0], y_value - y_thickness / 2, lower_left[1])
    mesh.upper_right = (upper_right[0], y_value + y_thickness / 2, upper_right[1])

    mesh_filter = openmc.MeshFilter(mesh)
    mesh_tally = openmc.Tally(name=name_mesh_tally)
    particle_filter = openmc.ParticleFilter([particule_type])
    mesh_tally.filters = [mesh_filter, particle_filter]
    mesh_tally.scores = ['flux']
    return mesh_tally


def mesh_tally_yz(
    name_mesh_tally="flux_mesh",
    particule_type='neutrons',
    bin_number=400,
    lower_left=(-50.0, -50.0),
    upper_right=(50.0, 50.0),
    x_value:float = 0.0,
    x_thickness:float = 1.0
    )-> object:
    """
    Create a mesh tally in the YZ plane at a specified X value.

    Parameters:
        name_mesh_tally (str): Name of the tally.
        particule_type (str): Particle type.
        bin_number (int): Number of bins in each direction.
        lower_left (tuple): Lower left corner of the mesh (y, z).
        upper_right (tuple): Upper right corner of the mesh (y, z).
        x_value (float): X coordinate of the mesh center.
        x_thickness (float): Thickness of the mesh in the X direction.

    Returns:
        openmc.Tally: The mesh tally object.
    """
    mesh = openmc.RegularMesh()
    mesh.dimension = [1, bin_number, bin_number]
    mesh.lower_left = (x_value - x_thickness / 2, lower_left[0], lower_left[1])
    mesh.upper_right = (x_value + x_thickness / 2, upper_right[0], upper_right[1])

    mesh_filter = openmc.MeshFilter(mesh)
    mesh_tally = openmc.Tally(name=name_mesh_tally)
    particle_filter = openmc.ParticleFilter([particule_type])
    mesh_tally.filters = [mesh_filter, particle_filter]
    mesh_tally.scores = ['flux']
    return mesh_tally


def dammage_energy_mesh_xy(
    name_mesh_tally="dammage_energy_mesh",
    bin_number=400,
    lower_left=(-50.0, -50.0),
    upper_right=(50.0, 50.0),
    z_value:float = 0.0,
    z_thickness:float = 1.0
    )-> object:
    """
    Create a mesh tally for damage energy in the XY plane at a specified Z value.

    Parameters:
        name_mesh_tally (str): Name of the tally.
        bin_number (int): Number of bins in each direction.
        lower_left (tuple): Lower left corner of the mesh (x, y).
        upper_right (tuple): Upper right corner of the mesh (x, y).
        z_value (float): Z coordinate of the mesh center.
        z_thickness (float): Thickness of the mesh in the Z direction.

    Returns:
        openmc.Tally: The mesh tally object.
    """
    mesh = openmc.RegularMesh()
    mesh.dimension = [bin_number, bin_number, 1]
    mesh.lower_left = (lower_left[0], lower_left[1], z_value - z_thickness / 2)
    mesh.upper_right = (upper_right[0], upper_right[1], z_value + z_thickness / 2)

    mesh_filter = openmc.MeshFilter(mesh)
    mesh_tally = openmc.Tally(name=name_mesh_tally)
    mesh_tally.filters = [mesh_filter]
    mesh_tally.scores = ['damage-energy']
    return mesh_tally


def dammage_energy_mesh_yz(
    name_mesh_tally="dammage_energy_mesh",
    bin_number=400,
    lower_left=(-50.0, -50.0),
    upper_right=(50.0, 50.0),
    x_value:float = 0.0,
    x_thickness:float = 1.0
    )-> object:
    """
    Create a mesh tally for damage energy in the YZ plane at a specified X value.

    Parameters:
        name_mesh_tally (str): Name of the tally.
        bin_number (int): Number of bins in each direction.
        lower_left (tuple): Lower left corner of the mesh (y, z).
        upper_right (tuple): Upper right corner of the mesh (y, z).
        x_value (float): X coordinate of the mesh center.
        x_thickness (float): Thickness of the mesh in the X direction.

    Returns:
        openmc.Tally: The mesh tally object.
    """
    mesh = openmc.RegularMesh()
    mesh.dimension = [1, bin_number, bin_number]
    mesh.lower_left = (x_value - x_thickness / 2, lower_left[0], lower_left[1])
    mesh.upper_right = (x_value + x_thickness / 2, upper_right[0], upper_right[1])

    mesh_filter = openmc.MeshFilter(mesh)
    mesh_tally = openmc.Tally(name=name_mesh_tally)
    mesh_tally.filters = [mesh_filter]
    mesh_tally.scores = ['damage-energy']
    return mesh_tally


def dammage_energy_mesh_xz(
    name_mesh_tally="dammage_energy_mesh",
    bin_number=400,
    lower_left=(-50.0, -50.0),
    upper_right=(50.0, 50.0),
    y_value:float = 0.0,
    y_thickness:float = 1.0
    )-> object:
    """
    Create a mesh tally for damage energy in the XZ plane at a specified Y value.

    Parameters:
        name_mesh_tally (str): Name of the tally.
        bin_number (int): Number of bins in each direction.
        lower_left (tuple): Lower left corner of the mesh (x, z).
        upper_right (tuple): Upper right corner of the mesh (x, z).
        y_value (float): Y coordinate of the mesh center.
        y_thickness (float): Thickness of the mesh in the Y direction.

    Returns:
        openmc.Tally: The mesh tally object.
    """
    mesh = openmc.RegularMesh()
    mesh.dimension = [bin_number, 1, bin_number]
    mesh.lower_left = (lower_left[0], y_value - y_thickness / 2, lower_left[1])
    mesh.upper_right = (upper_right[0], y_value + y_thickness / 2, upper_right[1])

    mesh_filter = openmc.MeshFilter(mesh)
    mesh_tally = openmc.Tally(name=name_mesh_tally)
    mesh_tally.filters = [mesh_filter]
    mesh_tally.scores = ['damage-energy']
    return mesh_tally



def mesh_tally_dose_xy(
    name_mesh_tally="flux_mesh",
    particule_type='neutron',
    irradiation_geometry='ISO',
    bin_number=400,
    lower_left=(-50.0, -50.0),
    upper_right=(50.0, 50.0),
    z_value:float = 0.0,
    z_thickness:float = 1.0
    )-> object:
    """
    Create a mesh tally for dose in the XY plane at a specified Z value.
    Data source is IRCP 116.

    Parameters:
        name_mesh_tally (str): Name of the tally.
        particule_type (str): Particle type ('neutron' or 'photon').
        irradiation_geometry (str): Irradiation geometry for dose coefficients ('AP', 'PA', 'LLAT', 'RLAT', 'ROT', 'ISO').
        bin_number (int): Number of bins in each direction.
        lower_left (tuple): Lower left corner of the mesh (x, y).
        upper_right (tuple): Upper right corner of the mesh (x, y).
        z_value (float): Z coordinate of the mesh center.
        z_thickness (float): Thickness of the mesh in the Z direction.

    Returns:
        openmc.Tally: The mesh tally object.
    """
    mesh = openmc.RegularMesh()
    if particule_type == 'neutron':
        energy_bins, dose_coeffs = openmc.data.dose_coefficients(particle="neutron", geometry=irradiation_geometry)
        energy_function_filter = openmc.EnergyFunctionFilter(energy_bins, dose_coeffs)
    elif particule_type == 'photon':
        energy_bins, dose_coeffs = openmc.data.dose_coefficients(particle="photon", geometry=irradiation_geometry)
        energy_function_filter = openmc.EnergyFunctionFilter(energy_bins, dose_coeffs)
    else:
        raise ValueError("particule_type must be 'neutron' or 'photon'")

    mesh.dimension = [bin_number, bin_number, 1]
    mesh.lower_left = (lower_left[0], lower_left[1], z_value - z_thickness / 2)
    mesh.upper_right = (upper_right[0], upper_right[1], z_value + z_thickness / 2)

    mesh_filter = openmc.MeshFilter(mesh)
    mesh_tally = openmc.Tally(name=name_mesh_tally)
    particle_filter = openmc.ParticleFilter([particule_type])
    mesh_tally.filters = [mesh_filter, particle_filter, energy_function_filter]
    mesh_tally.scores = ['flux']
    return mesh_tally