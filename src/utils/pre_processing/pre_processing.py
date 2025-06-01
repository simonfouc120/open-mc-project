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


def plot_geometry(materials:list, plane:str="xy", 
                  width: float = 10.0, height: float = 10.0, 
                  origin: tuple = (0, 0, 0), 
                  pixels: tuple = (600, 600)):
    """
    Plots the OpenMC geometry in a plane.
    """

    plot = openmc.Plot()

    plot.origin = origin
    plot.width = (width, height)
    plot.pixels = pixels

    X_VALUES = np.linspace(-width//2, width//2, pixels[0])
    Y_VALUES = np.linspace(-height//2, height//2, pixels[1])

    X, Y = np.meshgrid(X_VALUES, Y_VALUES)

    default_colors = ['red', 'green', 'lightblue', 'gray', 'brown', 'orange', 'purple', 'yellow', 'pink', 'cyan']
    colors = (default_colors * ((len(materials) + len(default_colors) - 1) // len(default_colors)))[:len(materials)]
    plot.colors = {mat: color for mat, color in zip(materials, colors)}

    plot.color_by = 'material'
    plot.basis = plane
    plot.filename = "plot_" + plane + ".png"
    plots = openmc.Plots([plot])
    plots.export_to_xml()
    openmc.plot_geometry()

    img = Image.open("plot_" + plane + ".png")
    plt.imshow(img)
    plt.title("OpenMC Geometry Plot - " + plane.upper() + " Plane")
    plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.xticks(
        np.linspace(0, img.size[0], num=7),
        [f"{x:.1f}" for x in np.linspace(np.min(X_VALUES), np.max(X_VALUES), num=7)]
    )
    plt.yticks(
        np.linspace(0, img.size[1], num=7),
        [f"{y:.1f}" for y in np.linspace(np.min(Y_VALUES), np.max(Y_VALUES), num=7)]
    )
    plt.xlabel("X (cm)")
    plt.ylabel("Y (cm)")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()


def mesh_tally(name_mesh_tally = "flux_mesh", particule_type='neutrons', 
               bin_number=400, lower_left=(-50.0, -50.0), upper_right=(50.0, 50.0)):
    # Mesh tally de dose 
    mesh = openmc.RegularMesh()
    mesh.dimension = [bin_number, bin_number]  # XY
    mesh.lower_left = lower_left
    mesh.upper_right = upper_right

    mesh_filter = openmc.MeshFilter(mesh)
    mesh_tally = openmc.Tally(name=name_mesh_tally)
    particle_filter = openmc.ParticleFilter([particule_type])
    mesh_tally.filters = [mesh_filter, particle_filter]
    mesh_tally.scores = ['flux']
    return mesh_tally