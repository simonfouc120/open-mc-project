import os
import openmc

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

