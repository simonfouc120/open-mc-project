import openmc
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

def print_calculation_finished():
    """
    Print a message indicating that the calculation has finished.
    """
    print("######################################################")
    print("Calculation finished successfully.")  # TODO: Implement a more sophisticated logging mechanism if needed.
    print("######################################################")


def load_mesh_tally(cwd, statepoint_file: object, name_mesh_tally:str = "flux_mesh_tally", 
                    bin_number:int=400, lower_left:tuple=(-10.0, -10.0), 
                    upper_right:tuple=(10.0, 10.0), zoom_x:tuple=(-10, 10), 
                    zoom_y:tuple=(-10.0, 10.0), plane:str = "xy",):
    """
    Load and plot the mesh tally from the statepoint file.

    Parameters:
    - cwd: Path or directory where the mesh tally image will be saved.
    - statepoint_file: The OpenMC statepoint file object containing the tally data.
    - name_mesh_tally_saving: Name of the tally (default is "mesh_tally.png").
    - bin_number: Number of bins for the mesh tally in each dimension (default is 400).
    - lower_left: Tuple specifying the lower left corner of the mesh (default is (-10.0, -10.0)).
    - upper_right: Tuple specifying the upper right corner of the mesh (default is (10.0, 10.0)).
    - zoom_x: Tuple specifying the x-axis limits for zooming (default is (-10, 10)).
    - zoom_y: Tuple specifying the y-axis limits for zooming (default is (-10.0, 10.0)).

    This function extracts the mesh tally from the statepoint file, reshapes the data,
    and plots it using matplotlib with a logarithmic color scale. The resulting plot
    is saved as a PNG file in the specified directory and displayed.
    """
    mesh_tally = statepoint_file.get_tally(name=name_mesh_tally)
    flux_data = mesh_tally.mean.reshape((bin_number, bin_number))
    plt.imshow(flux_data, 
            origin='lower', 
            extent=[lower_left[0], upper_right[1], lower_left[1], upper_right[1]], 
            cmap='plasma',
            norm=LogNorm(vmin=np.min(flux_data[flux_data!=0]), vmax=flux_data.max()))  
    plt.colorbar(label='Flux [a.u.] (log scale)')
    if plane == "xy":
        plt.title('Carte de flux XY (échelle log)')
        plt.xlabel('X [cm]')
        plt.ylabel('Y [cm]')
    elif plane == "xz":
        plt.title('Carte de flux XZ (échelle log)')
        plt.xlabel('X [cm]')
        plt.ylabel('Z [cm]')
    elif plane == "yz":
        plt.title('Carte de flux YZ (échelle log)')
        plt.xlabel('Y [cm]')
        plt.ylabel('Z [cm]')
    else:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")
    plt.tight_layout()
    plt.xlim(zoom_x[0], zoom_x[1])
    plt.ylim(zoom_y[0], zoom_y[1])
    name_mesh_tally_saving = name_mesh_tally + ".png"
    plt.savefig(cwd / name_mesh_tally_saving)
    plt.show()

