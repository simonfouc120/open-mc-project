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


def load_mesh_tally(cwd, statepoint_file: object, name_mesh_tally:str = "flux_mesh_tally", particule_type:str='neutron',
                    bin_number:int=400, lower_left:tuple=(-10.0, -10.0), 
                    upper_right:tuple=(10.0, 10.0), zoom_x:tuple=(-10, 10), 
                    zoom_y:tuple=(-10.0, 10.0), plane:str = "xy", saving_figure:bool = True):
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
        plt.title(f"Flux map XY {particule_type} (log scale)")
        plt.xlabel('X [cm]')
        plt.ylabel('Y [cm]')
    elif plane == "xz":
        plt.title(f'Flux map XY {particule_type} (log scale)')
        plt.xlabel('X [cm]')
        plt.ylabel('Z [cm]')
    elif plane == "yz":
        plt.title(f'Flux map XY {particule_type} (log scale)')
        plt.xlabel('Y [cm]')
        plt.ylabel('Z [cm]')
    else:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")
    plt.tight_layout()
    plt.xlim(zoom_x[0], zoom_x[1])
    plt.ylim(zoom_y[0], zoom_y[1])
    name_mesh_tally_saving = name_mesh_tally + ".png"
    if saving_figure:
        plt.savefig(cwd / name_mesh_tally_saving)
    plt.show()

def load_dammage_energy_tally(cwd, statepoint_file: object, name_mesh_tally:str = "dammage_energy_mesh", 
                                bin_number:int=500, lower_left:tuple=(-10.0, -10.0), 
                                upper_right:tuple=(10.0, 10.0), zoom_x:tuple=(-10, 10), 
                                zoom_y:tuple=(-10.0, 10.0), plane:str = "xy", saving_figure:bool = True):
    """
    Load and plot the damage energy mesh tally from the statepoint file.

    Parameters:
    - cwd: Path or directory where the mesh tally image will be saved.
    - statepoint_file: The OpenMC statepoint file object containing the tally data.
    - name_mesh_tally: Name of the tally (default is "dammage_energy_mesh").
    - bin_number: Number of bins for the mesh tally in each dimension (default is 500).
    - lower_left: Tuple specifying the lower left corner of the mesh (default is (-10.0, -10.0)).
    - upper_right: Tuple specifying the upper right corner of the mesh (default is (10.0, 10.0)).
    - zoom_x: Tuple specifying the x-axis limits for zooming (default is (-10, 10)).
    - zoom_y: Tuple specifying the y-axis limits for zooming (default is (-10.0, 10.0)).
    
    This function extracts the damage energy mesh tally from the statepoint file,
    reshapes the data, and plots it using matplotlib with a logarithmic color scale.
    The resulting plot is saved as a PNG file in the specified directory and displayed.
    """
    dpa_result = statepoint_file.get_tally(name=name_mesh_tally)

    dpa_data = dpa_result.get_slice(scores=['damage-energy']).mean.ravel()
    dpa_data.shape = (500, 500)

    plt.figure(figsize=(8, 6))
    plt.imshow(dpa_data, norm=LogNorm(), extent=[lower_left[0], upper_right[1], lower_left[1], upper_right[1]], origin='lower', cmap='plasma')
    plt.title('Dammage energy Map')
    if plane == "xy":
        plt.title('Damage energy map XY (log scale)')
        plt.xlabel('X [cm]')
        plt.ylabel('Y [cm]')    
    elif plane == "xz":
        plt.title('Damage energy map XY (log scale)')
        plt.xlabel('X [cm]')
        plt.ylabel('Z [cm]')
    elif plane == "yz":
        plt.title('Damage energy map XY (log scale)')
        plt.xlabel('Y [cm]')
        plt.ylabel('Z [cm]')
    else:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")
    if saving_figure:
        plt.savefig(cwd / f"{name_mesh_tally}.png")
    plt.colorbar(label='eV/p-source')
    plt.tight_layout()
    plt.show()


def load_mesh_tally_dose(cwd, statepoint_file: object, name_mesh_tally:str = "flux_mesh_neutrons_dose_xy", 
                        n_per_second:int=1, particule_type:str='neutrons',
                        bin_number:int=400, lower_left:tuple=(-10.0, -10.0), 
                        upper_right:tuple=(10.0, 10.0), zoom_x:tuple=(-10, 10), 
                        zoom_y:tuple=(-10.0, 10.0), plane:str = "xy", saving_figure:bool = True):
    """
    Load and plot the dose mesh tally from the statepoint file.

    Parameters:
    - cwd: Path or directory where the mesh tally image will be saved.
    - statepoint_file: The OpenMC statepoint file object containing the tally data.
    - name_mesh_tally: Name of the tally (default is "flux_mesh_neutrons_xy").
    - bin_number: Number of bins for the mesh tally in each dimension (default is 400).
    - lower_left: Tuple specifying the lower left corner of the mesh (default is (-10.0, -10.0)).
    - upper_right: Tuple specifying the upper right corner of the mesh (default is (10.0, 10.0)).
    - zoom_x: Tuple specifying the x-axis limits for zooming (default is (-10, 10)).
    - zoom_y: Tuple specifying the y-axis limits for zooming (default is (-10.0, 10.0)).
    
    This function extracts the dose mesh tally from the statepoint file,
    reshapes the data, and plots it using matplotlib with a logarithmic color scale.
    The resulting plot is saved as a PNG file in the specified directory and displayed.
    """
    mesh_tally = statepoint_file.get_tally(name=name_mesh_tally)
    flux_data = mesh_tally.mean.reshape((bin_number, bin_number))
    flux_data = flux_data * n_per_second  # Convert to dose rate (assuming n_per_second is the number of particles per second)
    flux_data = flux_data * 1e-12 * 3600  # Convert from pSv/s to µSv/h

    
    plt.imshow(flux_data, 
            origin='lower', 
            extent=[lower_left[0], upper_right[1], lower_left[1], upper_right[1]], 
            cmap='plasma',
            norm=LogNorm(vmin=np.min(flux_data[flux_data!=0]), vmax=flux_data.max()))  
    plt.colorbar(label='Dose [μSv/h] (log scale)')
    
    if plane == "xy":
        plt.title(f'Dose {particule_type} cartography XY')
        plt.xlabel('X [cm]')
        plt.ylabel('Y [cm]')
    elif plane == "xz":
        plt.title(f'Dose {particule_type} cartography XY')
        plt.xlabel('X [cm]')
        plt.ylabel('Z [cm]')
    elif plane == "yz":
        plt.title(f'Dose {particule_type} cartography XY')
        plt.xlabel('Y [cm]')
        plt.ylabel('Z [cm]')
    else:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")
    plt.tight_layout()
    plt.xlim(zoom_x[0], zoom_x[1])
    plt.ylim(zoom_y[0], zoom_y[1])
    name_mesh_tally_saving = name_mesh_tally + ".png"
    if saving_figure:
        plt.savefig(cwd / name_mesh_tally_saving)
    plt.show()