import openmc
import openmc.lib 
import os 
import json
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
import sys 
from PIL import Image
import numpy as np
from copy import deepcopy
from mpl_toolkits.axes_grid1 import make_axes_locatable

CWD = Path(__file__).parent.resolve() 
project_root = Path(__file__).resolve().parents[8]  
sys.path.append(str(project_root))

from parameters.parameters_paths import PATH_TO_CROSS_SECTIONS, IMAGE_PATH
from parameters.parameters_materials import *
from src.utils.pre_processing.pre_processing import *
from src.utils.post_preocessing.post_processing import *
from src.models.model_complete_reactor_class import *
from src.utils.weight_window.weight_window import *
from src.models.model_complete_reactor_class import material_dict

my_reactor = Reactor_model(materials=material_dict, 
                           total_height_active_part=500.0, 
                           light_water_pool=True, 
                           light_water_length=900.0, # cm
                           light_water_height=900.0, # cm
                           cavity=False,
                           top_shielding= False,
                           slab_thickness=100,
                           concrete_wall_thickness=100,
                           calculation_sphere_coordinates=(0, 0, 500), 
                           calculation_sphere_radius=50.0)
os.environ["OPENMC_CROSS_SECTIONS"] = PATH_TO_CROSS_SECTIONS

def create_reduced_density_material_dict(material_dict, factor=10):
    """
    Returns a deepcopy of material_dict with all material densities reduced by the given factor.
    """
    new_material_dict = deepcopy(material_dict)
    for material_name, material in new_material_dict.items():
        new_material_dict[material_name] = reducing_density(material, factor=factor)
    return new_material_dict

factor = 16
MATERIAL_DICT_ORIGINAL = deepcopy(material_dict)
new_material_dict = create_reduced_density_material_dict(material_dict, factor=factor)

new_material_dict["FUEL_UO2_MATERIAL"].remove_element("U")

my_reactor.material = new_material_dict
model = my_reactor.model

# add random ray 

# Mesh (from model geometry)
mesh_dimension = (50, 50, 50)
mesh_size = 850.0
mesh = openmc.RegularMesh().from_domain(model.geometry)
mesh.dimension = tuple(mesh_dimension)
mesh.lower_left = (-mesh_size, -mesh_size, -mesh_size)
mesh.upper_right = (mesh_size, mesh_size, mesh_size)

mesh_filter = openmc.MeshFilter(mesh)

flux_tally_neutrons = openmc.Tally(name="flux_tally_neutron")
flux_tally_neutrons.filters = [mesh_filter, openmc.ParticleFilter("neutron")]
flux_tally_neutrons.scores = ["flux"]
flux_tally_neutrons.id = 55  # we set the ID number here as we need to access it during the openmc lib run

flux_tally_photons = openmc.Tally(name="flux_tally_photon")
flux_tally_photons.filters = [mesh_filter, openmc.ParticleFilter("photon")]
flux_tally_photons.scores = ["flux"]
flux_tally_photons.id = 56  # we set the ID number here as we need to access it during the openmc lib run

tallies = openmc.Tallies([flux_tally_neutrons, flux_tally_photons])

settings = openmc.Settings()
batches_number= 10
particles_per_batch = 100000
max_history_splits = 1_000
src = openmc.FileSource('surface_source.h5')
particle_types = ("neutron", "photon")

settings.batches = batches_number
settings.particles = particles_per_batch
settings.run_mode = "fixed source"
src.particles = list(particle_types)  
settings.source = src
settings.photon_transport = True
settings.output = {'tallies': True}

model.settings = settings
model.tallies = tallies
model.export_to_xml()

def plot_mesh_tally_and_weight_window(statepoint_filename, 
                                      weight_window_filename, 
                                      image_filename,
                                      particle_type:str='neutron'):
    # load flux tally from statepoint file
    with openmc.StatePoint(statepoint_filename) as sp:
        flux_tally = sp.get_tally(name=f"flux_tally_{particle_type}")

    tally_mesh = flux_tally.find_filter(openmc.MeshFilter).mesh
    tally_mesh_extent = tally_mesh.bounding_box.extent['xy']

    flux_mean = flux_tally.get_reshaped_data(value='mean', expand_dims=True).squeeze()[:,:,int(mesh.dimension[2]/2)]
    flux_std_dev = flux_tally.get_reshaped_data(value='std_dev', expand_dims=True).squeeze()[:,:,int(mesh.dimension[2]/2)]
    
    flux_rel_err = np.divide(flux_std_dev, flux_mean, out=np.zeros_like(flux_std_dev), where=flux_mean!=0)
    flux_rel_err[flux_rel_err == 0.0] = np.nan
    
    wws=openmc.hdf5_to_wws(weight_window_filename)
    if particle_type == 'photon':
        ww = wws[1]  # get the one and only mesh tally
    else:
        ww = wws[0]  # get the one and only mesh tally
    ww_mesh = ww.mesh  # get the mesh that the weight window is mapped on
    ww_mesh_extent = ww_mesh.bounding_box.extent['xy']
    reshaped_ww_vals = ww.lower_ww_bounds.reshape(mesh.dimension)

    slice_of_ww = reshaped_ww_vals[:,:,int(mesh.dimension[1]/2)]
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    def add_colourbar(ax, im):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        return fig.colorbar(im, cax=cax)

    im_flux = axes[0].imshow(
        flux_mean.T,
        # origin="lower",
        extent=tally_mesh_extent,
        norm=LogNorm(vmin=1e-10, vmax=1)
    )
    axes[0].set_title("Flux Mean")
    add_colourbar(axes[0], im_flux)

    im_std_dev = axes[1].imshow(
        flux_rel_err.T,
        # origin="lower",
        extent=tally_mesh_extent,
        vmin=0.0,
        vmax=1.0,
        cmap='RdYlGn_r'
    )
    axes[1].set_title("Flux Mean rel. error")
    add_colourbar(axes[1], im_std_dev)

    im_ww_lower = axes[2].imshow(
        slice_of_ww.T,
        # origin="lower",
        extent=ww_mesh_extent,
        norm=LogNorm(vmin=1e-14, vmax=1e-1),
    )
    axes[2].set_title("WW lower bound")
    add_colourbar(axes[2], im_ww_lower)
    
    plt.tight_layout()
    plt.savefig(image_filename + f'_{particle_type}.png')
    plt.close()


intermediate_step = 10
# AFFICHER YZ
with openmc.lib.run_in_memory():

    tally_neutron = openmc.lib.tallies[55]
    tally_photon = openmc.lib.tallies[56]

    wws = openmc.lib.WeightWindows.from_tally(tally_neutron, particle="neutron")
    wws_photon = openmc.lib.WeightWindows.from_tally(tally_photon, particle="photon")

    import copy
    original_materials = copy.deepcopy(openmc.lib.materials)

    mult_factor = 1.5
    while factor >= 1.0:
        # Deepcopy original materials to reset densities each iteration
        materials = copy.deepcopy(original_materials)

        for i_step in range(intermediate_step):

            print(f"--- Simulation avec facteur de densité = {factor} ---")
            openmc.lib.materials = materials
            openmc.lib.run()

            wws.update_magic(tally_neutron)
            wws_photon.update_magic(tally_photon)

            i = int(factor)
            openmc.lib.export_weight_windows(f'weight_windows{i}.h5')
            openmc.lib.statepoint_write(f'statepoint_simulation_{i}.h5')

        plot_mesh_tally_and_weight_window(
            f'statepoint_simulation_{i}.h5',
            f'weight_windows{i}.h5',
            f'plot_{i}',
            particle_type='neutron'
        )
        plot_mesh_tally_and_weight_window(
            f'statepoint_simulation_{i}.h5',
            f'weight_windows{i}.h5',
            f'plot_{i}',
            particle_type='photon'
        )
        for mat, name in zip(materials.values(), materials.keys()):
            mat.set_density(materials[name].get_density('g/cm3') * mult_factor, 'g/cm3')
            print(f"Material: {mat.name}, New Density: {mat.get_density('g/cm3')} g/cm3")

        factor /= mult_factor
