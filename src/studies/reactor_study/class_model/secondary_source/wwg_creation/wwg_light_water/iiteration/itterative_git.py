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

factor = 5
MATERIAL_DICT_ORIGINAL = deepcopy(material_dict)
new_material_dict = deepcopy(material_dict)

for material_name, material in new_material_dict.items():
    new_material_dict[material_name] = reducing_density(material, factor=factor)

new_material_dict["FUEL_UO2_MATERIAL"].remove_element("U")

my_reactor.material = new_material_dict
model = my_reactor.model

# add random ray 

# Mesh (from model geometry)
def create_weight_window(
    model: openmc.Model,
    mesh_dimension: tuple[int, int, int] = (50, 50, 50),
    side_length_mesh: float = 850.0,
    batches_number: int = 10,
    particles_per_batch: int = 10000,
    particle_types: tuple[str, ...] = ("neutron", "photon"),
    num_iterations: int = 15, 
    rm_intermediate_files: bool = True
) -> None:
    mesh = openmc.RegularMesh().from_domain(model.geometry)
    mesh.dimension = tuple(mesh_dimension)
    mesh.lower_left = (-side_length_mesh, -side_length_mesh, -side_length_mesh)
    mesh.upper_right = (side_length_mesh, side_length_mesh, side_length_mesh)

    mesh_filter = openmc.MeshFilter(mesh)

    tallies = openmc.Tallies()

    if "neutron" in particle_types:
        flux_tally_neutrons = openmc.Tally(name="flux_tally_neutron")
        flux_tally_neutrons.filters = [mesh_filter, openmc.ParticleFilter("neutron")]
        flux_tally_neutrons.scores = ["flux"]
        flux_tally_neutrons.id = 55  
        tallies.append(flux_tally_neutrons)

    if "photon" in particle_types:
        flux_tally_photons = openmc.Tally(name="flux_tally_photon")
        flux_tally_photons.filters = [mesh_filter, openmc.ParticleFilter("photon")]
        flux_tally_photons.scores = ["flux"]
        flux_tally_photons.id = 56
        tallies.append(flux_tally_photons)

    settings = openmc.Settings()
    src = openmc.FileSource('surface_source.h5')
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

    def plot_mesh_tally_and_weight_window(
        statepoint_filename: str, 
        weight_window_filename: str, 
        image_filename: str,
        particle_type: str = 'neutron'
    ) -> None:

        with openmc.StatePoint(statepoint_filename) as sp:
            flux_tally = sp.get_tally(name=f"flux_tally_{particle_type}")

        tally_mesh = flux_tally.find_filter(openmc.MeshFilter).mesh
        tally_mesh_extent = tally_mesh.bounding_box.extent['xy']

        flux_mean_xy = flux_tally.get_reshaped_data(value='mean', expand_dims=True).squeeze()[:,:,int(mesh.dimension[2]/2)]
        flux_std_dev_xy = flux_tally.get_reshaped_data(value='std_dev', expand_dims=True).squeeze()[:,:,int(mesh.dimension[2]/2)]
        
        flux_rel_err_xy = np.divide(flux_std_dev_xy, flux_mean_xy, out=np.zeros_like(flux_std_dev_xy), where=flux_mean_xy!=0)
        flux_rel_err_xy[flux_rel_err_xy == 0.0] = np.nan
        
        flux_mean_xz = flux_tally.get_reshaped_data(value='mean', expand_dims=True).squeeze()[:,int(mesh.dimension[1]/2),:]
        flux_std_dev_xz = flux_tally.get_reshaped_data(value='std_dev', expand_dims=True).squeeze()[:,int(mesh.dimension[1]/2),:]   
        flux_rel_err_xz = np.divide(flux_std_dev_xz, flux_mean_xz, out=np.zeros_like(flux_std_dev_xz), where=flux_mean_xz!=0)
        flux_rel_err_xz[flux_rel_err_xz == 0.0] = np.nan

        # get slice of ww lower bounds
        wws=openmc.hdf5_to_wws(weight_window_filename)
        if particle_type == 'photon':
            ww = wws[1]  
        else:
            ww = wws[0]  
        ww_mesh = ww.mesh  # get the mesh that the weight window is mapped on
        ww_mesh_extent = ww_mesh.bounding_box.extent['xy']
        reshaped_ww_vals = ww.lower_ww_bounds.reshape(mesh.dimension)

        # slice on XZ basis, midplane Y axis
        slice_of_ww = reshaped_ww_vals[:,:,int(mesh.dimension[1]/2)]

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        def add_colourbar(ax, im):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            return fig.colorbar(im, cax=cax)

        # add slice of flux to subplots
        im_flux = axes[0, 0].imshow(
            flux_mean_xy.T,
            extent=tally_mesh_extent,
            norm=LogNorm(vmin=1e-10, vmax=1)
        )
        axes[0, 0].set_title("Flux Mean")
        axes[0, 0].set_xlabel("X (cm)")
        axes[0, 0].set_ylabel("Y (cm)")
        add_colourbar(axes[0, 0], im_flux)

        # add slice of flux std dev to subplots
        im_std_dev = axes[0, 1].imshow(
            flux_rel_err_xy.T,
            extent=tally_mesh_extent,
            vmin=0.0,
            vmax=1.0,
            cmap='RdYlGn_r'
        )
        axes[0, 1].set_title("Flux Mean rel. error")
        axes[0, 1].set_xlabel("X (cm)")
        axes[0, 1].set_ylabel("Y (cm)")
        add_colourbar(axes[0, 1], im_std_dev)

        im_ww_lower = axes[0, 2].imshow(
            slice_of_ww.T,
            extent=ww_mesh_extent,
            norm=LogNorm(vmin=1e-14, vmax=1e-1),
        )
        axes[0, 2].set_xlabel("X (cm)")
        axes[0, 2].set_ylabel("Y (cm)")
        axes[0, 2].set_title("WW lower bound")
        add_colourbar(axes[0, 2], im_ww_lower)

        im_flux_xz = axes[1, 0].imshow(
            flux_mean_xz.T,
            extent=ww_mesh.bounding_box.extent['xz'],
            norm=LogNorm(vmin=1e-10, vmax=1)
        )
        axes[1, 0].set_title("Flux Mean")
        axes[1, 0].set_xlabel("X (cm)")
        axes[1, 0].set_ylabel("Z (cm)")
        add_colourbar(axes[1, 0], im_flux_xz)

        im_std_dev_xz = axes[1, 1].imshow(
            flux_rel_err_xz.T,
            extent=ww_mesh.bounding_box.extent['xz'],
            vmin=0.0,
            vmax=1.0,
            cmap='RdYlGn_r'
        )
        axes[1, 1].set_title("Flux Mean rel. error")
        axes[1, 1].set_xlabel("X (cm)")
        axes[1, 1].set_ylabel("Z (cm)")


        add_colourbar(axes[1, 1], im_std_dev_xz)
        im_ww_lower_xz = axes[1, 2].imshow(
            slice_of_ww.T,
            extent=ww_mesh.bounding_box.extent['xz'],
            norm=LogNorm(vmin=1e-14, vmax=1e-1),
        )
        axes[1, 2].set_xlabel("X (cm)")
        axes[1, 2].set_ylabel("Z (cm)")
        axes[1, 2].set_title("WW lower bound")
        add_colourbar(axes[1, 2], im_ww_lower_xz)


        plt.tight_layout()
        plt.savefig(image_filename + f'_{particle_type}.png')
        plt.close()

    with openmc.lib.run_in_memory():
        tally_neutron = openmc.lib.tallies[55]
        tally_photon = openmc.lib.tallies[56]
        wws = openmc.lib.WeightWindows.from_tally(tally_neutron, particle="neutron", )
        wws_photon = openmc.lib.WeightWindows.from_tally(tally_photon, particle='photon')
        
        for i in range(num_iterations):
            openmc.lib.run()
            wws.update_magic(tally_neutron)
            wws_photon.update_magic(tally_photon)
            openmc.lib.export_weight_windows(filename=f'weight_windows{i}.h5')
            openmc.lib.statepoint_write(filename=f'statepoint_itteration_{i}.h5')
            openmc.lib.settings.weight_windows_on = True

            plot_mesh_tally_and_weight_window(
                f'statepoint_itteration_{i}.h5',
                f'weight_windows{i}.h5',
                f'plot_{i}',
                particle_type='neutron'
            )
            plot_mesh_tally_and_weight_window(
                f'statepoint_itteration_{i}.h5',
                f'weight_windows{i}.h5',
                f'plot_{i}',
                particle_type='photon'
            )
        if rm_intermediate_files :
            remove_previous_results()
# Example usage:
create_weight_window(model, num_iterations=3, particles_per_batch=1000)
