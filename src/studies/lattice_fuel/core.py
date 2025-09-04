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

project_root = Path(__file__).resolve().parents[3]  
sys.path.append(str(project_root))
from parameters.parameters_paths import PATH_TO_CROSS_SECTIONS
from parameters.parameters_materials import FUEL_MATERIAL, HELIUM_MATERIAL, AIR_MATERIAL, CONCRETE_MATERIAL, GRAPHITE_MATERIAL, STEEL_MATERIAL, WATER_MATERIAL
from src.utils.pre_processing.pre_processing import (remove_previous_results, rpp, plot_geometry, mesh_tally_plane, mesh_tally_dose_plane)
from src.utils.post_preocessing.post_processing import load_mesh_tally, load_dammage_energy_tally, load_mesh_tally_dose
os.environ["OPENMC_CROSS_SECTIONS"] = PATH_TO_CROSS_SECTIONS

from src.models.model_hexagon_lattice_fuel import GEOMETRY, MATERIALS, MAIN_CELL

main_cell = MAIN_CELL
material = MATERIALS
geometry = GEOMETRY

material.export_to_xml()

geometry.export_to_xml()

plot_geometry(materials = material, plane="xy", width=16, height=16)

plot_geometry(materials = material, plane="xz", width=16, height=16)

plot_geometry(materials = material, plane="yz", width=70, height=70)

# Calcul de criticité simple 
settings = openmc.Settings()
batches_number= 1500
settings.batches = batches_number
settings.inactive = 20
settings.particles = 50000
settings.source = openmc.Source()
settings.source.space = openmc.stats.Point((0, 0, 0))
settings.source.particle = 'neutron'
settings.photon_transport = True
settings.source.angle = openmc.stats.Isotropic()

# tally pour le flux dans le détecteur
tally = openmc.Tally(name="flux_tally")
tally.scores = ['flux']
tally.filters = [openmc.CellFilter(main_cell)]

# Tally for fission rate
fission_tally = openmc.Tally(name="fission_rate_tally")
fission_tally.scores = ['fission']
fission_tally.filters = [openmc.CellFilter(main_cell)]

# Tally for nu-fission (nu * fission rate)
nu_fission_tally = openmc.Tally(name="nu_fission_rate_tally")
nu_fission_tally.scores = ['nu-fission']
nu_fission_tally.filters = [openmc.CellFilter(main_cell)]

tallies = openmc.Tallies([tally, fission_tally, nu_fission_tally])

# Mesh tally for neutron flux in a specific region
mesh_tally_neutron_xy = mesh_tally_plane(name_mesh_tally = "flux_mesh_neutrons_xy", particule_type='neutron', plane="xy",
                                      bin_number=400, lower_left=(-10.0, -10.0), upper_right=(10.0, 10.0),
                                      thickness= 4.0, coord_value=0.0)
tallies.append(mesh_tally_neutron_xy)

mesh_tally_photon_xy = mesh_tally_plane(name_mesh_tally = "flux_mesh_photons_xy", particule_type='photon', plane="xy",
                                      bin_number=400, lower_left=(-10.0, -10.0), upper_right=(10.0, 10.0),
                                      thickness= 4.0, coord_value=0.0)
tallies.append(mesh_tally_photon_xy)

mesh_tally_neutron_yz = mesh_tally_plane(name_mesh_tally = "flux_mesh_neutrons_yz", particule_type='neutron', plane="yz",
                                      bin_number=400, lower_left=(-10.0, -10.0), upper_right=(10.0, 10.0),
                                      thickness= 4.0, coord_value=0.0)
tallies.append(mesh_tally_neutron_yz)

mesh_tally_photon_yz = mesh_tally_plane(name_mesh_tally = "flux_mesh_photons_yz", particule_type='photon', plane="yz",
                                      bin_number=400, lower_left=(-10.0, -10.0), upper_right=(10.0, 10.0),
                                      thickness= 4.0, coord_value=0.0)
tallies.append(mesh_tally_photon_yz)


dammage_energy_tally_xy = mesh_tally_plane(name_mesh_tally="dammage_energy_mesh_xy", bin_number=500, score='damage-energy', 
                                           particule_type='all', plane="xy", lower_left=(-10.0, -10.0), 
                                           upper_right=(10.0, 10.0), coord_value=0.0, thickness=4.0)
tallies.append(dammage_energy_tally_xy)

dammage_energy_tally_yz = mesh_tally_plane(name_mesh_tally="dammage_energy_mesh_yz", bin_number=500, score='damage-energy', 
                                           particule_type='all', plane="yz", lower_left=(-10.0, -10.0), 
                                           upper_right=(10.0, 10.0), coord_value=0.0, thickness=4.0)
tallies.append(dammage_energy_tally_yz)


# dommage_energy_tally_yz = dammage_energy_mesh_yz(name_mesh_tally="dammage_energy_mesh_yz", bin_number=500, lower_left=(-10.0, -10.0), upper_right=(10.0, 10.0), x_value=0.0, x_thickness=1.0)
# tallies.append(dommage_energy_tally_yz)

mesh_tally_neutron_xy_dose = mesh_tally_dose_plane(name_mesh_tally = "flux_mesh_neutrons_xy_dose", particule_type='neutron', plane="xy", 
                                                    bin_number=400, lower_left=(-10.0, -10.0), upper_right=(10.0, 10.0), thickness= 4.0, coord_value=0.0)
tallies.append(mesh_tally_neutron_xy_dose)

mesh_tally_photon_xy_dose = mesh_tally_dose_plane(name_mesh_tally = "flux_mesh_photons_xy_dose", particule_type='photon', plane="xy", 
                                                  bin_number=400, lower_left=(-10.0, -10.0), upper_right=(10.0, 10.0), thickness= 4.0, coord_value=0.0)
tallies.append(mesh_tally_photon_xy_dose)

settings.export_to_xml()
tallies.export_to_xml()

# Run the simulation
remove_previous_results(batches_number=batches_number)
openmc.run()

# Load the statepoint file
statepoint_file = openmc.StatePoint(f'statepoint.{batches_number}.h5')
tally = statepoint_file.get_tally(name="flux_tally")
mean_flux = tally.mean.flatten().tolist()
std_flux = tally.std_dev.flatten().tolist()
# Get fission rate and nu-bar
fission_tally_result = statepoint_file.get_tally(name="fission_rate_tally")
nu_fission_tally_result = statepoint_file.get_tally(name="nu_fission_rate_tally")

fission_rate = float(fission_tally_result.mean.flatten()[0])
nu_fission_rate = float(nu_fission_tally_result.mean.flatten()[0])
nu_bar = nu_fission_rate / fission_rate if fission_rate != 0 else float('nan')
# Estimate fissions per second (fission/s) for a 5 MW reactor

reactor_power = 5e6  # 5 MW in Watts

# Energy released per fission depends on isotope composition
# 19.75% U-235, 80.25% U-238 (by atom fraction)
# Typical energy per fission: U-235 = 202.5 MeV, U-238 = 205.0 MeV

fraction_u235 = 0.1975
fraction_u238 = 1.0 - fraction_u235

energy_u235 = 202.5e6 * 1.60218e-19  # J
energy_u238 = 205.0e6 * 1.60218e-19  # J

# Weighted average energy per fission
energy_per_fission = fraction_u235 * energy_u235 + fraction_u238 * energy_u238

fissions_per_second = reactor_power / energy_per_fission
neutrons_emitted_per_second = fissions_per_second * nu_bar

results = {
    "mean_flux": {
        "value": mean_flux,
        "units": "neutrons/cm^2/source particle"
    },
    "fission_rate": {
        "value": fission_rate,
        "units": "fissions/source particle"
    },
    "fission_rate_per_second": {
        "value": fissions_per_second,
        "units": "fissions/s (for 5 MW reactor)"
    },
    "nu_fission_rate": {
        "value": nu_fission_rate,
        "units": "neutrons produced by fission/source particle"
    },
    "nu_bar": {
        "value": nu_bar,
        "units": "dimensionless (average neutrons per fission)"
    },
    "neutrons_emitted_per_second": {
        "value": neutrons_emitted_per_second,
        "units": "neutrons/s (for 5 MW reactor)"
    },
}

with open(CWD / "simulation_results.json", "w") as f:
    json.dump(results, f, indent=2)

### mesh tallty ####

load_mesh_tally(cwd = CWD, statepoint_file = statepoint_file, name_mesh_tally="flux_mesh_neutrons_xy", particule_type="neutron", bin_number=400,
                lower_left=(-10.0, -10.0), upper_right=(10.0, 10.0), zoom_x=(-10, 10), zoom_y=(-10, 10), plane="xy")

load_mesh_tally(cwd = CWD, statepoint_file = statepoint_file, name_mesh_tally="flux_mesh_photons_xy", particule_type="photon", bin_number=400,
                lower_left=(-10.0, -10.0), upper_right=(10.0, 10.0), zoom_x=(-10, 10), zoom_y=(-10, 10), plane="xy")

load_mesh_tally(cwd = CWD, statepoint_file = statepoint_file, name_mesh_tally="flux_mesh_neutrons_yz", particule_type="neutron", bin_number=400,
                lower_left=(-40.0, -40.0), upper_right=(40.0, 40.0), zoom_x=(-40, 40), zoom_y=(-40, 40), plane="yz")

load_mesh_tally(cwd = CWD, statepoint_file = statepoint_file, name_mesh_tally="flux_mesh_photons_yz", particule_type="photon", bin_number=400,
                lower_left=(-40.0, -40.0), upper_right=(40.0, 40.0), zoom_x=(-40, 40), zoom_y=(-40, 40), plane="yz")

load_dammage_energy_tally(cwd = CWD, statepoint_file = statepoint_file, name_mesh_tally="dammage_energy_mesh_xy",
                         bin_number=500, lower_left=(-10.0, -10.0), upper_right=(10.0, 10.0), 
                         zoom_x=(-10, 10), zoom_y=(-10.0, 10.0), plane="xy")

load_dammage_energy_tally(cwd = CWD, statepoint_file = statepoint_file, name_mesh_tally="dammage_energy_mesh_yz",
                         bin_number=500, lower_left=(-10.0, -10.0), upper_right=(10.0, 10.0), 
                         zoom_x=(-10, 10), zoom_y=(-10.0, 10.0), plane="yz")

load_mesh_tally_dose(cwd = CWD, statepoint_file = statepoint_file, name_mesh_tally="flux_mesh_neutrons_xy_dose",
                     particles_per_second=neutrons_emitted_per_second, particule_type='neutron',
                     bin_number=400, lower_left=(-10.0, -10.0), upper_right=(10.0, 10.0),
                     zoom_x=(-10, 10), zoom_y=(-10.0, 10.0), plane="xy")

load_mesh_tally_dose(cwd = CWD, statepoint_file = statepoint_file, name_mesh_tally="flux_mesh_photons_xy_dose",
                     particles_per_second=neutrons_emitted_per_second, particule_type='photon',
                     bin_number=400, lower_left=(-10.0, -10.0), upper_right=(10.0, 10.0),
                     zoom_x=(-10, 10), zoom_y=(-10.0, 10.0), plane="xy")