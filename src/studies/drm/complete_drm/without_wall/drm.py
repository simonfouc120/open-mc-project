# -*- coding: utf-8 -*-
import openmc
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[5]))  # Adjust path to
from parameters.parameters_paths import PATH_TO_CROSS_SECTIONS
from parameters.parameters_materials import CS137_MATERIAL, CDTE_MATERIAL, AIR_MATERIAL, CONCRETE_MATERIAL
from src.utils.pre_processing.pre_processing import remove_previous_results, parallelepiped, plot_geometry, mesh_tally_plane
from src.utils.post_preocessing.post_processing import load_mesh_tally, gaussian_energy_broadening, Pulse_height_tally
os.environ["OPENMC_CROSS_SECTIONS"] = PATH_TO_CROSS_SECTIONS

materials = openmc.Materials([CS137_MATERIAL, CDTE_MATERIAL, AIR_MATERIAL, CONCRETE_MATERIAL])

# Surfaces
sphere = openmc.Sphere(r=1.0, surface_id=1)
detector = openmc.Sphere(x0=10., r=1.0, surface_id=2)
outer_boundary = openmc.Sphere(r=200.0, surface_id=3, boundary_type='vacuum')


detector_cell = openmc.Cell(name="detector_cell")
detector_cell.fill = CDTE_MATERIAL
detector_cell.region = -detector

# Air cell (everything else inside the outer boundary, minus source and detector)
outer_boundary_cell = -outer_boundary
void_region = outer_boundary_cell & ~detector_cell.region
void_cell = openmc.Cell(name="air_cell", fill=AIR_MATERIAL, region=void_region)

universe = openmc.Universe(cells=[detector_cell, void_cell])
geometry = openmc.Geometry(universe)

geometry.export_to_xml()
materials.export_to_xml()

energy = np.linspace(1, 1.0e6, 100)  # Energy range for the photon source (1 keV to 1 MeV)
dict_spectrum = {} 
for e in energy:
    # Création de la source
    source = openmc.IndependentSource()
    source.space = openmc.stats.spherical_uniform(r_outer=1.25)
    source.energy = openmc.stats.Discrete([e], [1.0])  # Énergie du photon de 662 keV pour Cs137
    source.angle = openmc.stats.Monodirectional([1.0, 0.0, 0.0])
    source.particle = "photon"

    tallies = openmc.Tallies([])
    # Tally pour le spectre d'énergie déposée dans le détecteur
    energy_bins = np.linspace(1e-3, 1.0e6, 1001)
    energy_filter = openmc.EnergyFilter(energy_bins)
    cell_filter = openmc.CellFilter(detector_cell)

    energy_dep_tally = openmc.Tally(name="pulse-height")
    energy_dep_tally.filters = [cell_filter, energy_filter]
    energy_dep_tally.scores = ["pulse-height"]
    tallies.append(energy_dep_tally)

    # Configuration de la simulation
    settings = openmc.Settings()
    batches_number = 100
    settings.batches = batches_number
    settings.particles = 10**4
    settings.source = source
    settings.photon_transport = True
    settings.run_mode = "fixed source"
    settings.verbose = True

    # Export des fichiers nécessaires pour la simulation
    settings.export_to_xml()
    tallies.export_to_xml()

    # Exécution de la simulation
    remove_previous_results(batches_number)
    os.environ["OMP_NUM_THREADS"] = "1"
    openmc.run()
    print("Calcul fini")

    sp = openmc.StatePoint(f"statepoint.{batches_number}.h5")
    pht = Pulse_height_tally(name="pulse-height")
    spectrum, spectrum_std = pht.get_spectrum(statepoint_file=sp, normalize=True)
    
    # Store the spectrum along with the corresponding energy value
    dict_spectrum[f"{e:.1f}"] = {
        "spectrum": np.array(spectrum),
        "spectrum_std": np.array(spectrum_std)
    }

energy_bins /= 1e6
energy /= 1e6  # Convert energy to MeV for plotting
spectrums = []

for e, data in dict_spectrum.items():
    spectrum = data["spectrum"]
    spectrum_std = data["spectrum_std"]
    spectrums.append(spectrum)

spectrums = np.array(spectrums)
spectrums = spectrums.reshape(-1, len(spectrums[0]))
spectrums = spectrums.T  
X, Y = np.meshgrid(energy, energy_bins[:-1])

plt.figure()
plt.pcolormesh(X, Y, spectrums, cmap='viridis', norm=LogNorm(), shading='auto')
plt.colorbar(label='Counts')
plt.ylabel('Energy [MeV]')
plt.xlabel('Incident Energy [MeV]')
plt.title('DRM')
num_ticks = 10
tick_indices = np.linspace(0, len(energy) - 1, num_ticks, dtype=int)
plt.xticks(
    ticks=energy[tick_indices],
    labels=[f"{energy[i]:.1f}" for i in tick_indices],
    rotation=45
)
plt.tight_layout()
plt.savefig("drm_spectrum.png")
plt.show()

np.savez("drm_spectrum.npz", energy=energy, energy_bins=energy_bins, spectrums=spectrums, spectrum_std=spectrum_std)