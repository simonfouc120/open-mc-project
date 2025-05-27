import openmc
import openmc_plotter
import os 
import json
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
import sys 
import numpy as np

CWD = Path(__file__).parent.resolve()

project_root = Path(__file__).resolve().parents[3]  # remonte de src/studies/simulation_cs_137
sys.path.append(str(project_root))
from parameters.parameters_paths import PATH_TO_CROSS_SECTIONS
from parameters.parameters_materials import CS137_MATERIAL, CDTE_MATERIAL, AIR_MATERIAL, CONCRETE_MATERIAL
from src.utils.pre_processing.pre_processing import remove_previous_results, parallelepiped
os.environ["OPENMC_CROSS_SECTIONS"] = PATH_TO_CROSS_SECTIONS

materials = openmc.Materials([CS137_MATERIAL, CDTE_MATERIAL, AIR_MATERIAL, CONCRETE_MATERIAL])

# Surfaces
sphere = openmc.Sphere(r=1.0, surface_id=1)
detector = openmc.Sphere(x0=30., r=10.0, surface_id=2)
outer_boundary = openmc.Sphere(r=200.0, surface_id=3, boundary_type='vacuum')  # Limite du monde

# Create concrete wall using parallelepiped
wall_region = parallelepiped(-40, -20, -50, 50, -50, 50, surface_id_start=10)
wall_cell = openmc.Cell(name="concrete_wall", fill=CONCRETE_MATERIAL, region=wall_region)

# Cells
source_cell = openmc.Cell(name="source_cell")
source_cell.fill = CS137_MATERIAL
source_cell.region = -sphere

detector_cell = openmc.Cell(name="detector_cell")
detector_cell.fill = CDTE_MATERIAL
detector_cell.region = -detector

# Air cell (everything else inside the outer boundary, minus source, detector, and wall)
outer_boundary_cell = -outer_boundary 
void_region = outer_boundary_cell & ~source_cell.region & ~detector_cell.region & ~wall_region
void_cell = openmc.Cell(name="air_cell", fill=AIR_MATERIAL, region=void_region)

universe = openmc.Universe(cells=[source_cell, detector_cell, wall_cell, void_cell])
geometry = openmc.Geometry(universe)

# Création de la source

source = openmc.Source()
source.space = openmc.stats.Point((0, 0, 0))
source.energy = openmc.stats.Discrete([661_700], [1.0])  # Énergie du photon de 662 keV pour Cs137
# source.energy = openmc.data.decay_photon_energy("Ba137_m1")
source.angle = openmc.stats.Isotropic()  # Distribution isotrope des angles
source.particle = "photon"
# source.strength = 1E6  # TODO mettre dans une constante ou variable 

# Création des tallies

# Tally du calcul du flux dans le détecteur en p/p-source
tally = openmc.Tally(name="detector_tally")
tally.scores = ["flux"]
particle_filter = openmc.ParticleFilter(['photon'])
tally.filters = [openmc.CellFilter(detector_cell), particle_filter]
tallies = openmc.Tallies([tally])

# Mesh tally de dose 
mesh = openmc.RegularMesh()
mesh.dimension = [100, 100]  # XY
mesh.lower_left = [-50.0, -50.0]
mesh.upper_right = [50.0, 50.0]

mesh_filter = openmc.MeshFilter(mesh)
mesh_tally = openmc.Tally(name='flux_mesh')
particle_filter = openmc.ParticleFilter(['photon'])
mesh_tally.filters = [mesh_filter, particle_filter]
mesh_tally.scores = ['flux']
tallies.append(mesh_tally)

# Tally pour le spectre d'énergie déposée dans le détecteur
energy_bins = np.linspace(1e-3, 1.0e6, 500)  # de 1 keV à 2 MeV en 500 bins
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
materials.export_to_xml()
geometry.export_to_xml()
settings.export_to_xml()
tallies.export_to_xml()

# Exécution de la simulation

remove_previous_results(batches_number)
os.environ["OMP_NUM_THREADS"] = "1"
openmc.run()

print("Calcul fini")  # TODO : faire fonction print calcul terminé 

sp = openmc.StatePoint(f"statepoint.{batches_number}.h5")
tally = sp.get_tally(name="detector_tally")
flux_mean = tally.mean.flatten()
flux_std_dev = tally.std_dev.flatten()

print("Flux moyen :", flux_mean[0]/ source.strength)  # TODO remplacer par constante 
print("Écart-type :", flux_std_dev[0]/ source.strength)


results = {
    "flux_mean": float(flux_mean[0] / source.strength),
    "flux_std_dev": float(flux_std_dev[0] / source.strength)
}

with open(CWD / "results.json", "w") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

# Charger le fichier de sortie
sp = openmc.StatePoint(f"statepoint.{batches_number}.h5")

### mesh tallty ####
# Récupérer le tally du maillage
mesh_tally = sp.get_tally(name='flux_mesh')
flux_data = mesh_tally.mean.reshape((100, 100))
flux_data /= source.strength   # TODO remplacerr par constante 
# Affichage avec échelle logarithmique
plt.imshow(flux_data, 
           origin='lower', 
           extent=[-50, 50, -50, 50], 
           cmap='plasma',
           norm=LogNorm(vmin=np.min(flux_data[flux_data!=0]), vmax=flux_data.max()))  # vmin > 0 obligatoire

plt.colorbar(label='Flux [a.u.] (log scale)')
plt.title('Carte de flux XY (échelle log)')
plt.xlabel('X [cm]')
plt.ylabel('Y [cm]')
plt.tight_layout()
plt.savefig(CWD / "mesh_tally.png")
plt.show()

### spectre ####
sp = openmc.StatePoint(f"statepoint.{batches_number}.h5")
tally = sp.get_tally(name="pulse-height")
pulse_height_values = tally.get_values(scores=['pulse-height']).flatten()

# Récupération des énergies moyennes par bin (approximation)
energy_bin_centers = energy_bins[1:] + 0.5 * (energy_bins[1] - energy_bins[0])
energy_bin_centers /= 1e6
# Moyenne et écart-type de l'énergie déposée
spectrum = tally.mean.flatten()
spectrum_std = tally.std_dev.flatten()

# Tracé
plt.figure()
plt.semilogy(energy_bin_centers, spectrum)
plt.xlabel("Énergie [MeV]")
plt.ylabel("Occurence")
plt.title("Spectre d'énergie déposée dans le détecteur")
plt.grid(True)
plt.tight_layout()
plt.savefig(CWD / "spectrum.png")
plt.show()
