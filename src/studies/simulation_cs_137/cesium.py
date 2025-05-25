import openmc
import openmc_plotter
import os 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
import sys 
import numpy as np
project_root = Path(__file__).resolve().parents[3]  # remonte de src/studies/simulation_cs_137
sys.path.append(str(project_root))
from parameters.parameters_paths import PATH_TO_CROSS_SECTIONS

os.environ["OPENMC_CROSS_SECTIONS"] = PATH_TO_CROSS_SECTIONS
cwd = Path.cwd()

# Création des matériaux
cs137 = openmc.Material(name="Cs137")
cs137.add_nuclide("Cs137", 1.0)
cs137.set_density("g/cm3", 4.0)

cdte = openmc.Material(name="CdTe")
cdte.add_element("Cd", 1.0)
cdte.add_element("Te", 1.0)
cdte.set_density("g/cm3", 6.2)

materials = openmc.Materials([cs137, cdte])

air = openmc.Material(name="Air")
air.add_element("N", 0.78)
air.add_element("O", 0.21)
air.add_element("Ar", 0.01)
air.set_density("g/cm3", 0.001225)  

materials.append(air)

sphere = openmc.Sphere(r=1.0, surface_id=1)
detector = openmc.Sphere(x0=30., r=10.0, surface_id=2)
outer_boundary = openmc.Sphere(r=100.0, surface_id=3, boundary_type='vacuum')  # Limite du monde

source_cell = openmc.Cell(name="source_cell")
source_cell.fill = cs137
source_cell.region = -sphere

detector_cell = openmc.Cell(name="detector_cell")
detector_cell.fill = cdte
detector_cell.region = -detector

outer_boundary_cell = -outer_boundary & +detector
void_cell = openmc.Cell(name="air_cell", fill=air, region=outer_boundary_cell)

universe = openmc.Universe(cells=[source_cell, detector_cell, void_cell])
geometry = openmc.Geometry(universe)

# Création de la source
source = openmc.Source()
source.space = openmc.stats.Point((0, 0, 0))
source.energy = openmc.stats.Discrete([661_700], [1.0])  # Énergie du photon de 662 keV pour Cs137
# source.energy = openmc.data.decay_photon_energy("Ba137_m1")
source.angle = openmc.stats.Isotropic()  # Distribution isotrope des angles
source.particle = "photon"
source.strength = 1E6

# Création des tallies

# Tally du calcul du flux dans le détecteur en p/p-source
tally = openmc.Tally(name="detector_tally")
tally.scores = ["flux"]
particle_filter = openmc.ParticleFilter(['photon'])
tally.filters = [openmc.CellFilter(detector_cell), particle_filter]
tallies = openmc.Tallies([tally])

# Mesh tally de dose 
mesh = openmc.RegularMesh()
mesh.dimension = [500, 500]  # XY
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
batches_number = 10
settings.batches = batches_number
settings.particles = 10**5
settings.source = source
settings.photon_transport = True 
settings.run_mode = "fixed source"

# Export des fichiers nécessaires pour la simulation
materials.export_to_xml()
geometry.export_to_xml()
settings.export_to_xml()
tallies.export_to_xml()

# Exécution de la simulation
if os.path.exists("summary.h5"):
    os.remove("summary.h5")

if os.path.exists(f"statepoint.{batches_number}.h5"):
    os.remove(f"statepoint.{batches_number}.h5")  #TODO à mettre en fonction dans utils 

openmc.run()

print("Calcul fini")  # TODO : faire fonction print calcul terminé 

sp = openmc.StatePoint(f"statepoint.{batches_number}.h5")
tally = sp.get_tally(name="detector_tally")
flux_mean = tally.mean.flatten()
flux_std_dev = tally.std_dev.flatten()

print("Flux moyen :", flux_mean[0]/ source.strength)
print("Écart-type :", flux_std_dev[0]/ source.strength)

# Charger le fichier de sortie
sp = openmc.StatePoint(f"statepoint.{batches_number}.h5")


### mesh tallty ####
# Récupérer le tally du maillage
tally = sp.get_tally(name='flux_mesh')
flux_data = tally.mean.reshape((500, 500))
flux_data /= source.strength
# Affichage avec échelle logarithmique
plt.imshow(flux_data, 
           origin='lower', 
           extent=[-50, 50, -50, 50], 
           cmap='plasma',
           norm=LogNorm(vmin=1e-6, vmax=flux_data.max()))  # vmin > 0 obligatoire

plt.colorbar(label='Flux [a.u.] (log scale)')
plt.title('Carte de flux XY (échelle log)')
plt.xlabel('X [cm]')
plt.ylabel('Y [cm]')
plt.tight_layout()
plt.show()

### spectre ####
sp = openmc.StatePoint(f"statepoint.{batches_number}.h5")
tally = sp.get_tally(name="pulse-height")
pulse_height_values = tally.get_values(scores=['pulse-height']).flatten()

# Récupération des énergies moyennes par bin (approximation)
energy_bin_centers = energy_bins[1:] + 0.5 * (energy_bins[1] - energy_bins[0])

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
plt.show()

