import openmc
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

csi = openmc.Material(name="CsI")
csi.add_element("Cs", 1.0)
csi.add_element("I", 1.0)
csi.set_density("g/cm3", 4.51)

materials = openmc.Materials([cs137, csi])

air = openmc.Material(name="Air")
air.add_element("N", 0.78)
air.add_element("O", 0.21)
air.add_element("Ar", 0.01)
air.set_density("g/cm3", 0.001225)  

materials.append(air)

sphere = openmc.Sphere(r=1.0, surface_id=1)
detector = openmc.Sphere(y0=25., r=6.0, surface_id=2)
outer_boundary = openmc.Sphere(r=100.0, surface_id=3, boundary_type='vacuum')  # Limite du monde

source_cell = openmc.Cell(name="source_cell")
source_cell.fill = cs137
source_cell.region = -sphere

detector_cell = openmc.Cell(name="detector_cell")
detector_cell.fill = csi
detector_cell.region = -detector & +sphere

outer_boundary_cell = -outer_boundary & +detector
void_cell = openmc.Cell(name="air_cell", fill=air, region=outer_boundary_cell)

universe = openmc.Universe(cells=[source_cell, detector_cell, void_cell])
geometry = openmc.Geometry(universe)

# Création de la source
source = openmc.Source()
source.space = openmc.stats.Point((0, 0, 0))
source.energy = openmc.stats.Discrete([0.662], [1.0])  # Énergie du photon de 662 keV pour Cs137
source.particle = "photon"

# Création des tallies
tally = openmc.Tally(name="detector_tally")
tally.scores = ["flux"]
tally.filters = [openmc.CellFilter(detector_cell)]

tallies = openmc.Tallies([tally])

mesh = openmc.RegularMesh()

mesh.dimension = [500, 500, 1]  # XY
mesh.lower_left = [-50.0, -50.0, -1.0]
mesh.upper_right = [50.0, 50.0, 1.0]

mesh_filter = openmc.MeshFilter(mesh)
mesh_tally = openmc.Tally(name='flux_mesh')
mesh_tally.filters = [mesh_filter]
mesh_tally.scores = ['flux']

tallies.append(mesh_tally)

energy_bins = np.linspace(1e-3, 1.0, 500)  # de 1 keV à 2 MeV en 500 bins
energy_filter = openmc.EnergyFilter(energy_bins)

# Tally pour le spectre d'énergie déposée dans le détecteur
energy_dep_tally = openmc.Tally(name="energy_deposition_spectrum")
energy_dep_tally.filters = [openmc.CellFilter(detector_cell), energy_filter]
energy_dep_tally.scores = ["heating"]

tallies.append(energy_dep_tally)


# Configuration de la simulation
settings = openmc.Settings()
settings.batches = 5
settings.particles = 10000000
settings.source = source
settings.run_mode = "fixed source"

# Export des fichiers nécessaires pour la simulation
materials.export_to_xml()
geometry.export_to_xml()
settings.export_to_xml()
tallies.export_to_xml()

# Exécution de la simulation
if os.path.exists("summary.h5"):
    os.remove("summary.h5")

if os.path.exists("statepoint.5.h5"):
    os.remove("statepoint.5.h5")

openmc.run()

print("Calcul fini")


sp = openmc.StatePoint("statepoint.5.h5")
tally = sp.get_tally(name="detector_tally")
flux_mean = tally.mean.flatten()
flux_std_dev = tally.std_dev.flatten()

print("Flux moyen :", flux_mean[0])
print("Écart-type :", flux_std_dev[0])

# Charger le fichier de sortie
sp = openmc.StatePoint('statepoint.5.h5')


### mesh tallty ####
# Récupérer le tally du maillage
tally = sp.get_tally(name='flux_mesh')
flux_data = tally.mean.reshape((500, 500))

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

sp = openmc.StatePoint("statepoint.5.h5")
tally = sp.get_tally(name="energy_deposition_spectrum")

# Récupération des énergies moyennes par bin (approximation)
energy_filter = tally.filters[1]
energy_bins = energy_filter.values
bin_centers = 0.5 * (np.array(energy_bins[:-1]) + np.array(energy_bins[1:]))

# Moyenne et écart-type de l'énergie déposée
spectrum = tally.mean.flatten()
spectrum_std = tally.std_dev.flatten()

# Tracé
plt.figure()
plt.plot(bin_centers, spectrum, drawstyle='steps-mid', label='Énergie déposée')
plt.fill_between(bin_centers, spectrum - spectrum_std, spectrum + spectrum_std, alpha=0.3)
plt.xlabel("Énergie [MeV]")
plt.ylabel("Énergie déposée [a.u.]")
plt.title("Spectre d'énergie déposée dans le détecteur")
plt.grid(True)
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.show()
