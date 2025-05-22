import openmc
import os 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

os.environ["OPENMC_CROSS_SECTIONS"] = "/Users/simonfoucambert/endfb-viii.0-hdf5/cross_sections.xml"

# Création des matériaux
cs137 = openmc.Material(name="Cs137")
cs137.add_nuclide("Cs137", 1.0)
cs137.set_density("g/cm3", 4.0)

csi = openmc.Material(name="CsI")
csi.add_element("Cs", 1.0)
csi.add_element("I", 1.0)
csi.set_density("g/cm3", 4.51)

materials = openmc.Materials([cs137, csi])

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
void_cell = openmc.Cell(name="void_cell", region=outer_boundary_cell)  # vide entre les deux

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


# Configuration de la simulation
settings = openmc.Settings()
settings.batches = 5
settings.particles = 1000000
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

