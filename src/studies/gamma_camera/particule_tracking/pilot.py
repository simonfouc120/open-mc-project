import openmc
from pathlib import Path
import sys
import os 
import numpy as np

CWD = Path.cwd().resolve()
project_root = Path.cwd().parents[3]
sys.path.append(str(project_root))

from src.studies.gamma_camera.model.model_gamma_camera import MATERIAL, GEOMETRY, PIXEL_CELLS
from src.utils.pre_processing.pre_processing import mesh_tally_yz

from parameters.parameters_paths import PATH_TO_CROSS_SECTIONS
os.environ["OPENMC_CROSS_SECTIONS"] = PATH_TO_CROSS_SECTIONS

GEOMETRY.export_to_xml()
MATERIAL.export_to_xml()


pixel_cells = PIXEL_CELLS

# Define a 137 Cs source 
source = openmc.Source()
source.space = openmc.stats.Point((3, 0, 0))  # Point source at the origin
source.particle = 'photon'  # Source emits photons
source.angle = openmc.stats.Isotropic()  # Isotropic emission
source.energy = openmc.stats.Discrete([661_700], [1.0])  # Énergie du photon de 662 keV pour Cs137
source_strength = 1e6  # Source strength in photons per second
source.strength = source_strength  # Set the source strength
source.angle = openmc.stats.PolarAzimuthal(
    mu=openmc.stats.Uniform(a=np.cos(np.radians(30)), b=1.0),  # cône de 30°
    phi=openmc.stats.Uniform(0.0, 2*np.pi)
)
source.angle.reference_uvw = (-1.0, 0.0, 0.0)  # orienté vers -x

settings = openmc.Settings()
batches_number = 40
settings.batches = batches_number
settings.particles = 10**5
settings.source = source
settings.photon_transport = True 
settings.run_mode = "fixed source"
settings.verbose = True
settings.max_tracks = 1000000
settings.export_to_xml()

openmc.run(tracks=True)

