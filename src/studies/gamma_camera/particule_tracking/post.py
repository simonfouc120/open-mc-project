import h5py
import numpy as np
import openmc
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sys
CWD = Path.cwd().resolve()
project_root = Path.cwd().parents[3]
sys.path.append(str(project_root))

from src.studies.gamma_camera.model.model_gamma_camera import CELL_SIZE, LENGTH_DETECTOR

def num_pixel(x_coord, y_coord) : 
    x_pix = np.int32((x_coord + LENGTH_DETECTOR/2) / (LENGTH_DETECTOR/16))
    y_pix = np.int32((y_coord + LENGTH_DETECTOR/2) / (LENGTH_DETECTOR/16))
    return x_pix, y_pix

tracks_data = {}

tracks = openmc.Tracks('tracks.h5')
# tracks.plot()
# track_in_cdte = tracks.filter(state_filter=lambda s:s['material_id'] == 5)

energy_change_positions = []
energy_loss_values = []
for i in range(len(tracks)) : 
    states = tracks[i].particle_tracks[0].states
    # On cherche les indices où l'énergie change
    E = states['E']
    energy_change_indices = np.where(np.diff(E) != 0)[0] + 1 # car diff décale d'un cran
    energy_loss = 0.0
    # if True in (np.diff(E) != 0):
    #     print(np.diff(E) != 0)
    #     print(states)
    for idx in energy_change_indices:
        pos = states['r'][idx]
        # save in an arrat the position of the energy change
        # get the energy loss in the cell 
        energy_loss += np.abs(E[idx] - E[idx-1])
        energy_change_positions.append(pos)
        # print(f"Changement d'énergie à x={pos['x']}, y={pos['y']}, z={pos['z']} (E={E[idx]})")
    energy_loss_values.append(energy_loss)

energy_loss_values = np.array(energy_loss_values)
non_null_energy_loss = energy_loss_values[energy_loss_values != 0]


# plot the energy change positions
import matplotlib.pyplot as plt
# Extraire x, y, z
xs = [pos['x'] for pos in energy_change_positions]
ys = [pos['y'] for pos in energy_change_positions]
zs = [pos['z'] for pos in energy_change_positions]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs, s=1)
ax.set_xlabel('x (cm)')
ax.set_ylabel('y (cm)')
ax.set_zlabel('z (cm)')
ax.set_title('Positions des changements d\'énergie')
plt.show()

PIXELS_X = [num_pixel(pos['y'], pos['z'])[0] for pos in energy_change_positions]
PIXELS_Y = [num_pixel(pos['y'], pos['z'])[1] for pos in energy_change_positions]

# Afficher les pixels sous la forme d'un histogramme 2D
plt.figure(figsize=(8, 6))
plt.hist2d(PIXELS_X, PIXELS_Y, bins=(16, 16), cmap='Blues')
plt.colorbar(label='Nombre de changements d\'énergie')
plt.xlabel('Pixel X')
plt.ylabel('Pixel Y')
plt.title('Distribution des changements d\'énergie dans les pixels')
plt.grid(False)
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(non_null_energy_loss/1e6, bins=45, color='mediumseagreen', edgecolor='black', alpha=0.8)
plt.yscale("log")
plt.xlabel("Perte d'énergie [MeV]")
plt.ylabel("Nombre d'événements")
plt.title("Distribution des pertes d'énergie par événement")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# TODO : faire filtre pour éviter anneau
# TODO : faire spectre simple 
# TODO : faire spectre par pixel
# TODO : faire spectre des doubles

# TODO : faire spectre des triples

# TODO : faire note book avec les résultats