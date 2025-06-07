import h5py
import numpy as np
import openmc
import matplotlib.pyplot as plt
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

def in_camera(x_coord, y_coord) : 
    """
    Check if the coordinates are in the camera area
    """
    return -LENGTH_DETECTOR/2 <= x_coord <= LENGTH_DETECTOR/2 and -LENGTH_DETECTOR/2 <= y_coord <= LENGTH_DETECTOR/2


tracks_data = {}

tracks = openmc.Tracks('tracks.h5')
# tracks.plot()
# track_in_cdte = tracks.filter(state_filter=lambda s:s['material_id'] == 5)

energy_change_positions = []
energy_loss_values = []
mult = []

xs = []
ys = []
zs = []
history_number = []

for i in range(len(tracks)) : 
    if i % 10000 == 0:
        print(f"Processing track {i}/{len(tracks)}")
    states = tracks[i].particle_tracks[0].states
    # On cherche les indices où l'énergie change
    E = states['E']
    energy_change_indices = np.where(np.diff(E) != 0)[0] + 1 # car diff décale d'un cran
    energy_loss = 0.0
    mult_value = 0
    if len(energy_change_indices) != 0:
        for idx in energy_change_indices:
            history_number.append(i)

            pos = states['r'][idx]
            # save in an arrat the position of the energy change
            # get the energy loss in the cell 
            energy_loss += np.abs(E[idx] - E[idx-1])
            mult_value += 1
            xs.append(pos['x'])
            ys.append(pos['y'])
            zs.append(pos['z'])
            # print(f"Changement d'énergie à x={pos['x']}, y={pos['y']}, z={pos['z']} (E={E[idx]})")
            mult.append(len(energy_change_indices))
        # Add the energy loss value to the list, repeated x times (x = last value of mult)
        for _ in range((mult[-1])):
          energy_loss_values.append(energy_loss)
    else:   
        pass

# Convertir les listes en tableaux numpy
energy_change_positions = np.array(energy_change_positions)
xs = np.array(xs)
ys = np.array(ys)
zs = np.array(zs)
mult = np.array(mult)
history_number = np.array(history_number)

energy_loss_values = np.array(energy_loss_values)
# non_null_energy_loss = energy_loss_values[energy_loss_values != 0]
energy_loss_values_mult_1 = energy_loss_values[mult == 1]
energy_loss_values_mult_2 = energy_loss_values[mult == 2]

# prendre une valeur sur deux de energy_loss_values_mult_2
energy_loss_values_mult_2 = energy_loss_values_mult_2[::2]

# plot the energy change positions

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs, s=1)
ax.set_xlabel('x (cm)')
ax.set_ylabel('y (cm)')
ax.set_zlabel('z (cm)')
ax.set_title('Positions des changements d\'énergie')
plt.show()

PIXELS_X = num_pixel(ys, zs)[0]
PIXELS_Y = num_pixel(ys, zs)[1]

# Afficher les pixels sous la forme d'un histogramme 2D
plt.figure(figsize=(8, 6))
plt.hist2d(PIXELS_X, PIXELS_Y, bins=(16, 16), cmap='Blues')
plt.colorbar(label='Nombre de changements d\'énergie')
plt.xlabel('Pixel X')
plt.ylabel('Pixel Y')
plt.title('Distribution des changements d\'énergie dans les pixels')
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(energy_loss_values/1e6, bins=45, color='mediumseagreen', edgecolor='black', alpha=0.8)
plt.yscale("log")
plt.xlabel("Perte d'énergie [MeV]")
plt.ylabel("Nombre d'événements")
plt.title("Distribution des pertes d'énergie par événement")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(energy_loss_values_mult_1/1e6, bins=45, color='mediumseagreen', edgecolor='black', alpha=0.8)
plt.yscale("log")
plt.xlabel("Perte d'énergie [MeV]")
plt.ylabel("Nombre d'événements")
plt.title("Distribution des pertes d'énergie par événement")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(energy_loss_values_mult_2/1e6, bins=45, color='mediumseagreen', edgecolor='black', alpha=0.8)
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


