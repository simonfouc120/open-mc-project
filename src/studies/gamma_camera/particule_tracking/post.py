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

from src.studies.gamma_camera.model.model_gamma_camera import CELL_SIZE, LENGTH_DETECTOR, DETECTOR_THICKNESS
from matplotlib.colors import LogNorm

def num_pixel(x_coord, y_coord) : 
    x_pix = np.int32((x_coord + LENGTH_DETECTOR/2) / (LENGTH_DETECTOR/16))
    y_pix = np.int32((y_coord + LENGTH_DETECTOR/2) / (LENGTH_DETECTOR/16))
    return x_pix, y_pix

def in_camera(x_coord, y_coord, z_coord) : 
    """
    Check if the coordinates are in the camera area
    """
    return -LENGTH_DETECTOR/2 <= y_coord <= LENGTH_DETECTOR/2 and -LENGTH_DETECTOR/2 <= z_coord <= LENGTH_DETECTOR/2 and -DETECTOR_THICKNESS/2 <= x_coord <= DETECTOR_THICKNESS/2


tracks_data = {}

tracks = openmc.Tracks('tracks.h5')
# tracks.plot()
track_in_cdte = tracks.filter(state_filter=lambda s:s['material_id'] == 5)
tracks = track_in_cdte

energy_change_positions = []
energy_loss_values_tot = []
energy_loss_values = []
mult = []

xs = []
ys = []
zs = []
pixels = []  # List to store pixel coordinates
time = []
history_number = []

for i in range(len(tracks)) : 

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
            
            xs.append(pos['x'])
            ys.append(pos['y'])
            zs.append(pos['z'])
            time.append(states['time'][idx])
            # récupère les coordonnées des pixels pour chaque énergie déposée
            pixels.append(num_pixel(ys[-1], zs[-1]))

            # if x_pixel[i]
            mult_value += 1

            # print(f"Changement d'énergie à x={pos['x']}, y={pos['y']}, z={pos['z']} (E={E[idx]})")
            mult.append(len(energy_change_indices))
            energy_loss_values.append(np.abs(E[idx] - E[idx-1]))
        # Add the energy loss value to the list, repeated x times (x = last value of mult)
        for _ in range((mult[-1])):
          # get the number of different pixels from the last values from the array pixels
           
          energy_loss_values_tot.append(energy_loss)
    else:   
        pass
    if i % 10000 == 0:
        print(f"Processing track {i}/{len(tracks)}")

# Convertir les listes en tableaux numpy
energy_change_positions = np.array(energy_change_positions)
xs = np.array(xs)
ys = np.array(ys)
zs = np.array(zs)
mult = np.array(mult)
history_number = np.array(history_number)
energy_loss_values = np.array(energy_loss_values)
energy_loss_values_tot = np.array(energy_loss_values_tot)
pixels_array = np.array(pixels)
time = np.array(time)

mask_mult_1 = (mult == 1)
mask_mult_2 = (mult == 2)
mask_mult_3 = (mult == 3)

# non_null_energy_loss = energy_loss_values[en ergy_loss_values != 0]
energy_loss_values_tot_mult_1 = energy_loss_values_tot[mask_mult_1]
energy_loss_values_tot_mult_2 = energy_loss_values_tot[mask_mult_2]
energy_loss_values_tot_mult_3 = energy_loss_values_tot[mask_mult_3]
# prendre une valeur sur deux de energy_loss_values_mult_2
energy_loss_values_tot_mult_2 = energy_loss_values_tot_mult_2[::2]
energy_loss_values_tot_mult_3 = energy_loss_values_tot_mult_3[::3]
# plot the energy change positions
energy_loss_mult_2 = energy_loss_values[mult == 2]

# Efficiently extract first and second energy losses for multiplicity 2 events

energy_loss_mult_2 = energy_loss_values[mask_mult_2]

time_mult_2 = time[mask_mult_2]
time_mult_1 = time[mask_mult_1]

# Reshape to (N, 2) where N is the number of multiplicity 2 events
energy_pairs = energy_loss_mult_2.reshape(-1, 2)
first_energy = energy_pairs[:, 0]
second_energy = energy_pairs[:, 1]

# plot the first energt vs second energy in a 2D histogram with 50, 50 bins
plt.figure(figsize=(8, 6))
plt.hist2d(first_energy/1e6, second_energy/1e6, bins=(50, 50), cmap='Blues', norm=LogNorm())
plt.colorbar(label='Number of energy changes')
plt.xlabel('First energy loss [MeV]')
plt.ylabel('Second energy loss [MeV]')
plt.title('Distribution of energy losses for multiplicity 2 events')
plt.xlim(0, 0.7)
plt.ylim(0, 0.7)
plt.grid(axis='both', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("first_second_energy_mult_2.png")
plt.show()

PIXELS_X = pixels_array[:, 0]
PIXELS_Y = pixels_array[:, 1]

plt.figure(figsize=(8, 6))
plt.hist2d(PIXELS_X, PIXELS_Y, bins=(16, 16), cmap='Blues')
plt.colorbar(label='Number of energy changes')
plt.xlabel('Pixel X')
plt.ylabel('Pixel Y')
plt.title('Distribution of energy changes in pixels')
plt.savefig("histo_pixel.png")
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(energy_loss_values_tot/1e6, bins=100, color='mediumseagreen', edgecolor='black', alpha=0.8)
plt.yscale("log")
plt.xlabel("Energy [MeV]")
plt.ylabel("Number of events")
plt.title("Energy spectrum")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("spectrum_energy.png")
plt.show()

plt.figure(figsize=(8, 6))
bins = np.linspace(0, 1, 101)  # 50 bins between 0 and 1 MeV
plt.hist(energy_loss_values_tot_mult_1/1e6, bins=bins, color='mediumseagreen', edgecolor='black', alpha=0.8, label='Multiplicity 1', histtype='stepfilled')
plt.hist(energy_loss_values_tot_mult_2/1e6, bins=bins, color='royalblue', edgecolor='black', alpha=0.6, label='Multiplicity 2', histtype='stepfilled')
plt.hist(energy_loss_values_tot_mult_3/1e6, bins=bins, color='orange', edgecolor='black', alpha=0.5, label='Multiplicity 3', histtype='stepfilled')
plt.yscale("log")
plt.xlabel("Energy [MeV]")
plt.ylabel("Number of events")
plt.title("Energy spectra by multiplicity")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend()
plt.xlim(-0.05, 0.8)
plt.tight_layout()
plt.savefig("spectrum_energy_multiplicity.png")
plt.show()

time_diff_mult_2 = time_mult_2[1::2] - time_mult_2[::2]

fig, axs = plt.subplots(2, 1, figsize=(8, 10))

# First subplot: histogram of time_mult_2
axs[0].hist(time_mult_2*1E9, bins=100, color='royalblue', edgecolor='black', alpha=0.8, label='Multiplicity 2')
axs[0].set_xlabel("Time [ns]")
axs[0].set_ylabel("Number of events")
axs[0].set_title("Time distribution for multiplicity 2")
axs[0].legend()

# Second subplot: histogram of time differences
axs[1].hist(time_diff_mult_2*1E9, bins=100, color='mediumseagreen', edgecolor='black', alpha=0.8)
axs[1].set_xlabel("Time difference [ns]")
axs[1].set_ylabel("Number of events")
axs[1].set_title("Distribution of time differences between the first and second interaction\nfor multiplicity 2")

plt.tight_layout()
plt.show()


# TODO : faire filtre pour éviter anneau
# TODO : faire spectre simple 
# TODO : faire spectre par pixel
# TODO : faire spectre des doubles

# TODO : faire spectre des triples

# TODO : faire note book avec les résultats


