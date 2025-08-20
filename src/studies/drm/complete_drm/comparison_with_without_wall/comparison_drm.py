import numpy as np


import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from pathlib import Path
 
cwd = Path(__file__).parent.resolve().parents[0]

drm_wall = np.load(cwd / "with_wall" / "drm_spectrum_wall.npz")
drm_without_wall = np.load(cwd / "without_wall" / "drm_spectrum.npz")

energy = drm_wall['energy']
energy_bins = drm_wall['energy_bins']
spectrums = drm_wall['spectrums']
spectrum_std = drm_wall['spectrum_std']

energy_without_wall = drm_without_wall['energy']
energy_bins_without_wall = drm_without_wall['energy_bins']
spectrums_without_wall = drm_without_wall['spectrums']
spectrum_std_without_wall = drm_without_wall['spectrum_std']

difference = spectrums - spectrums_without_wall

X, Y = np.meshgrid(energy, energy_bins[:-1])

plt.figure()
plt.pcolormesh(X, Y, difference, cmap='viridis', norm=LogNorm(), shading='auto')
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
plt.savefig("drm_spectrum_difference.png")
plt.show()