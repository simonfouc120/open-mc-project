import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Paramètres de la grille
nx, ny, nz = 25, 25, 25  # grille 2D pour visualisation (z=1)
lower_left = np.array([-500.0, -500.0, -500])
upper_right = np.array([500.0, 500.0, 500])
target = np.array([50.0, 0.0, 0.0])  # position de la cible

x_vals = np.linspace(lower_left[0], upper_right[0], nx)
y_vals = np.linspace(lower_left[1], upper_right[1], ny)
z_vals = np.linspace(lower_left[2], upper_right[2], nz)

importance_map = np.zeros((nx, ny, nz))

# Calcul de l'importance : inverse de la distance à la cible
for i, x in enumerate(x_vals):
    for j, y in enumerate(y_vals):
        for k, z in enumerate(z_vals):
            pos = np.array([x, y, z])
            dist = np.linalg.norm(pos - target)
            importance_map[i, j, k] = ((dist + 50.0))/1e5

# Affichage avec matplotlib et échelle logarithmique
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(
    importance_map[:,:,12].T,  # transpose pour correspondance x/y
    origin='lower',
    extent=(lower_left[0], upper_right[0], lower_left[1], upper_right[1]),
    cmap='plasma',
    norm=LogNorm(vmin=importance_map[importance_map > 0].min(), vmax=importance_map.max())
)
ax.set_title(f"Carte d'importance centrée sur la cible ({target[0]}, {target[1]}, {target[2]}) cm")
ax.set_xlabel("x [cm]")
ax.set_ylabel("y [cm]")
fig.colorbar(im, ax=ax, label="Importance (distance)")
plt.tight_layout()
plt.show()
