import numpy as np

# BORNES D’ÉNERGIE DES BINS (21 valeurs pour 20 bins)
energy_bins = np.array([
    0.0100, 0.0158, 0.0250, 0.0398, 0.0631,
    0.1000, 0.1580, 0.2510, 0.3980, 0.6310,
    1.0000, 1.5800, 2.5100, 3.9800, 6.3100,
    10.0000, 12.5900, 15.8500, 17.7800, 19.9500, 22.3900
])  # en MeV

# FACTEURS DE CONVERSION (1 par bin)
conversion_factors = np.array([
    1.13e-10, 2.07e-10, 3.78e-10, 6.87e-10, 1.17e-09,
    1.93e-09, 2.88e-09, 4.09e-09, 5.43e-09, 6.66e-09,
    7.60e-09, 8.23e-09, 8.53e-09, 8.50e-09, 8.31e-09,
    8.16e-09, 8.05e-09, 7.97e-09, 7.91e-09, 7.86e-09
])  # en Sv·cm²/photon