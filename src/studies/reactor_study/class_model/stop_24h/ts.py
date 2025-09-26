"""
Script OpenMC — Terme source approximatif pour un SMR 60 MWth, 24 h après arrêt
Hypotheses (modifiable):
 - Puissance nominale : 60 MWth
 - Fraction de la puissance de décroissance à 24 h : 0.4 %
 - Fraction de la décroissance emissée en photons : 20 %
 - Energie photon moyenne modelisee par une loi exponentielle (mean = 1.0 MeV)

Le script construit :
 - un spectre gamma tabulé (MeV) utilisable par openmc.stats.Tabular
 - une Source (photon) ponctuelle au centre du repère
 - un fichier settings.xml prêt à lancer en mode 'fixed source'

Important : OpenMC echantillonne des "particules statistiques" (n particles x batches).
Pour obtenir des grandeurs physiques (photons/s) il faut multiplier les tallies par
un facteur de normalisation :
    facteur = photons_par_seconde_réel / N_particles_echantillonnées_total
avec N_particles_echantillonnées_total = settings.batches * settings.particles

Exemple de calcul dans ce script.
"""

import numpy as np
import openmc

# -----------------------
# Parametres physiques
# -----------------------
P_th = 50e6                 # 60 MWth en W
decay_frac_24h = 0.004      # 0.4 % a 24 h
gamma_frac = 0.20           # 20 % de la puissance de decroissance en gamma
mean_gamma_energy_MeV = 1.0 # energie moyenne (MeV) pour le modele exponentiel

# Calculs simples
P_decay_24h = P_th * decay_frac_24h           # W
P_gamma = P_decay_24h * gamma_frac            # W

# convertir MeV -> J
eV_to_J = 1.602176634e-19
MeV_to_J = 1e6 * eV_to_J

# photons par seconde approximatif (si energie moyenne = 1 MeV)
photons_per_s = P_gamma / (mean_gamma_energy_MeV * MeV_to_J)

# Arrondir pour lecture
print(f"Hypotheses : P_th={P_th/1e6:.0f} MWth, decay_frac_24h={decay_frac_24h*100:.2f} %, gamma_frac={gamma_frac*100:.1f}%")
print(f"Puissance de decroissance 24 h: {P_decay_24h/1e3:.1f} kW ; P_gamma = {P_gamma/1e3:.1f} kW")
print(f"Photons/s (estimation, E_mean={mean_gamma_energy_MeV} MeV) = {photons_per_s:.3e}")

# -----------------------
# Spectre gamma approche (loi exponentielle troncée) -> Tabular
# energies en MeV (bornes 0.1 - 10 MeV)
# densité f(E) = C * exp(-E/mean)
# -----------------------
E = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0])  # MeV
pdf_unnorm = np.exp(-E / mean_gamma_energy_MeV)
# PDF discrétisée associée aux noeuds (approche simple par trapezes sur les intervalles)
# On va construire une PDF continue par interpolation linéaire via openmc.stats.Tabular:
# tabular demande les x et la pdf normalisee sur ces x.

# Calcul d'un pdf normalise aux points E (approximation):
# on calcule l'aire sous la courbe exp(-E/mean) entre les points pour obtenir la "prob" associee
areas = []
for i in range(len(E)-1):
    a = mean_gamma_energy_MeV * (np.exp(-E[i]/mean_gamma_energy_MeV) - np.exp(-E[i+1]/mean_gamma_energy_MeV))
    areas.append(a)
# attribuer la probabilite aux noeuds (on met la moyenne des aires incidentes sur le noeud)
node_prob = np.zeros_like(E)
node_prob[0] = areas[0]/2
node_prob[-1] = areas[-1]/2
for i in range(1, len(E)-1):
    node_prob[i] = 0.5*(areas[i-1] + areas[i])

# Normaliser
pdf = node_prob / np.sum(node_prob)

# Verification rapide
assert np.isclose(np.sum(pdf), 1.0, atol=1e-8)

# -----------------------
# Construire la Source OpenMC
# -----------------------
src = openmc.Source()
# Source ponctuelle au centre du coeur (modifiable) :
src.space = openmc.stats.Point((0.0, 0.0, 0.0))
# isotropie
src.angle = openmc.stats.Isotropic()
# energie (tabular) : openmc s'attend a energy (x) et pdf (y) -> x en eV
energies_eV = (E * 1e6).tolist()          # convertir MeV -> eV
pdf_list = pdf.tolist()
src.energy = openmc.stats.Tabular(energies_eV, pdf_list, interpolation='linear-linear')
src.particle = 'photon'

# Ecrire dans settings
settings = openmc.Settings()
settings.source = src
settings.run_mode = 'fixed source'
settings.batches = 100
settings.particles = 10000
# activer le transport photonique (si build d'OpenMC avec photon transport)
settings.photon_transport = True
# ecrire les fichiers xml
settings.export_to_xml()

# -----------------------
# Indications pour la normalisation des tallies
# -----------------------
N_total_sampled = settings.batches * settings.particles
normalisation_factor = photons_per_s / N_total_sampled

print('\nFichiers settings.xml et source.xml ecrits.')
print(f"Parametres de run: batches={settings.batches}, particles={settings.particles}")
print(f"Total d'echantillons sources = {N_total_sampled:,d}")
print(f"Facteur de normalisation (photons/s par particule echantillonee) = {normalisation_factor:.3e}")
print('\nApres la simulation, multiplie les tallies obtenus par ce facteur pour exprimer des resultats en photons/s ou en puissances reales.')

# Optionnel : sauvegarder le spectre de sortie (pour reference)
with open('gamma_spectrum_tabular.txt', 'w') as f:
    f.write('# Energy_MeV \t PDF\n')
    for e,p in zip(E, pdf):
        f.write(f"{e:.6f}\t{p:.8e}\n")

print('\nSpectre tabule sauvegarde dans gamma_spectrum_tabular.txt')

# Fin du script
