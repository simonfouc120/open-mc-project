import openmc
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys 
from PIL import Image
import numpy as np

CWD = Path(__file__).parent.resolve()

project_root = Path(__file__).resolve().parents[3]  
sys.path.append(str(project_root))
from parameters.parameters_paths import NUCLIDE_LARA_PATH

class Isotope:
    def __init__(self, name):
        self.name = name
        self.decay_constant = openmc.data.decay_constant(isotope=name)
        self.half_life = openmc.data.half_life(isotope=name)
        self.atomic_mass = openmc.data.atomic_mass(isotope=name)

    @property
    def massic_activity(self):
        return (6.022e23 * np.log(2)) / (self.atomic_mass * self.half_life)

    def activity(self, mass):
        return self.massic_activity * mass

    def mass(self, activity):
        return activity / self.massic_activity
    
    def activity_over_time(self, mass, time):
        return self.activity(mass) * np.exp(-self.decay_constant * time)
    
class Radionuclide_lara:
    """Class to handle radionuclide data from LARA files.
    Format of nuclide LARA files:
    - Each file is named after the radionuclide (e.g., 'Co-60.lara.txt').
    - The file contains a header with metadata and a section for emissions.
    - The emissions section lists energy levels, intensities, and types of emissions.
    """
    def __init__(self, name):
        self.name = name

    @property
    def radionuclide_data(self):
        file_path = NUCLIDE_LARA_PATH + self.name + ".lara.txt"
        rn_data = {}
        emissions = []
        with open(file_path, encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            section = None
            for line in lines:
                if line.startswith('Emissions') or line.startswith('Energy (keV)'):
                    section = 'emissions'
                    continue
                if section == 'emissions':
                    if line.startswith('=') or line.startswith('-'):
                        continue
                    # Parse emission lines
                    parts = [p.strip() for p in line.split(';')]
                    if len(parts) >= 8:
                        emissions.append({
                            'Energy (keV)': parts[0],
                            'Ener. unc. (keV)': parts[1],
                            'Intensity (%)': parts[2],
                            'Int. unc. (%)': parts[3],
                            'Type': parts[4],
                            'Origin': parts[5],
                            'Lvl. start': parts[6],
                            'Lvl. end': parts[7],
                        })
                else:
                    if ';' in line:
                        key, *values = [p.strip() for p in line.split(';')]
                        if len(values) == 1:
                            rn_data[key] = values[0]
                        else:
                            rn_data[key] = values
        rn_data['Emissions'] = emissions
        return rn_data
    
    def get_rays_emission_data(self, energy_filter=None, intensity_filter=None, photon_only=False):
        rays_energies =  np.array([float(em['Energy (keV)']) for em in self.radionuclide_data["Emissions"]])
        rays_intensities = np.array([float(em['Intensity (%)']) for em in self.radionuclide_data["Emissions"]])
        rays_types = np.array([em['Type'] for em in self.radionuclide_data["Emissions"]])

        if energy_filter is not None:
            mask = rays_energies >= energy_filter
            rays_energies = rays_energies[mask]
            rays_intensities = rays_intensities[mask]
            rays_types = rays_types[mask]

        if intensity_filter is not None:
            mask = rays_intensities >= intensity_filter
            rays_energies = rays_energies[mask]
            rays_intensities = rays_intensities[mask]
            rays_types = rays_types[mask]

        if photon_only:
            mask = np.isin(rays_types, ['g']) | np.char.startswith(rays_types, 'X')
            rays_energies = rays_energies[mask]
            rays_intensities = rays_intensities[mask]
            rays_types = rays_types[mask]
        return rays_energies, rays_intensities, rays_types

    def plot_emissions(self, saving_figure:bool=False, photon_only:bool=False):
        rays_energies, rays_intensities, rays_types = self.get_rays_emission_data(photon_only=photon_only)

        gamma_rays_energies = rays_energies[rays_types == 'g']
        gamma_rays_intensities = rays_intensities[rays_types == 'g']
        x_rays_energies = rays_energies[np.char.startswith(rays_types, 'X')]
        x_rays_intensities = rays_intensities[np.char.startswith(rays_types, 'X')]

        alpha_rays_energies = rays_energies[rays_types == 'a']
        alpha_rays_intensities = rays_intensities[rays_types == 'a']

        if photon_only:
            max_energy = max(
            max(gamma_rays_energies) if len(gamma_rays_energies) > 0 else 0,
            max(x_rays_energies) if len(x_rays_energies) > 0 else 0)
            min_energy = min(
            min(gamma_rays_energies) if len(gamma_rays_energies) > 0 else 0,
            min(x_rays_energies) if len(x_rays_energies) > 0 else 0)
        else:
            max_energy = max(
                max(gamma_rays_energies) if len(gamma_rays_energies) > 0 else 0,
                max(x_rays_energies) if len(x_rays_energies) > 0 else 0,
                max(alpha_rays_energies) if len(alpha_rays_energies) > 0 else 0)
            min_energy = min(
                min(gamma_rays_energies) if len(gamma_rays_energies) > 0 else 0,
                min(x_rays_energies) if len(x_rays_energies) > 0 else 0,
                min(alpha_rays_energies) if len(alpha_rays_energies) > 0 else 0)

        delta_energy = max_energy - min_energy
        width = delta_energy // 110 if delta_energy > 0 else 1
        if len(gamma_rays_energies) != 0:
            plt.bar(gamma_rays_energies, gamma_rays_intensities, width=width, label ='Gamma Emissions', color='blue', alpha=1)
        if len(x_rays_energies) != 0:
            plt.bar(x_rays_energies, x_rays_intensities, width=width, label ='X Emissions', color='orange', alpha=1)
        if len(alpha_rays_energies) != 0:
            plt.bar(alpha_rays_energies, alpha_rays_intensities, width=width, label ='Alpha Emissions', color='green', alpha=1)
        plt.xlabel('Energy [keV]')
        plt.ylabel('Intensity [%]')
        plt.title(f'Emissions of {self.name}')
        plt.yscale('log')
        plt.grid()
        plt.legend()
        plt.savefig(f"{self.name}_emissions.png") if saving_figure else None
        plt.show()