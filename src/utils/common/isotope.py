import openmc
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys 
from PIL import Image
import pandas as pd
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
    
    @property
    def get_half_life(self):
        """
        Returns:
            float: Half-life in seconds
        """
        half_life_str = self.radionuclide_data.get('Half-life (s)', None)[0]
        if half_life_str is not None:
            return float(half_life_str)
        else:
            raise ValueError(f"Half-life data not found for {self.name}")
    
    @property
    def get_decay_constant(self):
        """
        Returns:
            the decay constant of the radionuclide in s^-1.
            The decay constant is calculated using the formula:
        """
        half_life_seconds = self.get_half_life
        return np.log(2) / half_life_seconds if half_life_seconds > 0 else 0.0

    @property
    def get_massic_activity(self):
        """
        Returns:
            the massic activity of the radionuclide in Bq/g.
            The massic activity is calculated using the formula:
            massic_activity = (6.022e23 * decay_constant) / atomic_mass
        """
        massic_activity = self.radionuclide_data.get('Specific activity (Bq/g)')[0]
        return massic_activity if massic_activity is not None else 0.0

    def get_massic_activity_after_time(self, time: float = 0.0):
        """
        Args:
            time (float): Time in seconds.
        Returns:
            The massic activity of the radionuclide after a given time in Bq/g.
            The massic activity is calculated using the formula:
            massic_activity = massic_activity_0 * exp(-decay_constant * time)
        """
        massic_activity_0 = float(self.get_massic_activity)
        decay_constant = self.get_decay_constant
        return massic_activity_0 * np.exp(-decay_constant * time) if massic_activity_0 > 0 else 0.0

    def get_activity(self, mass:float=1.0):
        """
        Args : 
            mass (float): Mass of the radionuclide in grams.
        Returns :
            the activity of the radionuclide in Bq.
            The activity is calculated using the formula:
            activity = mass * massic_activity
        """
        massic_activity = float(self.get_massic_activity)
        return mass * massic_activity if massic_activity > 0 else 0.0

    def get_mass(self, activity: float = 1.0):
        """
        Args:
            activity (float): Activity of the radionuclide in Bq.
        Returns:
            the mass of the radionuclide in grams.
            The mass is calculated using the formula:
            mass = activity / massic_activity
        """
        massic_activity = float(self.get_massic_activity)
        return activity / massic_activity if massic_activity > 0 else 0.0

    def get_activity_after_time(self, mass:float=1.0, time:float=0.0):
        """
        Args:
            mass (float): Mass of the radionuclide in grams.
            time (float): Time in seconds.
        Returns:
            the activity of the radionuclide after a given time in Bq.
            The activity is calculated using the formula:
            activity = mass * massic_activity * exp(-decay_constant * time)
        """
        massic_activity = self.get_massic_activity_after_time(time)
        return float(mass * massic_activity) if massic_activity > 0 else 0.0

    def get_rays_emission_data(self, 
                               energy_filter=None, 
                               intensity_filter=None, 
                               photon_only=False) -> tuple:
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
        rays_intensities = rays_intensities / 100 if rays_intensities.sum() > 0 else rays_intensities
        return rays_energies, rays_intensities, rays_types

    def plot_emissions(self, saving_figure:bool=False, photon_only:bool=False, log_scale:bool=True):
        rays_energies, rays_intensities, rays_types = self.get_rays_emission_data(photon_only=photon_only)
        rays_intensities = rays_intensities * 100
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
        if log_scale:
            plt.yscale('log')
        plt.grid()
        plt.legend()
        plt.savefig(f"{self.name}_emissions.png") if saving_figure else None
        plt.show()
    
    def __str__(self):
        return f"Radionuclide: {self.name}, Half-life: {self.get_half_life} s, Massic Activity: {self.get_massic_activity} Bq/g"
    
    def __repr__(self):
        return self.__str__()


UNIT_FACTORS = {
            "s": 1,
            "min": 60,
            "h": 3600,
            "d": 3600 * 24,
            "w": 3600 * 24 * 7,
            "mo": 3600 * 24 * 30
        }
class Radionuclide_list:

    def __init__(self, 
                 file: str = "rn.csv", 
                 given_information:str = "Mass[g]"):
        self.df = pd.read_csv(file, sep=";")
        self.information = given_information
        self.df["Radionuclide"] = self.df["Radionuclide"].apply(lambda x: f"{''.join(filter(str.isalpha, x))}-{''.join(filter(str.isdigit, x))}")
        self.dict_rn = self.df.set_index("Radionuclide").to_dict()[self.information]

    def compute_source_term(self, 
                                  time: int = 0, 
                                  unit_energy: str = "keV") -> tuple:
        """
        Compute the total photon emission spectrum for a set of radionuclides.

        Parameters:
            dict_rn (dict): Dictionary of radionuclide names and their masses.
            time (int): Time in seconds after which to compute the activity.
            unit_energy (str): Energy unit for output rays ("keV", "MeV", or "eV").

        Returns:
            rays (np.ndarray): Array of photon energies.
            weights (np.ndarray): Corresponding weighted intensities.
            total_weight (float): Sum of all weights.
        """
        rays = []
        weights = []
        for rn, mass in self.dict_rn.items():
            rn_lara = Radionuclide_lara(rn)
            energy, intensity, _ = rn_lara.get_rays_emission_data(photon_only=True)
            activity = rn_lara.get_activity_after_time(mass=mass, time=time)
            rays.extend(energy)
            weights.extend(intensity * activity)
        rays = np.array(rays)
        weights = np.array(weights)
        if unit_energy == "keV":
            pass
        elif unit_energy == "MeV":
            rays = rays / 1000
        elif unit_energy == "eV":
            rays = rays * 1000
        total_weight = np.sum(weights)
        return rays, weights, total_weight

    def plot_source_term(self, time: int = 0, 
                         unit_energy: str = "keV", 
                         width: float = 0.05) -> plt.Figure:
        rays, weights, _ = self.compute_source_term(time=time, unit_energy=unit_energy)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.bar(rays, weights, width=width)
        ax.set_xlabel("Energy [keV]")
        ax.set_ylabel("Intensity [a.u.]")
        ax.set_title("Total Source Term")
        plt.show()
        return fig

    def compute_total_activity_decay(self, time_points: np.ndarray) -> list:
        """
        Compute the total activity (sum of all photon emission intensities) at multiple time points.

        Parameters:
            time_points (iterable): Sequence of time points (in seconds) at which to compute the total activity.

        Returns:
            total_weight_at_time (list): List of total photon emission intensities at each time point.
        """
        total_weight_at_time = []
        for time in time_points:
            _, _, total_weight = self.compute_source_term(time=time)
            total_weight_at_time.append(total_weight)
        return total_weight_at_time

    def plot_total_activity_decay(self, 
                            time_points:np.ndarray = np.linspace(0, 3600 * 24, 100), 
                            time_unit: str = "s",
                            savefig: bool = False, 
                            plot: bool = True) -> plt.Figure:
        """
        Plot the total activity (sum of all photon emission intensities) at multiple time points.

        Args:
            time_points (iterable): Sequence of time points (in seconds) at which to compute the total activity.
            time_unit (str): Unit for the x-axis ("s", "min", "h", "d", "w", "mo").
        """
        total_activity = self.compute_total_activity_decay(time_points)

        factor = UNIT_FACTORS.get(time_unit, 1)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(time_points / factor, total_activity, marker='x')
        ax.set_xlabel(f"Time [{time_unit}]")
        ax.set_ylabel("Total Activity [Bq]")
        ax.set_title("Total Activity of Radionuclides Over Time")
        ax.grid(True)
        fig.tight_layout()
        if savefig:
            plt.savefig("total_activity_decay.png")
        if plot:
            plt.show()
        return True

    def plot_total_activity_decay_per_rn(self, 
                                   time_points: np.ndarray = np.linspace(0, 3600 * 24, 100),
                                   time_unit: str = "s", 
                                   savefig: bool = False, 
                                   plot: bool = True) -> plt.Figure:
        """
        Plot the activity decay curve for each radionuclide in the list over the specified time points.

        Args:
            time_points (np.ndarray): Array of time points (in seconds) at which to compute the activity.
            time_unit (str): Unit for the x-axis ("s", "min", "h", "d", "w", "mo").
            savefig (bool): Whether to save the figure as a PNG file.
            plot (bool): Whether to display the plot.

        Returns:
            plt.Figure: The matplotlib Figure object containing the plot.
        """
        total_weights_per_rn = {}
        for rn, mass in self.dict_rn.items():
            rn_lara = Radionuclide_lara(rn)
            activities = np.array([rn_lara.get_activity_after_time(mass=mass, time=t) for t in time_points])
            total_weights_per_rn[rn] = activities

        factor = UNIT_FACTORS.get(time_unit, 1)
        fig, ax = plt.subplots(figsize=(9, 6))
        for rn, activities in total_weights_per_rn.items():
            ax.plot(time_points / factor, activities, label=rn)
        ax.set_xlabel(f"Time [{time_unit}]")
        ax.set_ylabel("Activity [Bq]")
        ax.set_title("Total Weighted Activity of Each Radionuclide vs Time")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_yscale("log")
        ax.grid(True)
        fig.tight_layout()
        if savefig:
            plt.savefig("total_activity_per_rn.png")
        if plot:
            plt.show()
        return fig


    def plot_activities_per_rn(self, 
                               time: float = 0.0, 
                               time_unit: str = "s",
                               savefig: bool = False,
                               plot: bool = True) -> plt.Figure:
        """
        Calculate and plot the activity of each radionuclide at a given time.

        Parameters:
            dict_rn (dict): Dictionary of radionuclide names and their masses.
            time (float): Time in seconds at which to calculate the activity.
        """
        activities_at_time = {}
        for rn, mass in self.dict_rn.items():
            rn_lara = Radionuclide_lara(rn)
            activity = rn_lara.get_activity_after_time(mass=mass, time=time)
            activities_at_time[rn] = activity

        factor = UNIT_FACTORS.get(time_unit, 1)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.bar(activities_at_time.keys(), activities_at_time.values())
        ax.set_xlabel("Radionuclide")
        ax.set_ylabel(f"Activity at t={time} [Bq]")
        ax.set_title(f"Activity of Each Radionuclide at t={time/factor:.2f} {time_unit}")
        ax.set_yscale("log")
        ax.set_xticklabels(activities_at_time.keys(), rotation=45)
        ax.grid(True)
        fig.tight_layout()
        if savefig:
            plt.savefig("activities_per_rn.png")
        if plot:
            plt.show()
        return fig


    def plot_weighted_activities_per_rn(self, 
                                        time=0, 
                                        time_unit="s",
                                        savefig=False,
                                        plot=True):
        """
        Calculate and plot the weighted activity (total photon emission) of each radionuclide at a given time.

        Parameters:
            dict_rn (dict): Dictionary of radionuclide names and their masses.
            time (float): Time in seconds at which to calculate the weighted activity.
        """
        weighted_activities = {}
        for rn, mass in self.dict_rn.items():
            rn_lara = Radionuclide_lara(rn)
            energy, intensity, _ = rn_lara.get_rays_emission_data(photon_only=True)
            activity = rn_lara.get_activity_after_time(mass=mass, time=time)
            weighted_activity = np.sum(intensity * activity)
            weighted_activities[rn] = weighted_activity

        factor = UNIT_FACTORS.get(time_unit, 1)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.bar(weighted_activities.keys(), weighted_activities.values())
        ax.set_xlabel("Radionuclide")
        ax.set_ylabel(f"Weighted Activity at t={time} [Bq]")
        ax.set_title(f"Activity of Each Radionuclide at t={time/factor:.2f} {time_unit}")
        ax.set_yscale("log")
        ax.set_xticklabels(weighted_activities.keys(), rotation=45)
        ax.grid(True)
        fig.tight_layout()
        if savefig:
            plt.savefig("weighted_activities_per_rn.png")
        if plot:
            plt.show()
        return fig


    def penalizing_source_term(self, 
                               time: float = 0.0, 
                               unit_energy: str = "MeV",
                               energy_min:float = 0.0, 
                               energy_max:float = 3.0, 
                               energy_width:float = 0.5):
        """
        Aggregate rays and weights into specified energy windows.

        Groups the photon emission spectrum into energy bins and sums the weights in each bin.

        Parameters:
            time (float): Time in seconds after which to compute the activity and spectrum.
            unit_energy (str): Energy unit for output rays ("keV", "MeV", or "eV").
            energy_min (float): Minimum energy of the window range (in unit_energy).
            energy_max (float): Maximum energy of the window range (in unit_energy).
            energy_width (float): Width of each energy window (in unit_energy).

        Returns:
            window_energies (list): Energy of the highest-energy ray in each window.
            window_weights (list): Sum of weights in each window.
        """
        rays, weights, _ = self.compute_source_term(time=time, unit_energy=unit_energy)
        rays = np.array(rays)
        weights = np.array(weights)

        bins = np.arange(energy_min, energy_max, energy_width)
        energy_windows = [(emin, emin + energy_width) for emin in bins]
        window_energies = []
        window_weights = []
        for (emin, emax) in energy_windows:
            mask = (rays >= emin) & (rays < emax)
            if np.any(mask):
                max_energy = rays[mask].max()
                sum_weight = weights[mask].sum()
                window_energies.append(max_energy)
                window_weights.append(sum_weight)
        return window_energies, window_weights

    def plot_penalizing_source_term(self, 
                                    time=0.0, 
                                    unit_energy="MeV", 
                                    energy_min:float = 0.0, 
                                    energy_max:float = 3.0, 
                                    energy_width:float = 0.5,
                                    savefig: bool = False, 
                                    plot: bool = True, 
                                    figsize:list=(9,6)) -> plt.Figure:
        """
        Plot the penalizing source term for a given time and energy unit.

        Parameters:
            time (float): Time in seconds at which to compute the source term.
            unit_energy (str): Energy unit for the plot ("keV", "MeV", or "eV").
        """
        window_energies, window_weights = self.penalizing_source_term(time=time, 
                                                                      unit_energy=unit_energy,
                                                                      energy_min=energy_min,
                                                                      energy_max=energy_max,
                                                                      energy_width=energy_width)
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(window_energies, window_weights, width=0.1)
        ax.set_xlabel(f"Energy [{unit_energy}]")
        ax.set_ylabel("Weight")
        ax.set_title(f"Penalizing Source Term at t={time} [s]")
        ax.grid(True)
        fig.tight_layout()
        if savefig:
            plt.savefig("penalizing_source_term.png")
        if plot:
            plt.show()
        return fig

    def __str__(self):
        return f"Radionuclide Source Term: {self.dict_rn}"
    