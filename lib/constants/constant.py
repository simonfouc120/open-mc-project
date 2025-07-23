import numpy as np
import openmc.data

# Constants for physical calculations
AVOGADRO_NUMBER = 6.022e23  # mol^-1
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
SPEED_OF_LIGHT = 299792458  # m/s
PI = np.pi  # Pi constant
# Atomic and nuclear constants
MASS_OF_ELECTRON = 9.10938356e-31  # kg
MASS_OF_PROTON = 1.6726219e-27  # kg
MASS_OF_NEUTRON = 1.675e-27  # kg
MASS_OF_NUCLEON = (MASS_OF_PROTON + MASS_OF_NEUTRON) / 2  # kg

# Time constants
SECONDS_PER_MINUTE = 60  # seconds in a minute
SECONDS_PER_HOUR = 3600  # seconds in an hour
SECONDS_PER_DAY = 86400  # seconds in a day
SECONDS_PER_WEEK = 86400 * 7  # seconds in a week
SECONDS_PER_MONTH = 86400 * 30.44  # seconds in a month (30.44 days)
SECONDS_PER_YEAR = 86400 * 365.25  # seconds in a year