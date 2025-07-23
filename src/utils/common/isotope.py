import openmc
import numpy as np

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
    
    