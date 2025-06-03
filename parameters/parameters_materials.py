import openmc
import numpy as np
import openmc



displacement_threshold_energy = {
    'O': 20.0,
    'Si': 25.0,
    'Ca': 40.0,
    'Al': 25.0,
    'Fe': 40.0,
    'Mg': 25.0,
    'Na': 20.0,
    'K': 25.0,
    'U': 90.0,
    'C': 30.0,
    'Zr': 45.0,
    'Ni': 40.0,
    'Pb': 25.0
}

# Création des matériaux

## GAS MATERIALS ##

air = openmc.Material(name="Air")
air.add_element("N", 0.78)
air.add_element("O", 0.21)
air.add_element("Ar", 0.01)
air.set_density("g/cm3", 0.001225)  
AIR_MATERIAL = air 

helium = openmc.Material(name="Helium")
helium.add_element("He", 1.0)
helium.set_density("g/cm3", 0.0001786)
HELIUM_MATERIAL = helium

co2 = openmc.Material(name="CO2")
co2.add_element("C", 1.0)
co2.add_element("O", 2.0)
co2.set_density("g/cm3", 0.001977)
CO2_MATERIAL = co2

## RADIOACTIVE SOURCE MATERIALS ##

cs137 = openmc.Material(name="Cs137")
cs137.add_nuclide("Cs137", 1.0)
cs137.set_density("g/cm3", 4.0)
CS137_MATERIAL = cs137

## DETECTOR MATERIALS ##

cdte = openmc.Material(name="CdTe")
cdte.add_element("Cd", 1.0)
cdte.add_element("Te", 1.0)
cdte.set_density("g/cm3", 6.2)
CDTE_MATERIAL = cdte

csi = openmc.Material(name="CsI")
csi.add_element("Cs", 1.0)
csi.add_element("I", 1.0)
csi.set_density("g/cm3", 4.51)
CSI_MATERIAL = csi

nai = openmc.Material(name="NaI")
nai.add_element("Na", 1.0)
nai.add_element("I", 1.0)
nai.set_density("g/cm3", 3.67)
NAI_MATERIAL = nai

cdznte = openmc.Material(name="CdZnTe")
cdznte.add_element("Cd", 0.9)
cdznte.add_element("Zn", 0.1)
cdznte.add_element("Te", 1.0)
cdznte.set_density("g/cm3", 6.08)
CDZNTE_MATERIAL = cdznte

## GENIE CIVIL ## 

concrete = openmc.Material(name="Concrete")
concrete.add_element("O", 0.525)
concrete.add_element("Si", 0.325)
concrete.add_element("Ca", 0.06)
concrete.add_element("Al", 0.025)
concrete.add_element("Fe", 0.02)
concrete.add_element("Mg", 0.015)
concrete.add_element("Na", 0.015)
concrete.add_element("K", 0.015)
concrete.set_density("g/cm3", 2.3)
CONCRETE_MATERIAL = concrete

lead = openmc.Material(name="Lead")
lead.add_element("Pb", 1.0)
lead.set_density("g/cm3", 11.34)
LEAD_MATERIAL = lead


# GRAPHITE FUEL MATERIALS ##
graphite = openmc.Material(name="Graphite")
graphite.add_element("C", 1.0)
graphite.set_density("g/cm3", 1.7)
GRAPHITE_MATERIAL = graphite

fuel = openmc.Material(name="Fuel")
fuel.add_nuclide("U235", 0.1975)
fuel.add_nuclide("U238", 0.8025)
fuel.set_density("g/cm3", 10.0)
FUEL_MATERIAL = fuel

# URANIUM NATURAL ##
uranium = openmc.Material(name="Uranium")
uranium.add_element("U", 1.0)
uranium.set_density("g/cm3", 18.95)
URANIUM_MATERIAL = uranium

# WATER MODERATOR MATERIALS ##
water = openmc.Material(name="Water")
water.add_element("H", 2.0)
water.add_element("O", 1.0)
water.set_density("g/cm3", 1.0)
WATER_MATERIAL = water

# WATER HEAVY MATERIALS ##
heavy_water = openmc.Material(name="Heavy Water")
heavy_water.add_nuclide("H2", 2.0)
heavy_water.add_element("O", 1.0)
heavy_water.set_density("g/cm3", 1.11)
HEAVY_WATER_MATERIAL = heavy_water

# BORON MATERIALS ##
boron = openmc.Material(name="Boron")
boron.add_element("B", 1.0)
boron.set_density("g/cm3", 2.34)
BORON_MATERIAL = boron

# BERYLLIUM MATERIALS ##
beryllium = openmc.Material(name="Beryllium")
beryllium.add_element("Be", 1.0)
beryllium.set_density("g/cm3", 1.85)
BERYLLIUM_MATERIAL = beryllium

# STEEL 
steel = openmc.Material(name="Steel")
steel.add_element("Fe", 0.98)
steel.add_element("C", 0.02)
steel.set_density("g/cm3", 7.85)
STEEL_MATERIAL = steel
