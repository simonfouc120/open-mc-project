import openmc
import numpy as np
import openmc

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
