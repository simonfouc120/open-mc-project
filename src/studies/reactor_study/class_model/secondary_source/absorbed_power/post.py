import openmc
import os 
import json
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
import sys 
from PIL import Image
import numpy as np

CWD = Path(__file__).parent.resolve() 
project_root = Path(__file__).resolve().parents[6]  
sys.path.append(str(project_root))

from parameters.parameters_paths import PATH_TO_CROSS_SECTIONS, IMAGE_PATH
from parameters.parameters_materials import *
from src.utils.pre_processing.pre_processing import *
from src.utils.post_preocessing.post_processing import *
from src.models.model_complete_reactor_class import *

results_path = "simulation_results.json"
with open(results_path, "r") as f:
    results = json.load(f)
neutron_emission_rate = results["neutrons_emitted_per_second"]["value"]

statepoint = openmc.StatePoint(f"statepoint.100.h5")
tally = statepoint.get_tally(name="heating_tally")

df = tally.get_pandas_dataframe()
# new cell of mean in J/source
df["Power [W]"] = df["mean"] * 1.602e-19 * neutron_emission_rate
df["Std Dev [W]"] = df["std. dev."] * 1.602e-19 * neutron_emission_rate

max_power = df["Power [W]"].max()

print(f"Max power deposited in concrete walls: {max_power} W")