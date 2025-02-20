#%%
from utils import *
from SMEFT_utils import *
import numpy as np
import matplotlib.pyplot as plt
from Plotter import Plotter


plotter = Plotter()

dfs = get_dfs(sample_path=sample_path)
fig, ax = plt.subplots(figsize=(10, 8))
plotter.histogram(dfs["ttH"]["deltaR_sel"], bins=80, xlabel="pT (GeV)", ylabel="Events", legend_label="ttH", color=plotter.colors["blue"], axes=ax, density=True)

