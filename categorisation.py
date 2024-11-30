import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *

procs = {
    "background" : ["Background", "black"],
    "ttH" : ["ttH (x10)", "mediumorchid"],
    "ggH" : ["ggH (x10)", "cornflowerblue"],
    "VBF" : ["VBF (x10)", "green"],
    "VH" : ["VH (x10)", "brown"]
}
col_name = "_sel"

def get_pt_cat(data, bins = [0,60,120,200,300]):
    conditions = [(data>=bins[i]) & (data<bins[i+1]) for i in range(len(bins)-1)]
    conditions.append(data>=bins[-1])
    return np.select(conditions, [i for i in range(len(bins))])

def get_categorisation(dfs):
    for i, proc in enumerate(procs.keys()):   
        mass = dfs[proc]["mass"+col_name]
        pt_mass = dfs[proc]["pt-over-mass"+col_name]
        pt = pt_mass*mass

        dfs[proc]['category'] =  get_pt_cat(pt, bins=[0,60,120,200,300])
    return dfs