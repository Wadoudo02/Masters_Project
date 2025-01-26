#%%
import matplotlib.pyplot as plt
import numpy as np
#from SMEFT_classification import special_features
from SMEFT_utils import *
from chi2 import mu_c
from nll import *
from background_dist import get_back_int
import joblib
import copy

#Extract relevant columns from overall df
cats = [0, 0.4, 0.5, 0.6, 0.7,1]
special_features = ["deltaR_sel", "HT_sel", "n_jets_sel", "delta_phi_gg_sel","lead_pt-over-mass_sel"] 
ttH_df = get_tth_df()
scaler = joblib.load('saved_models/scaler.pkl')

dfs = get_dfs(new_sample_path)

for i, proc in enumerate(procs.keys()):
    #dfs[proc].dropna(inplace=True)
    dfs[proc] = get_selection(dfs[proc], proc)

    invalid_weights = dfs[proc]["true_weight_sel"] <= 0
    if invalid_weights.sum() > 0:
        print(f" --> Removing {invalid_weights.sum()} rows with invalid weights.")
        dfs[proc] = dfs[proc][~invalid_weights]

dfs_copy = copy.deepcopy(dfs)
dfs["ttH"] = ttH_df
dfs["ttH_EFT"] = dfs["ttH"].copy()
#Rescaling eft weights so that sum of eft weights = sum of sm weights
dfs["ttH_EFT"]["true_weight_sel"] = dfs["ttH_EFT"]["EFT_weight"]*sum(dfs["ttH"]["true_weight_sel"])/sum(dfs["ttH_EFT"]["EFT_weight"])


for proc, df in dfs.items():
    
    weight_col = "true_weight_sel"
    
    new_df = pd.concat([df[feature] for feature in special_features], axis=1)
    mass = df["mass_sel"]
    weight = df[weight_col]
    new_df=scaler.transform(new_df)
    new_df = np.concatenate([new_df, mass.to_numpy().reshape(-1,1), weight.to_numpy().reshape(-1,1)], axis=1)
    dfs[proc] = torch.tensor(new_df, dtype=torch.float32)

#Get predictions for all events
dfs_preds = {}
dfs_cats = {}

#Loading model
input_dim = len(special_features)
hidden_dim = [256, 64, 32, 16, 16, 8]
model = ComplexNN(input_dim, hidden_dim, 1)
model.load_state_dict(torch.load("saved_models/model.pth"))
model.eval()

for proc, df in dfs.items():
    #print(proc, df)
    mass,weight = df[:,-2],df[:,-1]
    df = df[:,:-2]
    
    probs = eval_in_batches(model, df)
    # fig, ax, = plt.subplots(figsize=(10, 6))
    # plot_classifier_output(probs, np.zeros(len(probs)), weight.flatten(), ax)
    # probs = model(df)
    #print(proc, list(probs))
    dfs_preds[proc] = [probs, mass.flatten().numpy(),weight.flatten().numpy()]
with_back=True
order = ["ttH_EFT","background","ggH","VBF", "VH","ttH"]
for proc in order:
    preds,mass, weights = dfs_preds[proc]
    dfs_cats[proc]={"mass":[],"weights":[]}
    for i in range(1, len(cats)):
        bools = ((cats[i-1]<preds) & (preds<cats[i])).squeeze()
        #dfs_cats[proc]["events"].append(df[bools.numpy()])
        # if proc != "ttH_EFT":
        #     dfs_copy[proc]["categories"] = 1
        #     dfs_copy[proc]["categories"] = np.where(bools, i, dfs_copy[proc]["categories"])
        dfs_cats[proc]["weights"].append(weights[bools])
        dfs_cats[proc]["mass"].append(mass[bools])

#Over number of cats
fig, ax = plt.subplots(ncols=len(dfs_cats["ggH"]["mass"]),figsize=(30, 5))    
for i in range(len(dfs_cats["ggH"]["mass"])):
    for proc in order:
        if not with_back and proc=="background":
            continue
        mass = dfs_cats[proc]["mass"][i]
        weight = dfs_cats[proc]["weights"][i]
        #print(proc, "cat: ", i, dfs_cats[proc]["mass"][i].shape)
        sns.histplot(x=mass, weights=weight, bins=50, label=proc, ax=ax[i], fill=False, element="step")
        ax[i].legend()
        ax[i].set_title(f"Category {i}")
        ax[i].set_xlabel("mass (GeV)")
        ax[i].set_ylabel("Events")

#%%
#Getting weighted average of coefficients

ttH_probs = dfs_preds["ttH"][0]
ttH_cats = []
for i in range(1, len(cats)):
    bools = ((cats[i-1]<ttH_probs) & (ttH_probs<cats[i])).squeeze()
    ttH_cats.append(ttH_df[bools])

a_cgs = []
a_ctgs = []
b_cg_cgs = []
b_cg_ctgs = []
b_ctg_ctgs = []

for cat in ttH_cats:
    a_cgs.append(np.average(cat["a_cg"], weights=cat["true_weight_sel"]))
    a_ctgs.append(np.average(cat["a_ctgre"], weights=cat["true_weight_sel"]))
    b_cg_cgs.append(np.average(cat["b_cg_cg"], weights=cat["true_weight_sel"]))
    b_cg_ctgs.append(np.average(cat["b_cg_ctgre"], weights=cat["true_weight_sel"]))
    b_ctg_ctgs.append(np.average(cat["b_ctgre_ctgre"], weights=cat["true_weight_sel"]))


hists = {}
mass_range = (120,130)
mass_bins = 5
v = 'mass'
c_vals = np.linspace(-3, 3, 100)

#TODO: Not fitting background dsitribution right now but can.
for cat in range(len(cats)-1):
    hists[cat] = {}
    for proc in procs.keys():
        #Adding background events from background distribution
        # if proc=="background":
        #     hist_counts=get_back_int(dfs_copy["background"], cat, mass_range, mass_bins, len(cats))
        # else:
            #cat_mask = dfs_cats[proc]['category'] == cat
        hist_counts = np.histogram(dfs_cats[proc][v][cat], mass_bins, mass_range, weights=dfs_cats[proc]['weights'][cat])[0]
        hists[cat][proc] = hist_counts

nll_vals = []
for c_g in c_vals:
    nll_val = calc_nll_simple(hists, mu_c(c_g=c_g, c_tg=0, a_cgs=a_cgs,a_ctgs=a_ctgs,b_cg_cgs=b_cg_cgs,b_ctg_ctgs=b_ctg_ctgs,b_cg_ctgs=b_cg_ctgs,second_order=True))
    #print(nll_val)
    nll_vals.append(nll_val)
#print(nll_vals)
dnll = TwoDeltaNLL(nll_vals)
plt.plot(c_vals, dnll)