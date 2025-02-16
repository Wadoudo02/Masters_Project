import matplotlib.pyplot as plt
import numpy as np
#from SMEFT_classification import special_features
from SMEFT_utils import *
from chi2 import mu_c
from nll import *
from background_dist import get_back_int
import joblib
import copy
from Plotter import Plotter
from chi2 import get_chi_squared

plt.style.use(hep.style.CMS)

plotter = Plotter()

#Extract relevant columns from overall df
categories = [0, 0.4, 0.5, 0.6, 0.7,1]
special_features = ["deltaR_sel", "HT_sel", "n_jets_sel", "delta_phi_gg_sel", "pt-over-mass_sel"]#,"lead_pt-over-mass_sel"] 


ttH_df = get_tth_df()
scaler = joblib.load('saved_models/scaler.pkl')

dfs = get_dfs(new_sample_path)

for i, proc in enumerate(procs.keys()):
    #dfs[proc].dropna(inplace=True)
    dfs[proc] = get_selection(dfs[proc], proc)

    invalid_weights = dfs[proc]["true_weight_sel"] <= 0
    init_yield = dfs[proc]["true_weight_sel"].sum()
    if invalid_weights.sum() > 0:
        print(f" --> Removing {invalid_weights.sum()} rows with invalid weights.")
        dfs[proc] = dfs[proc][~invalid_weights]
    
    new_yield = dfs[proc]["true_weight_sel"].sum()

    #Making sure final yield is the same as initial yield.
    dfs[proc]["true_weight_sel"] = dfs[proc]["true_weight_sel"]*init_yield/new_yield

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

# model = WadNeuralNetwork(input_dim, input_dim*3)
# model.load_state_dict(torch.load("saved_models/wad_neural_network.pth"))

model = ComplexNN(input_dim, hidden_dim, 1)
model.load_state_dict(torch.load("saved_models/model.pth"))
model.eval()
min_prob = float("inf")
max_prob = float("-inf")
#Plotting classifier output
for proc, df in dfs.items():
    #print(proc, df)
    mass,weight = df[:,-2],df[:,-1]
    df = df[:,:-2]
    
    probs = eval_in_batches(model, df)
    #plot_classifier_output(probs, np.zeros(len(probs)), weight.flatten(), ax[i])
    min_prob = min(min_prob, probs.min())
    max_prob = max(max_prob, probs.max())
    dfs_preds[proc] = [probs, mass.flatten().numpy(),weight.flatten().numpy(), df.numpy()]
with_back=True
order = ["ttH_EFT","background","ggH","VBF", "VH","ttH"]
#%%

#Trying to minimise this function
def get_constraints(cats = categories):
    cats = [0] + list(cats) + [1]
    for proc in order:
        preds,mass, weights, feats = dfs_preds[proc]
        dfs_cats[proc]={"mass":[],"weights":[], "features":[]}
        for i in range(1, len(cats)):
            bools = ((cats[i-1]<preds) & (preds<cats[i])).squeeze()
            dfs_cats[proc]["weights"].append(weights[bools])
            dfs_cats[proc]["mass"].append(mass[bools])
            dfs_cats[proc]["features"].append(feats[bools])

    # num_cats = len(dfs_cats["ggH"]["mass"])
    
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
    print(cats)
    for cat in ttH_cats:
        print(len(cat))
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


    for proc in procs.keys():
        hists[proc] = np.array([])
        for cat in range(len(cats)-1):
            #Adding background events from background distribution
            if proc=="background":
                hist_counts=get_back_int(dfs_copy["background"], cat, mass_range, mass_bins, len(cats)-1)
            else:
                #cat_mask = dfs_cats[proc]['category'] == cat
                hist_counts = np.histogram(dfs_cats[proc][v][cat], mass_bins, mass_range, weights=dfs_cats[proc]['weights'][cat])[0]
            hists[proc] = np.append(hists[proc], hist_counts)
    
    nll_vals_cg = []
    nll_vals_ctg = []

    for c in c_vals:
        res_cg = minimize(
            lambda x: calc_nll_simple(hists, mu_c(c_g=c, c_tg=x, a_cgs=a_cgs,a_ctgs=a_ctgs,b_cg_cgs=b_cg_cgs,b_ctg_ctgs=b_ctg_ctgs,b_cg_ctgs=b_cg_ctgs,second_order=True), "ttH"),0,method='Nelder-Mead')
        res_ctg = minimize(
            lambda x: calc_nll_simple(hists, mu_c(c_g=x, c_tg=c, a_cgs=a_cgs,a_ctgs=a_ctgs,b_cg_cgs=b_cg_cgs,b_ctg_ctgs=b_ctg_ctgs,b_cg_ctgs=b_cg_ctgs,second_order=True), "ttH"),0,method='Nelder-Mead')
        nll_vals_cg.append(res_cg.fun)
        nll_vals_ctg.append(res_ctg.fun)

        #print(nll_val)
    #print(nll_vals)
    dnll_cg = TwoDeltaNLL(nll_vals_cg)
    dnll_ctg = TwoDeltaNLL(nll_vals_ctg)

    cg_fit, cg_cons_up, cg_cons_down = find_crossings([c_vals, dnll_cg], 1.)
    cg_cons = (cg_cons_up, cg_cons_down)
    ctg_fit, ctg_cons_up, ctg_cons_down = find_crossings([c_vals, dnll_ctg], 1.)
    ctg_cons = (ctg_cons_up, ctg_cons_down)

    return abs(cg_cons[0])+abs(cg_cons[1])+abs(ctg_cons[0])+abs(ctg_cons[1])
lower = min_prob+0.1
upper = max_prob-0.1
init_guess = [0.4, 0.5, 0.6, 0.7]
#init_guess = [0.4905974388217408, 0.5436479211218064, 0.6436479210861633, 0.7436479061841965]

bounds = [(lower, upper), (lower, upper), (lower, upper), (lower, upper)]

min_sep = 0.05
options = {'eps': 1e-4, 'ftol': 1e-6}  # Reduce step size and tolerance
# Define constraints to ensure increasing order
constraints = [{'type': 'ineq', 'fun': lambda x, i=i: x[i+1] - x[i] - min_sep} for i in range(len(init_guess) - 1)]
#constraints = [{'type': 'ineq', 'fun': lambda x: x[i+1] - x[i]} for i in range(len(init_guess) - 1)]


res = minimize(get_constraints, init_guess, method='SLSQP', bounds=bounds, constraints=constraints, options=options)

# Print the result
print("Optimization result:")
print(res)

# Extract the optimized values
optimized_values = res.x
print("Optimized values:", optimized_values)