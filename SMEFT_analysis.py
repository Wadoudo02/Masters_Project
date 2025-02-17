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
from Plotter import Plotter
from chi2 import get_chi_squared

plt.style.use(hep.style.CMS)

plotter = Plotter()
mine = True
norm_eft = False
param = False
cg = 0.3
ctg = 0.69

#Extract relevant columns from overall df
wad_cats = [0, 0.22491833, 0.27491833, 0.32491833, 0.69582565,1] if not param else [0, 0.17430348, 0.22430348, 0.27430348, 0.55810981,1]
my_cats = [0,0.32273505, 0.37273505, 0.42273505, 0.65670192,1] if not param else [0, 0.26127338, 0.31127338, 0.36136802, 0.7,1]

#cats = [0, 0.4, 0.5, 0.6, 0.7,1]
special_features = ["deltaR_sel", "HT_sel", "n_jets_sel", "delta_phi_gg_sel", "pt-over-mass_sel"]#,"lead_pt-over-mass_sel"] 

#%%
#plot_SMEFT_features(special_features)
#%%

ttH_df = get_tth_df(cg=cg, ctg=ctg)
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
    dfs[proc]["true_weight_sel"] = (dfs[proc]["true_weight_sel"]/new_yield)*init_yield

dfs_copy = copy.deepcopy(dfs)
dfs["ttH"] = ttH_df
dfs["ttH_EFT"] = dfs["ttH"].copy()
#Rescaling eft weights so that sum of eft weights = sum of sm weights
dfs["ttH_EFT"]["norm_weight_sel"] = dfs["ttH_EFT"]["EFT_weight"]*sum(dfs["ttH"]["true_weight_sel"])/sum(dfs["ttH_EFT"]["EFT_weight"])
if norm_eft:
    dfs["ttH_EFT"]["true_weight_sel"] = dfs["ttH_EFT"]["norm_weight_sel"]
else:
    dfs["ttH_EFT"]["true_weight_sel"] = dfs["ttH_EFT"]["EFT_weight"]

for proc, df in dfs.items():
    
    weight_col = "true_weight_sel"

    new_df = pd.concat([df[feature] for feature in special_features], axis=1)
    mass = df["mass_sel"]
    weight = df[weight_col]
    new_df=scaler.transform(new_df)
    if param:
        cg_vals = np.array([cg]*len(new_df))
        ctg_vals = np.array([ctg]*len(new_df))
        new_df = np.concatenate([new_df, cg_vals.reshape(-1,1), ctg_vals.reshape(-1,1)], axis=1)
    new_df = np.concatenate([new_df, mass.to_numpy().reshape(-1,1), weight.to_numpy().reshape(-1,1)], axis=1)
    dfs[proc] = torch.tensor(new_df, dtype=torch.float32)

#%%
#Get predictions for all events
order = ["ttH_EFT","background","ggH","VBF", "VH","ttH"]
with_back = True

#Loading model
input_dim = len(new_df[0])-2 #-2 for mass and weight column
if not param:
    hidden_dim = [256, 64, 32, 16,16, 8]
else:
    hidden_dim = [256, 64, 32, 16, 8]

if mine:
    model = ComplexNN(input_dim, hidden_dim, 1)
    if param:
        model.load_state_dict(torch.load("saved_models/param_model.pth"))
    else:
        model.load_state_dict(torch.load("saved_models/model.pth"))
    cats = my_cats
else:
    model = WadNeuralNetwork(input_dim, input_dim*3)
    if param:
        model.load_state_dict(torch.load("saved_models/wad_param_model.pth"))
    else:
        model.load_state_dict(torch.load("saved_models/wad_neural_network.pth"))
    cats = wad_cats

dfs_preds, dfs_cats = get_preds_cats(dfs,model=model, cats=cats, order=order)
num_cats = len(dfs_cats["ggH"]["mass"])


#Diphoton mass distribution
fig, ax = plt.subplots(ncols=num_cats,figsize=(30, 5))    

#Over number of cats
for i in range(len(dfs_cats["ggH"]["mass"])):
    for j in range(len(order)):
        proc = order[j]
        if not with_back and proc=="background":
            continue
        mass = dfs_cats[proc]["mass"][i]
        weight = dfs_cats[proc]["weights"][i]
        #print(proc, "cat: ", i, dfs_cats[proc]["mass"][i].shape)
        ax[i].hist(mass, bins=50, weights=weight, label=proc, histtype="step")
        #sns.histplot(x=mass, weights=weight, bins=50, label=proc, ax=ax[i], fill=False, element="step")
        ax[i].legend()
        ax[i].set_title(f"Category {i}")
        ax[i].set_xlabel("mass (GeV)")
        ax[i].set_ylabel("Events")

#Plotting features after categorisation
fig, ax = plt.subplots(ncols = len(special_features),figsize=(30, 5))

#Over features
for feat in range(len(special_features)):
    
    plotter.overlay_histograms(
        [dfs_cats["ttH"]["features"][i][:,feat] for i in range(len(dfs_cats["ttH"]["features"]))],
        bins=50,
        title=f"{special_features[feat]} distribution", xlabel=special_features[feat],
        ylabel="Events", labels=[f"Cat {i}" for i in range(num_cats)],
        colors=["red", "blue", "green", "black", "purple"],
        weights = [dfs_cats["ttH"]["weights"][i] for i in range(num_cats)],
        axes=ax[feat],
        type="step",
        density=True
        )

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

#%%
#Frozen scan over c_g and c_tg

nll_vals_cg = []
nll_vals_ctg = []

for c in c_vals:
    nll_vals_cg.append(calc_nll_simple(hists, mu_c(c_g=c, c_tg=0, a_cgs=a_cgs,a_ctgs=a_ctgs,b_cg_cgs=b_cg_cgs,b_ctg_ctgs=b_ctg_ctgs,b_cg_ctgs=b_cg_ctgs,second_order=True)))
    nll_vals_ctg.append(calc_nll_simple(hists, mu_c(c_g=0, c_tg=c, a_cgs=a_cgs,a_ctgs=a_ctgs,b_cg_cgs=b_cg_cgs,b_ctg_ctgs=b_ctg_ctgs,b_cg_ctgs=b_cg_ctgs,second_order=True)))

    #print(nll_val)
#print(nll_vals)
dnll_cg = TwoDeltaNLL(nll_vals_cg)
dnll_ctg = TwoDeltaNLL(nll_vals_ctg)

cg_fit, cg_cons_up, cg_cons_down = find_crossings([c_vals, dnll_cg], 1.)
cg_cons = (cg_cons_up, cg_cons_down)
ctg_fit, ctg_cons_up, ctg_cons_down = find_crossings([c_vals, dnll_ctg], 1.)
ctg_cons = (ctg_cons_up, ctg_cons_down)
#%%
#New NLL analysis for original method of pt categorisation

comb_hist = joblib.load("saved_models/comb_hist.pkl")

nll_vals_pt_cg = []
nll_vals_pt_ctg = []

for c in c_vals:
    nll_vals_pt_cg.append(calc_NLL_comb(comb_hist, mu_c(c_g=c, c_tg=0, a_cgs=a_cgs,a_ctgs=a_ctgs,b_cg_cgs=b_cg_cgs,b_ctg_ctgs=b_ctg_ctgs,b_cg_ctgs=b_cg_ctgs,second_order=True)))
    nll_vals_pt_ctg.append(calc_NLL_comb(comb_hist, mu_c(c_g=0, c_tg=c, a_cgs=a_cgs,a_ctgs=a_ctgs,b_cg_cgs=b_cg_cgs,b_ctg_ctgs=b_ctg_ctgs,b_cg_ctgs=b_cg_ctgs,second_order=True)))
    

dnll_pt_cg = TwoDeltaNLL(nll_vals_pt_cg)
dnll_pt_ctg = TwoDeltaNLL(nll_vals_pt_ctg)

pt_cg_fit, pt_cg_cons_up, pt_cg_cons_down = find_crossings([c_vals, dnll_pt_cg], 1.)
pt_cg_cons = (pt_cg_cons_up, pt_cg_cons_down)
pt_ctg_fit, pt_ctg_cons_up, pt_ctg_cons_down = find_crossings([c_vals, dnll_pt_ctg], 1.)
pt_ctg_cons = (pt_ctg_cons_up, pt_ctg_cons_down)

print(f"Crossing for cg with pt: {pt_cg_cons} and ctg: {pt_ctg_cons}")
print(f"Crossing for cg with nn: {cg_cons} and ctg: {ctg_cons}")

fig, ax = plt.subplots(1,2, figsize=(10, 6))
fig.suptitle("NLL frozen minimisation over c_g and c_tg")
ax[0].set_ylim(0, 3000)
ax[1].set_ylim(0, 3000)
plotter.overlay_line_plots(
    x=c_vals,
    y_datasets=[dnll_cg, dnll_pt_cg],
    title="Delta nll minimisation over c_g",
    xlabel="c_g",
    ylabel="2*Delta NLL",
    labels=[
        fr"NN cat ${cg_fit:.2f}^{{+{cg_cons[0]:.2f}}}_{{{cg_cons[1]:.2f}}}$",
        fr"Pt cat ${pt_cg_fit:.2f}^{{+{pt_cg_cons[0]:.2f}}}_{{{pt_cg_cons[1]:.2f}}}$"
    ],
    axes=ax[0],
    ylim=[0, 10])
plotter.overlay_line_plots(
    x=c_vals,
    y_datasets=[dnll_ctg, dnll_pt_ctg],
    title="Delta nll minimisation over c_tg",
    xlabel="c_tg", ylabel="2*Delta NLL",
    labels=[
        fr"NN cat ${ctg_fit:.2f}^{{+{ctg_cons[0]:.2f}}}_{{{ctg_cons[1]:.2f}}}$",
        fr"Pt cat ${pt_ctg_fit:.2f}^{{+{pt_ctg_cons[0]:.2f}}}_{{{pt_ctg_cons[1]:.2f}}}$"
    ],
    axes=ax[1],
    ylim=[0, 10])
ax[0].plot(c_vals, np.ones(len(c_vals)), linestyle="--")
ax[1].plot(c_vals, np.ones(len(c_vals)),linestyle="--" )

# %%
#Profiled scan over c_g and c_tg for new categorisation

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

if param:
    if mine:
        # Save the values using joblib
        joblib.dump({'dnll_cg': dnll_cg, 'dnll_ctg': dnll_ctg}, 'param_dnll_values.pkl')
    else:
        joblib.dump({'dnll_cg': dnll_cg, 'dnll_ctg': dnll_ctg}, 'wad_param_dnll_values.pkl')


cg_fit, cg_cons_up, cg_cons_down = find_crossings([c_vals, dnll_cg], 1.)
cg_cons = (cg_cons_up, cg_cons_down)
ctg_fit, ctg_cons_up, ctg_cons_down = find_crossings([c_vals, dnll_ctg], 1.)
ctg_cons = (ctg_cons_up, ctg_cons_down)

#Profiled  NLL analysis for original method of pt categorisation
nll_vals_pt_cg = []
nll_vals_pt_ctg = []

for c in c_vals:
    res_cg = minimize(
        lambda x: calc_NLL_comb(comb_hist, mu_c(c_g=c, c_tg=x, a_cgs=a_cgs,a_ctgs=a_ctgs,b_cg_cgs=b_cg_cgs,b_ctg_ctgs=b_ctg_ctgs,b_cg_ctgs=b_cg_ctgs,second_order=True), "ttH"),0,method='Nelder-Mead')
    res_ctg = minimize(
        lambda x: calc_NLL_comb(comb_hist, mu_c(c_g=x, c_tg=c, a_cgs=a_cgs,a_ctgs=a_ctgs,b_cg_cgs=b_cg_cgs,b_ctg_ctgs=b_ctg_ctgs,b_cg_ctgs=b_cg_ctgs,second_order=True), "ttH"),0,method='Nelder-Mead')
    nll_vals_pt_cg.append(res_cg.fun)
    nll_vals_pt_ctg.append(res_ctg.fun)
    

dnll_pt_cg = TwoDeltaNLL(nll_vals_pt_cg)
dnll_pt_ctg = TwoDeltaNLL(nll_vals_pt_ctg)

pt_cg_fit, pt_cg_cons_up, pt_cg_cons_down = find_crossings([c_vals, dnll_pt_cg], 1.)
pt_cg_cons = (pt_cg_cons_up, pt_cg_cons_down)
pt_ctg_fit, pt_ctg_cons_up, pt_ctg_cons_down = find_crossings([c_vals, dnll_pt_ctg], 1.)
pt_ctg_cons = (pt_ctg_cons_up, pt_ctg_cons_down)

print(f"Crossing for cg with pt: {pt_cg_cons} and ctg: {pt_ctg_cons}")
print(f"Crossing for cg with nn: {cg_cons} and ctg: {ctg_cons}")


#Getting values from chi squared analysis
chi_2_c_g, chi_2_c_tg, best_cg_chi, conf_cg_chi, best_ctg_chi, conf_ctg_chi=joblib.load("saved_models/chi2.pkl")
if not param:
    if mine:
        param_dnll = joblib.load("param_dnll_values.pkl")
        param_dnll_cg, param_dnll_ctg = param_dnll["dnll_cg"], param_dnll["dnll_ctg"]
    else:
        wad_param_dnll = joblib.load("wad_param_dnll_values.pkl")
        param_dnll_cg, param_dnll_ctg = wad_param_dnll["dnll_cg"], wad_param_dnll["dnll_ctg"]
    
    param_cg_fit, param_cg_cons_up, param_cg_cons_down = find_crossings([c_vals, param_dnll_cg], 1.)
    param_cg_cons = (pt_cg_cons_up, pt_cg_cons_down)
    param_ctg_fit, param_ctg_cons_up, param_ctg_cons_down = find_crossings([c_vals, param_dnll_ctg], 1.)
    param_ctg_cons = (pt_ctg_cons_up, pt_ctg_cons_down)

fig, ax = plt.subplots(1,2, figsize=(12, 6))
fig.suptitle("NLL Profiled minimisation over c_g and c_tg")
ax[0].set_ylim(0, 1500)
ax[1].set_ylim(0, 1500)
plotter.overlay_line_plots(
    x=c_vals,
    y_datasets=[dnll_cg, dnll_pt_cg]+([param_dnll_cg] if not param else []),# chi_2_c_g],
    title="Delta nll minimisation over c_g",
    xlabel="c_g",
    ylabel="2*Delta NLL",
    labels=[("Param " if param else "")+
        (fr"NN cat ${cg_fit:.2f}^{{+{cg_cons[0]:.2f}}}_{{{cg_cons[1]:.2f}}}$"),
        fr"STXS cat ${pt_cg_fit:.2f}^{{+{pt_cg_cons[0]:.2f}}}_{{{pt_cg_cons[1]:.2f}}}$",
        #fr"$\chi^2$ cat ${best_cg_chi:.2f}^{{+{conf_cg_chi[1]:.2f}}}_{{{conf_cg_chi[0]:.2f}}}$"
    ]+([fr"Param NN ${param_cg_fit:.2f}^{{+{param_cg_cons[0]:.2f}}}_{{{param_cg_cons[1]:.2f}}}$"] if not param else []),
    colors=["red", "blue"]+(["green"] if not param else []),# "green"],
    axes=ax[0],
    ylim=[0, 10])
plotter.overlay_line_plots(
    x=c_vals,
    y_datasets=[dnll_ctg, dnll_pt_ctg]+([param_dnll_ctg] if not param else []),# chi_2_c_tg],
    title="Delta nll minimisation over c_tg",
    xlabel="c_tg", ylabel="2*Delta NLL",
    labels=[("Param " if param else "")+
        (fr"NN cat ${ctg_fit:.2f}^{{+{ctg_cons[0]:.2f}}}_{{{ctg_cons[1]:.2f}}}$"),
        fr"STXS cat ${pt_ctg_fit:.2f}^{{+{pt_ctg_cons[0]:.2f}}}_{{{pt_ctg_cons[1]:.2f}}}$",
        #fr"$\chi^2$ cat ${best_ctg_chi:.2f}^{{+{conf_ctg_chi[1]:.2f}}}_{{{conf_ctg_chi[0]:.2f}}}$"
    ]+([fr"Param NN ${param_ctg_fit:.2f}^{{+{param_ctg_cons[0]:.2f}}}_{{{param_ctg_cons[1]:.2f}}}$"] if not param else []),
    colors=["red", "blue"] +(["green"] if not param else []),# "green"],
    axes=ax[1],
    ylim=[0, 10])
ax[0].plot(c_vals, np.ones(len(c_vals)), linestyle="--")
ax[1].plot(c_vals, np.ones(len(c_vals)),linestyle="--" )


#%%
#Grid minimisation

width = 4
hessian_comb = joblib.load("saved_models/hessian_comb.pkl")

cg_values = np.linspace(-width//2, width//2, 100)  # Adjust range as needed
ctg_values = np.linspace(-width//2, width//2, 100)  # Adjust range as needed

# Initialize a 2D grid for chi-squared values
chi_squared_grid = np.zeros((len(cg_values), len(ctg_values)))
nll_grid_pt = np.zeros((len(cg_values), len(ctg_values)))
nll_grid_nn = np.zeros((len(cg_values), len(ctg_values)))

for i, cg in enumerate(cg_values):
    for j, ctg in enumerate(ctg_values):
        chi_squared_grid[i][j] = get_chi_squared(np.ones(5), cg, ctg, hessian_comb, second_order=True)
        nll_grid_pt[i][j] = calc_NLL_comb(comb_hist, mu_c(c_g=cg, c_tg=ctg, a_cgs=a_cgs,a_ctgs=a_ctgs,b_cg_cgs=b_cg_cgs,b_ctg_ctgs=b_ctg_ctgs,b_cg_ctgs=b_cg_ctgs,second_order=True), "ttH")
        nll_grid_nn[i][j] = calc_nll_simple(hists, mu_c(c_g=cg, c_tg=ctg, a_cgs=a_cgs,a_ctgs=a_ctgs,b_cg_cgs=b_cg_cgs,b_ctg_ctgs=b_ctg_ctgs,b_cg_ctgs=b_cg_ctgs,second_order=True), "ttH")
#print(min(nll_grid_pt)+1, min(nll_grid_pt)+4)
fig, ax = plt.subplots(ncols=3, nrows=1,figsize=(20, 8))

cg_grid, ctg_grid = np.meshgrid(cg_values, ctg_values)  # Create grid for plotting

ax[0].contourf(cg_grid, ctg_grid, chi_squared_grid.T, levels=50, cmap='viridis')  # Transpose chi_squared to match grid
contour_plot = ax[0].contour(cg_grid, ctg_grid, chi_squared_grid.T, levels=[2.3, 5.99], colors=['yellow', 'green'], linestyles=['--', '-'])
ax[0].clabel(contour_plot, fmt={2.3: '68%', 5.99: '95%'}, inline=True, fontsize=20)  # Add labels to the contours
ax[0].scatter(best_cg_chi, best_ctg_chi, color='red', label='Minimum $\chi^2$', zorder=5)

# Plot the second contour plot
ax[1].contourf(cg_grid, ctg_grid, nll_grid_pt.T, levels=50, cmap='viridis')  # Transpose chi_squared to match grid
contour2 = ax[1].contour(cg_grid, ctg_grid, nll_grid_pt.T, levels=[np.min(nll_grid_pt)+1, np.min(nll_grid_pt)+4], colors=['yellow', 'green'], linestyles=['--', '-'])
ax[1].clabel(contour2, fmt={np.min(nll_grid_pt)+1: '68%', np.min(nll_grid_pt)+4: '95%'}, inline=True, fontsize=20)
ax[1].scatter(cg_fit, ctg_fit, color='red', label='Minimum NLL', zorder=5)


# Plot the third contour plot
contour3 = ax[2].contour(cg_grid, ctg_grid, nll_grid_nn.T, levels=[np.min(nll_grid_nn)+1, np.min(nll_grid_nn)+4], colors=['yellow', 'green'], linestyles=['--', '-'])
ax[2].clabel(contour3, fmt={np.min(nll_grid_nn)+1: '68%', np.min(nll_grid_nn)+4: '95%'}, inline=True, fontsize=20)
ax[2].contourf(cg_grid, ctg_grid, nll_grid_nn.T, levels=50, cmap='viridis')  # Transpose chi_squared to match grid
ax[2].scatter(pt_cg_fit, pt_ctg_fit, color='red', label='Minimum NLL', zorder=5)


# Add colorbar
cbar = fig.colorbar(ax[0].collections[0], ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label(r"$\chi^2$")

fig.supxlabel(r"Wilson coefficient $c_{g}$")
fig.supylabel(r"Wilson coefficient $c_{tg}$")

ax[0].set_title(r"Contour Plot of $\chi^2$")
ax[1].set_title(r"Contour Plot of $\mathrm{NLL}_{pt}$")
ax[2].set_title(r"Contour Plot of $\mathrm{NLL}_{nn}$")

ax[0].legend(frameon=True, edgecolor='black', loc='best')
ax[1].legend(frameon=True, edgecolor='black', loc='best')
ax[2].legend(frameon=True, edgecolor='black', loc='best')

#plt.grid()
plt.show()