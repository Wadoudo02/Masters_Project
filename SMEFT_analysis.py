import matplotlib.pyplot as plt
import numpy as np
from SMEFT_classification import *
from SMEFT_utils import *

#Extract relevant columns from overall df
cats = [0, 0.4, 0.5, 0.6, 0.7,1]

dfs = get_dfs(new_sample_path)
for proc, df in dfs.items():
    if proc=="ttH":
        df=ttH_df
    new_df = pd.concat([df[feature] for feature in special_features]+[df["mass_sel"],df["plot_weight"]], axis=1)
    dfs[proc] = torch.tensor(new_df.to_numpy(), dtype=torch.float32)
#%%
#Get predictions for all events
dfs_preds = {}
dfs_cats = {}

#Loading model
input_dim = len(special_features)
hidden_dim = [256, 64, 32, 16, 16, 8]
model = ComplexNN(input_dim, hidden_dim, 1)
model.load_state_dict(torch.load("C:/Users/avigh/Documents/MSci_proj/Avighna/Masters_Project/saved_models/model.pth"))
model.eval()

for proc, df in dfs.items():
    #print(proc, df)
    mass,weight = df[:,-2],df[:,-1]
    df = df[:,:-2]
    #new_df = torch.tensor(df,dtype=torch.float32)
    probs = eval_in_batches(model, df)
    #probs = model(df)

    dfs_preds[proc] = [probs, mass,weight]

#%%
#Split events by category

with_back=True
order = ["background","ggH","VBF", "VH","ttH"]
for proc in order:
    preds,mass, weights = dfs_preds[proc]
    dfs_cats[proc]={"mass":[],"weights":[]}
    for i in range(1, len(cats)):
        bools = ((cats[i-1]<preds) & (preds<cats[i])).squeeze()
        #print(i, bools.unique())
        #dfs_cats[proc]["events"].append(df[bools.numpy()])
        dfs_cats[proc]["weights"].append(weights[bools.numpy()])
        dfs_cats[proc]["mass"].append(mass[bools.numpy()])

#Over number of cats
fig, ax = plt.subplots(ncols=len(dfs_cats["ggH"]["mass"]),figsize=(20, 5))    
for i in range(len(dfs_cats["ggH"]["mass"])):
    for proc in order:
        if not with_back and proc=="background":
            continue
        #print(proc, "cat: ", i, dfs_cats[proc]["mass"][i].shape)
        ax[i].hist(dfs_cats[proc]["mass"][i],weights=dfs_cats[proc]["weights"][i], bins=50, alpha=0.5, label=proc, )
        ax[i].legend()
        ax[i].set_title(f"Category {i}")
        ax[i].set_xlabel("mass (GeV)")
        ax[i].set_ylabel("Events")



# %%
