import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize

sample_path_old = "/vols/cms/jl2117/icrf/hgg/MSci_projects/samples/Pass0"
sample_path = "/vols/cms/jl2117/icrf/hgg/MSci_projects/samples/Pass1"

plot_path = "all_plots"
analysis_path = "stat_anal"

vars_plotting_dict = {
    # var_name : [nbins, range, log-scale]
    "mass_sel" : [80, (100,180), False, "$m_{\\gamma\\gamma}$ [GeV]"],
    "mass" : [80, (100,180), False, "$m_{\\gamma\\gamma}$ [GeV]"],
    "n_jets" : [10, (0,10), False, "Number of jets"],
    "Muo0_pt" : [20, (0,100), False, "Lead Muon $p_T$ [GeV]"],
    "max_b_tag_score" : [50, (0,1), False, "Highest jet b-tag score"],
    "second_max_b_tag_score" : [50, (0,1), False, "Second highest jet b-tag score"],
    "deltaR" : [50, (0,5),False,"delta R"],
    "n_leptons":[50, (0,4), False, "Number of leptons"]
}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Useful function definitions
# Function to extract 2NLL from array of NLL values

def build_combined_histogram(hists, conf_matrix, signal='ttH', mass_bins = 5):
    """
    Builds a single 25-bin histogram by combining yields from all categories and scaling ttH yields
    according to the confusion matrix.

    Parameters:
        hists (dict): Observed histogram data for each reconstructed category.
        conf_matrix (ndarray): Confusion matrix for adjusting expected yields.
        signal (str): The signal process (default 'ttH').

    Returns:
        combined_histogram (dict): Combined histogram for each process, without \mu scaling.
        Note indexes 0,1,2,3,4 match the index of the labels ['0-60', '60-120', '120-200', '200-300', '>300']
    """
    num_reco_cats = len(hists)

    combined_bins = num_reco_cats * mass_bins
    #print(conf_matrix)
    num_truth_cats = len(conf_matrix[0])  # Number of truth categories, based on confusion matrix
    #breakpoint()
    # Initialize a dictionary for each process with the combined bins
    combined_histogram = {}
    for proc in hists[next(iter(hists))].keys():
        if proc == signal:
            # Create separate entries for each truth category of the signal
            for truth_cat_idx in range(num_truth_cats):
                combined_histogram[f"{signal}_{truth_cat_idx}"] = np.zeros(combined_bins)
        else:
            # Single entry for background or other processes
            combined_histogram[proc] = np.zeros(combined_bins)

    # Populate the combined histogram without any \mu scaling
    for j, (recon_cat, yields) in enumerate(hists.items()):
        for proc, bin_yields in yields.items():
            for truth_cat_idx in range(num_truth_cats):
                for bin_idx in range(mass_bins):
                    combined_bin_idx = j * mass_bins + bin_idx
                    if proc == signal:
                        # Adjust yield according to confusion matrix (but no \mu scaling)
                        scaled_yield = bin_yields[bin_idx] * conf_matrix[j][truth_cat_idx]
                        combined_histogram[f"{signal}_{truth_cat_idx}"][combined_bin_idx] += scaled_yield
                    else:
                        combined_histogram[proc][combined_bin_idx] += bin_yields[bin_idx]

    return combined_histogram

def TwoDeltaNLL(x):
    x = np.array(x)
    return 2*(x-x.min())

#Takes optional param cat which if provided only get NLL over that category
def calc_NLL(hists, mus, conf_matrix = [],signal='ttH'):
    NLL_vals = []
    # Loop over recon categories
    for cat, yields in hists.items():
        n_bins = len(list(yields.values())[0])
        e = np.zeros(n_bins)
        n = np.zeros(n_bins)
        #Loop over prod modes, truth bins so stuff inside log
        for proc, bin_yields in yields.items():
            if proc == signal:
                #Case where conf matrix provided
                if len(conf_matrix)!=0:
                    '''
                    For the particular recon category "cat" we sum over each truth category * the mu belonging to that category
                    which in this case is 1 for all categories other than the one we are looking at for which it is mu, as we 
                    are performing a frozen fit.
                    '''
                    for truth_cat in range(5):
                        e+=mus[truth_cat]*hists[truth_cat][signal]*conf_matrix[cat][truth_cat]
                else:
                    e+=mus[0]*bin_yields
                
            else:
                e += bin_yields
            n += bin_yields
        nll = e-n*np.log(e)
        NLL_vals.append(nll)
    #print(NLL_vals, np.array(NLL_vals).sum())
    return np.array(NLL_vals).sum()

#Taken from Wadoud
def calc_NLL_comb(combined_histogram, mus, signal='ttH'):
    """
    Calculate the NLL using the combined 25-bin histogram with variable \mu parameters.

    Parameters:
        combined_histogram (dict): Combined histogram for each process across 25 bins.
        mus (list or array): Signal strength modifiers, one for each truth category.
        conf_matrix (ndarray): Confusion matrix for adjusting expected yields.
        signal (str): The signal process (default 'ttH').

    Returns:
        float: Total NLL.
    """
    NLL_total = 0.0
    num_bins = len(next(iter(combined_histogram.values())))  # Total bins (should be 25)

    # Loop over each bin in the combined histogram
    for bin_idx in range(num_bins):
        expected_total = 0.0
        observed_count = 0.0

        for proc, yields in combined_histogram.items():
            if signal in proc:
                # Extract the truth category index from the signal label, e.g., "ttH_0"
                truth_cat_idx = int(proc.split('_')[1])
                #print(mus, truth_cat_idx)
                mu = mus[truth_cat_idx]  # Apply the appropriate \mu for this truth category
                expected_total += mu * yields[bin_idx]
            else:
                expected_total += yields[bin_idx]

            observed_count += yields[bin_idx]  # Observed count from all processes in this bin
        
        # Calculate NLL contribution for this bin
        NLL_total += observed_count * np.log(expected_total) - expected_total

    return -NLL_total

#[i,j] element of hessian
def get_hessian(i, j, hists, mu_vals, conf_matrix, signal="ttH"):
    cat_vals = []
    for cat, yields in hists.items():
        n_bins = len(list(yields.values())[0])
        n = np.zeros(n_bins)
        e = np.zeros(n_bins)

        #Loop over prod modes, gets me my sum over i
        for proc, bin_yields in yields.items():
            if proc == signal:
                for truth_cat in range(5):
                    #print(mu_vals[truth_cat], hists[truth_cat][signal], conf_matrix[cat][truth_cat])
                    e+=mu_vals[truth_cat]*hists[truth_cat][signal]*conf_matrix[cat][truth_cat]                
            else:
                e += bin_yields
            n += bin_yields
        

        s_i = conf_matrix[cat][i]*yields[signal]
        s_j = conf_matrix[cat][j]*yields[signal]
        cat_vals.append((s_i*s_j)/(e))
    return np.array(cat_vals).sum()

#Unfinished not sure
def get_hessian_comb(i, j, combined_histogram, mu_vals, conf_matrix, signal="ttH"):

    num_bins = len(next(iter(combined_histogram.values())))  # Total bins (should be 25)

    for bin_idx in range(num_bins):
        expected_total = 0.0
        observed_count = 0.0

        for proc, yields in combined_histogram.items():
            if signal in proc:
                # Extract the truth category index from the signal label, e.g., "ttH_0"
                truth_cat_idx = int(proc.split('_')[1])
                mu = mu_vals[truth_cat_idx]  # Apply the appropriate \mu for this truth category
                expected_total += mu * yields[bin_idx]
            else:
                expected_total += yields[bin_idx]

            observed_count += yields[bin_idx]  # Observed count from all processes in this bin
        
        # Calculate NLL contribution for this bin
        NLL_total += observed_count * np.log(expected_total) - expected_total

    
    cat_vals = []
    for cat,yields in hists.items():
        n_bins = len(list(yields.values())[0])
        n = np.zeros(n_bins)
        e = np.zeros(n_bins)

        #Loop over prod modes, gets me my sum over i
        for proc, bin_yields in yields.items():
            if proc == signal:
                for truth_cat in range(5):
                    #print(mu_vals[truth_cat], hists[truth_cat][signal], conf_matrix[cat][truth_cat])
                    e+=mu_vals[truth_cat]*hists[truth_cat][signal]*conf_matrix[cat][truth_cat]                
            else:
                e += bin_yields
            n += bin_yields
        

        s_i = conf_matrix[cat][i]*yields[signal]
        s_j = conf_matrix[cat][j]*yields[signal]
        cat_vals.append((s_i*s_j)/(e))
    return np.array(cat_vals).sum()

def add_val_label(val):
    return "$%.2f^{+%.2f}_{-%.2f}$"%(val[0],abs(val[1]),abs(val[2]))

def find_crossings(graph, yval, spline_type="cubic", spline_points=1000, remin=True, return_all_intervals=False):
    #print("graph 1: 2 delta NLL vals = ", graph[1])
    #print("when 0 ", graph[1]==0, graph[0], len(graph[0]), len(graph[1]))
    # Build spline
    f = interp1d(graph[0],graph[1],kind=spline_type)
    x_spline = np.linspace(graph[0].min(),graph[0].max(),spline_points)
    y_spline = f(x_spline)
    spline = (x_spline,y_spline)

    # Remin
    if remin:
        x,y = graph[0],graph[1]
        if y_spline.min() <= 0:
            y = y-y_spline.min()
            y_spline -= y_spline.min()
            # Add new point to graph
            x = np.append(x, x_spline[np.argmin(y_spline)])
            y = np.append(y, 0.)
            # Re-sort
            i_sort = np.argsort(x)
            x = x[i_sort]
            y = y[i_sort]
            graph = (x,y)

    # Extract bestfit
    bestfit = graph[0][graph[1]==0]
    #print(bestfit)
    #NO CLUE WHY BESTFIT HAS 2 VALUES SO ONLY TAKING THE FIRST ONE.
    bestfit=bestfit[0]
    #print("Bestfit", bestfit)
    crossings, intervals = [], []
    current = None

    for i in range(len(graph[0])-1):
        if (graph[1][i]-yval)*(graph[1][i+1]-yval) < 0.:
            # Find crossing as inverse of spline between two x points
            mask = (spline[0]>graph[0][i])&(spline[0]<=graph[0][i+1])
            f_inv = interp1d(spline[1][mask],spline[0][mask])

            # Find crossing point for catch when yval is out of domain of spline points (unlikely)
            if yval > spline[1][mask].max(): cross = f_inv(spline[1][mask].max())
            elif yval <= spline[1][mask].min(): cross = f_inv(spline[1][mask].min())
            else: cross = f_inv(yval)

            # Add information for crossings
            if ((graph[1][i]-yval) > 0.)&( current is None ):
                current = {
                    'lo':cross,
                    'hi':graph[0][-1],
                    'valid_lo': True,
                    'valid_hi': False
                }
            if ((graph[1][i]-yval) < 0.)&( current is None ):
                current = {
                    'lo':graph[0][0],
                    'hi':cross,
                    'valid_lo': False,
                    'valid_hi': True
                }
            if ((graph[1][i]-yval) < 0.)&( current is not None ):
                current['hi'] = cross
                current['valid_hi'] = True
                intervals.append(current)
                current = None

            crossings.append(cross)

    if current is not None:
        intervals.append(current)

    if len(intervals) == 0:
        current = {
            'lo':graph[0][0],
            'hi':graph[0][-1],
            'valid_lo': False,
            'valid_hi': False
        }
        intervals.append(current)

    for interval in intervals:
        interval['contains_bf'] = False

        print("lo ", interval["lo"])
        print("hi ", interval["hi"])
        print("bestfit: ", bestfit)

        if (interval['lo']<=bestfit)&(interval['hi']>=bestfit): interval['contains_bf'] = True

    for interval in intervals:
        if interval['contains_bf']:
            val = (bestfit, interval['hi']-bestfit, interval['lo']-bestfit)

    if return_all_intervals:
        return val, intervals
    else:
        return val
def get_pt_cat(data):
    conditions = [
    (data < 60),
    (data >= 60) & (data < 120),
    (data >= 120) & (data < 200), 
    (data >= 200) & (data < 300), 
    (data >= 300)     
    ]
    return np.select(conditions, [0,1,2,3,4])
def get_conf_mat(data):
    #print(data.columns[data.columns[:4]=="HTXS"])
    data = data[data["pt-over-mass_sel"]==data["pt-over-mass_sel"]].reset_index(drop=True)
    truth = data["HTXS_Higgs_pt_sel"]
    recon = data["pt-over-mass_sel"]*data["mass_sel"]

    truth_cat = get_pt_cat(truth)
    recon_cat = get_pt_cat(recon)
    
    #6x6 conf matrix where x axis is truth dimension and y axis is recon dimension
    conf_mat =np.zeros((5,5))

    for i in range(len(recon_cat)):
        #print(data["plot_weight"][i], recon_cat[i],truth_cat[i], conf_mat[recon_cat[i]][truth_cat[i]]+data["plot_weight"][i])
        conf_mat[recon_cat[i]][truth_cat[i]]=conf_mat[recon_cat[i]][truth_cat[i]]+data["plot_weight"][i] #Not working not sure why
        #print(conf_mat)
    #print(conf_mat)
    conf_mat_truth_prop = [[conf_mat[i][j]/sum(conf_mat[:,j]) for j in range(5)] for i in range(5)]
    conf_mat_recon_prop = [[conf_mat[i][j]/sum(conf_mat[i]) for j in range(5)] for i in range(5)]


    #print("conf matrix: ", conf_mat)
    #print("Conf matrix by proportion of truth: ", conf_mat_truth_prop)
    #print("Conf matrix by proportion of recon: ", conf_mat_recon_prop)
    labels = ["0-60", "60-120", "120-200", "200-300", "300-inf"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt=".2f", cmap="Blues", cbar=True,
                xticklabels=labels, yticklabels=labels)

    # Add labels to the plot
    plt.ylabel("Reconstructed pt")
    plt.xlabel("True pt")
    plt.title("Weighted Confusion Matrix")
    plt.savefig(f"{analysis_path}/conf.png")

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat_recon_prop, annot=True, fmt=".2f", cmap="Blues", cbar=True,
                xticklabels=labels, yticklabels=labels)

    # Add labels to the plot
    plt.ylabel("Reconstructed pt")
    plt.xlabel("True pt")
    plt.title("Weighted Confusion Matrix normalised by recon")
    plt.savefig(f"{analysis_path}/conf_recon.png")


    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat_truth_prop, annot=True, fmt=".2f", cmap="Blues", cbar=True,
                xticklabels=labels, yticklabels=labels)

    # Add labels to the plot
    plt.ylabel("Reconstructed pt")
    plt.xlabel("True pt")
    plt.title("Weighted Confusion Matrix normalised by truth")
    plt.savefig(f"{analysis_path}/conf_truth.png")

    return (conf_mat, conf_mat_truth_prop, conf_mat_recon_prop)

#data = pd.read_parquet(f"{sample_path}/ttH_processed_selected.parquet")

#get_conf_mat(data)

# Scans over 1 mu index and at each step minimises all others so gives best fit for that mu value
def profiled_NLL_fit(combined_histogram, conf_matrix, mu_idx):
    num_truth = len(conf_matrix[0])

    mu_vals = np.linspace(0, 3, 100)
    nll = []

    def calc_nll_fixed_mu(fixed, others):
        mus = np.array(others)
        mus = np.insert(mus, mu_idx, fixed)
        return calc_NLL_comb(combined_histogram, mus)
    
    for mu in mu_vals:
        #Generating ones for all but 1 mu
        guess = np.ones(num_truth-1)

        res = minimize(
            lambda other_mus: calc_nll_fixed_mu(mu, other_mus),
            guess,
            method="L-BFGS-B"
        )

        nll.append(res.fun)
    
    vals = find_crossings((mu_vals, TwoDeltaNLL(nll)), 1.)
    return (vals[0], nll)