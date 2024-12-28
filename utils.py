import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import mplhep as hep
from categorisation import get_pt_cat
plt.style.use(hep.style.CMS)

sample_path_old = "/vols/cms/jl2117/icrf/hgg/MSci_projects/samples/Pass0"
sample_path = "/vols/cms/jl2117/icrf/hgg/MSci_projects/samples/Pass1"
new_sample_path="/vols/cms/jl2117/icrf/hgg/MSci_projects/samples/Pass2"


plot_path = "all_plots"
analysis_path = "stat_anal"
col_name = "_sel"

total_lumi = 7.9804
target_lumi = 300

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

# Processes to plot
procs = {
    "background" : ["Background", "black"],
    "ttH" : ["ttH (x10)", "mediumorchid"],
    "ggH" : ["ggH (x10)", "cornflowerblue"],
    "VBF" : ["VBF (x10)", "green"],
    "VH" : ["VH (x10)", "brown"]
}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Useful function definitions

def order_data(*args, order):
    ans = []
    for data in args:
        ordered_data = [data[i] for i in order]
        ans.append(ordered_data)
    return ans

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

def plot_comb_hist(comb_hist, inc_background=True):
    num_bins = 25
    #bin_edges = np.array([*np.linspace(120, 130, 6)]*5)
    bin_edges = np.linspace(0,26, 26)
    print(bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
                   
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = sns.color_palette("husl", len(comb_hist))
    bottom = np.zeros(num_bins)
    for  i, (cat, hist_values) in enumerate(comb_hist.items()):
        if not inc_background and cat == "background":
            continue
        #ax.bar(bin_edges,comb_hist[cat], label=cat, color=colors[cat])
        ax.bar(bin_centers, comb_hist[cat], width=bin_edges[1] - bin_edges[0], bottom=bottom, label=cat, color=colors[i],edgecolor='black')
        bottom+=comb_hist[cat]

    # Add labels and title
    ax.set_xlabel('pt category (GeV/c)')
    ax.set_ylabel('Events')
    ax.set_title('Combined Histogram')
    ax.legend(title='Categories')

    group_labels = ['0-60', '60-120', '120-180', '180-240', '240-300']
    bin_labels = ['120-122', '122-124', '124-126', '126-128', '128-130']
    custom_labels = []
    # for group in group_labels:
    #     custom_labels.extend([f'{group}\n{bin_label}' for bin_label in bin_labels])

    ax.set_xticks(bin_centers[2:][::5])
    ax.set_xticklabels(group_labels)

    # Show the plot
    plt.tight_layout()
    plt.show()    

def plot_ind_cats(comb_hist):
    bin_edges = np.arange(len(next(iter(comb_hist.values()))))
    fig, axes = plt.subplots(5,2,figsize=(10, 7))
    for cat, hist in comb_hist.items():
        ax = axes[cat//5][cat%5]
        ax.bar(bin_edges,hist, label=cat)
        ax.set_title(f"Cat: {cat}")
        ax.set_ylabel("Events")
        ax.show()
def plot_diphoton_mass(dfs, cats_unique):
    # Plot diphoton mass distribution in each category
    v = "mass"+col_name
    fig, axes = plt.subplots(1,5, figsize=(30,7.5))
    for cat in cats_unique:
        ax = axes[cat]
        print(f" --> Plotting: {v} in cat{cat}")
        nbins, xrange, is_log_scale, sanitized_var_name = vars_plotting_dict[v]
        # Loop over procs and add histogram
        for proc in procs.keys():
            label, color = procs[proc]

            cat_mask = dfs[proc]['category']==cat

            x = np.array(dfs[proc][v][cat_mask])

            # Event weight
            w = np.array(dfs[proc]['plot_weight'])[cat_mask]

            ax.hist(x, nbins, xrange, label=label, histtype='step', weights=w, edgecolor=color, lw=2)

        ax.set_xlabel(sanitized_var_name)
        ax.set_ylabel("Events")

        if is_log_scale:
            ax.set_yscale("log")

        ax.legend(loc='best')

        hep.cms.label("", com="13.6", lumi=target_lumi, lumi_format="{0:.2f}", ax=ax)

        plt.tight_layout()
        ext = f"_cat{cat}"
        #fig.savefig(f"{plot_path}/{v}{ext}.pdf", bbox_inches="tight")
        fig.savefig(f"{analysis_path}/{v}{ext}.png", bbox_inches="tight")
        #ax.cla()

#[i,j] element of hessian
def get_hessian(p, q, hists, mu_vals, conf_matrix, signal="ttH"):
    cat_vals = []
    for cat, yields in hists.items():
        n_bins = len(list(yields.values())[0])
        n = np.zeros(n_bins)
        e = np.zeros(n_bins)

        #Loop over prod modes, gets me my sum over i
        for proc, bin_yields in yields.items():
            if proc == signal:
                for truth_cat in range(len(conf_matrix[0])):
                    #print(mu_vals[truth_cat], hists[truth_cat][signal], conf_matrix[cat][truth_cat])
                    e+=mu_vals[truth_cat]*hists[truth_cat][signal]*conf_matrix[cat][truth_cat]                
            else:
                e += bin_yields
            n += bin_yields
        

        s_i = conf_matrix[cat][p]*yields[signal]
        s_j = conf_matrix[cat][q]*yields[signal]
        cat_vals.append((s_i*s_j)/(e))
    return np.array(cat_vals).sum()


def get_hessian_comb(p, q, combined_histogram, mu_vals, signal ="ttH"):
    hess = 0
    #Iterate through recon cats
    for j in range(len(mu_vals)):
        expected_total=np.zeros(5)
        observed_count=np.zeros(5)
        #Iterate through truth cats
        for proc, yields in combined_histogram.items():
            if signal in proc:
                truth_cat_idx = int(proc.split('_')[1])
                mu = mu_vals[truth_cat_idx]
                expected_total += (mu * yields[j*5:(j*5 +5)])
            else:
                expected_total += (yields[j*5:(j*5 +5)])

            observed_count += yields[j*5:(j*5+5)]
        recon_j = (combined_histogram[signal+f"_{p}"][j*5:(j*5+5)] * combined_histogram[signal+f"_{q}"][j*5:(j*5+5)]) / expected_total
        hess+=recon_j.sum()
    return hess

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

        # print("lo ", interval["lo"])
        # print("hi ", interval["hi"])
        # print("bestfit: ", bestfit)

        if (interval['lo']<=bestfit)&(interval['hi']>=bestfit): interval['contains_bf'] = True

    for interval in intervals:
        if interval['contains_bf']:
            val = (bestfit, interval['hi']-bestfit, interval['lo']-bestfit)

    if return_all_intervals:
        return val, intervals
    else:
        return val


def get_conf_mat(data):
    #print(data.columns[data.columns[:4]=="HTXS"])
    data = data[data["pt-over-mass_sel"]==data["pt-over-mass_sel"]].reset_index(drop=True)
    data["plot_weight"] = data["plot_weight"]*(target_lumi/total_lumi)
    b_tag_scores = np.array(data[['j0_btagB_sel', 'j1_btagB_sel', 'j2_btagB_sel', 'j3_btagB_sel']])
    b_tag_scores = np.nan_to_num(b_tag_scores, nan=-1)
    max_b_tag_score = -1*np.sort(-1*b_tag_scores,axis=1)[:,0]
    second_max_b_tag_score = -1*np.sort(-1*b_tag_scores,axis=1)[:,1]
    data['max_b_tag_score'] = max_b_tag_score
    data['second_max_b_tag_score'] = second_max_b_tag_score


    #Applying selection cuts
    yield_before_sel = data['plot_weight'].sum()
    mask = data['n_jets_sel'] >= 2
    mask = mask & (data['max_b_tag_score'] > 0.7)
    mask = mask & (data['second_max_b_tag_score'] > 0.4)
    data = data[mask].reset_index(drop=True)
    yield_after_sel = data['plot_weight'].sum()
    eff = (yield_after_sel/yield_before_sel)*100
    print(f"N = {yield_before_sel:.2f} --> {yield_after_sel:.2f}, eff = {eff:.1f}%")
    bins = [0,60,120,200,300]
    #print(data)

    truth = data["HTXS_Higgs_pt_sel"]
    recon = data["pt-over-mass_sel"]*data["mass_sel"]

    truth_cat = get_pt_cat(truth, bins = bins)
    recon_cat = get_pt_cat(recon, bins = bins)
    #conf_matrix_hist, xedges, yedges = np.histogram2d(truth_cat, recon_cat, bins=[4, 4], weights=data["plot_weight"])
    #print(conf_matrix_hist)
    #6x6 conf matrix where x axis is truth dimension and y axis is recon dimension
    conf_mat =np.zeros((len(bins),len(bins)))

    for i in range(len(recon_cat)):
        #print(data["plot_weight"][i], recon_cat[i],truth_cat[i], conf_mat[recon_cat[i]][truth_cat[i]]+data["plot_weight"][i])
        conf_mat[recon_cat[i]][truth_cat[i]]=conf_mat[recon_cat[i]][truth_cat[i]]+data["plot_weight"][i] #Not working not sure why
        #print(conf_mat)
    #print(conf_mat)
    conf_mat_truth_prop = [[conf_mat[i][j]/sum(conf_mat[:,j]) for j in range(len(conf_mat[0]))] for i in range(len(conf_mat))]
    conf_mat_recon_prop = [[conf_mat[i][j]/sum(conf_mat[i]) for j in range(len(conf_mat[0]))] for i in range(len(conf_mat))]


    fig, ax = plt.subplots(ncols=3, figsize=(30, 10))

    labels = ["0-60", "60-120", "120-200", "200-300", "300-inf"]
    #plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt=".2f", cmap="Blues", cbar=True,
                xticklabels=labels, yticklabels=labels, ax=ax[0])

    # Add labels to the plot
    ax[0].set_ylabel("Reconstructed pt")
    ax[0].set_xlabel("True pt")
    ax[0].set_title("Weighted Confusion Matrix")
    #plt.savefig(f"{analysis_path}/conf.png")

    
    sns.heatmap(conf_mat_recon_prop, annot=True, fmt=".3f", cmap="Blues", cbar=True,
                xticklabels=labels, yticklabels=labels,ax=ax[1])

    # Add labels to the plot
    ax[1].set_ylabel("Reconstructed pt")
    ax[1].set_xlabel("True pt")
    ax[1].set_title("Weighted Confusion Matrix normalised by recon")
    #plt.savefig(f"{analysis_path}/conf_recon.png")


    #plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat_truth_prop, annot=True, fmt=".3f", cmap="Blues", cbar=True,
                xticklabels=labels, yticklabels=labels, ax = ax[2])

    # Add labels to the plot
    ax[2].set_ylabel("Reconstructed pt")
    ax[2].set_xlabel("True pt")
    ax[2].set_title("Weighted Confusion Matrix normalised by truth")
    #plt.savefig(f"{analysis_path}/conf_truth.png")
    plt.savefig(f"{analysis_path}/conf.png")

    return (conf_mat, conf_mat_truth_prop, conf_mat_recon_prop)
def plot_hcc(hessian, suptitle):
    fig, ax = plt.subplots(1,3, figsize=(10,7))
    fig.suptitle(suptitle)
    cov = get_cov(hessian)
    correlation = get_correlation_matrix(cov)
    show_matrix(hessian, "Hessian",ax[0])
    show_matrix(cov, "Covariance", ax[1])
    show_matrix(correlation, "Correlation", ax[2])
    plt.show()
    return (cov, correlation)
    

def get_cov(hessian):
    return np.linalg.inv(hessian)

def show_matrix(matrix, title, ax="plt"):
    #ax.figure(figsize=(8, 6))
    #plt.imshow(matrix, cmap='viridis', interpolation='none')
    if ax=="plt":
        sns.heatmap(matrix, annot=True, fmt=".4f", cmap='viridis', cbar=False, annot_kws={"size": 10})
        plt.title(title)
    else:
        sns.heatmap(matrix, annot=True, fmt=".4f", cmap='viridis', cbar=False, annot_kws={"size": 10}, ax=ax)
        ax.set_title(title, fontsize=10)
    #plt.xlabel('Column')
    #plt.ylabel('Row')
    #ax.show()

def get_correlation_matrix(cov_matrix):
    std_dev = np.sqrt(np.diag(cov_matrix))
    correlation_matrix = cov_matrix / np.outer(std_dev, std_dev)
    correlation_matrix[cov_matrix == 0] = 0  # Handle division by zero
    return correlation_matrix
def get_uncertainties(cov_matrix):
    return np.sqrt(np.diag(cov_matrix))
