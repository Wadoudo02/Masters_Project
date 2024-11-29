import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

sample_path = "/Users/wadoudcharbak/Downloads/Pass1"
plot_path = "/Users/wadoudcharbak/Downloads/plots"

vars_plotting_dict = {
    "plot_weight": [50, (0, 5), False, "Plot weight"],
    "mass_sel": [80, (100, 180), False, "$m_{\\gamma\\gamma}$ [GeV]"],
    "minIDMVA-with-cut": [50, (0, 1), False, "Minimum IDMVA (with cut)"],
    "n_jets": [10, (0, 10), False, "Number of jets"],
    "lead_pixelSeed": [2, (0, 2), False, "Lead pixel seed"],
    "sublead_pixelSeed": [2, (0, 2), False, "Sublead pixel seed"],
    "deltaR": [50, (0, 5), False, "$\\Delta R$"],
    "delta_eta_gg": [50, (0, 5), False, "$\\Delta \\eta_{\\gamma\\gamma}$"],
    "delta_phi_gg": [50, (0, 5), False, "$\\Delta \\phi_{\\gamma\\gamma}$"],
    "delta_eta_jj": [50, (0, 5), False, "$\\Delta \\eta_{jj}$"],
    "delta_phi_jj": [50, (0, 5), False, "$\\Delta \\phi_{jj}$"],
    "dijet_mass": [100, (0, 300), False, "Dijet mass [GeV]"],
    "delta_phi_gg_jj": [50, (0, 5), False, "$\\Delta \\phi_{\\gamma\\gamma, jj}$"],
    "min_delta_R_j_g": [50, (0, 5), False, "Minimum $\\Delta R(j, \\gamma)$"],
    "lead_pt/mass": [50, (0, 5), False, "Lead $p_T/m$"],
    "sublead_pt/mass": [50, (0, 5), False, "Sublead $p_T/m$"],
    "pt/mass": [50, (0, 5), False, "$p_T/m$"],
    "rapidity": [50, (0, 10), False, "Rapidity"],
    "lead_eta": [50, (0, 10), False, "Lead $\\eta$"],
    "sublead_eta": [50, (0, 10), False, "Sublead $\\eta$"],
    "lead_mvaID": [50, (0, 1), False, "Lead MVA ID"],
    "sublead_mvaID": [50, (0, 1), False, "Sublead MVA ID"],
    "lead_phi": [50, (0, 10), False, "Lead $\\phi$"],
    "sublead_phi": [50, (0, 10), False, "Sublead $\\phi$"],
    "lead_pt": [50, (0, 200), False, "Lead $p_T$ [GeV]"],
    "sublead_pt": [50, (0, 200), False, "Sublead $p_T$ [GeV]"],
    "lead_r9": [50, (0, 1), False, "Lead $r9$"],
    "sublead_r9": [50, (0, 1), False, "Sublead $r9$"],
    "sigma_m_over_m": [50, (0, 1), False, "$\\sigma_m/m$"],
    "lead_hoe": [50, (0, 0.1), False, "Lead $H/E$"],
    "sublead_hoe": [50, (0, 0.1), False, "Sublead $H/E$"],
    "PVScore": [50, (0, 1), False, "Primary vertex score"],
    "MET_pt": [50, (0, 200), False, "Missing $E_T$ [GeV]"],
    "MET_phi": [50, (0, 10), False, "Missing $\\phi$"],
    "HT": [50, (0, 300), False, "H$_T$ [GeV]"],
    
    # Jet variables
    "j0_pt": [50, (0, 200), False, "Jet 0 $p_T$ [GeV]"],
    "j1_pt": [50, (0, 200), False, "Jet 1 $p_T$ [GeV]"],
    "j2_pt": [50, (0, 200), False, "Jet 2 $p_T$ [GeV]"],
    "j3_pt": [50, (0, 200), False, "Jet 3 $p_T$ [GeV]"],
    "j0_eta": [50, (0, 10), False, "Jet 0 $\\eta$"],
    "j1_eta": [50, (0, 10), False, "Jet 1 $\\eta$"],
    "j2_eta": [50, (0, 10), False, "Jet 2 $\\eta$"],
    "j3_eta": [50, (0, 10), False, "Jet 3 $\\eta$"],
    "j0_phi": [50, (0, 10), False, "Jet 0 $\\phi$"],
    "j1_phi": [50, (0, 10), False, "Jet 1 $\\phi$"],
    "j2_phi": [50, (0, 10), False, "Jet 2 $\\phi$"],
    "j3_phi": [50, (0, 10), False, "Jet 3 $\\phi$"],
    "j0_btagB": [50, (0, 1), False, "Jet 0 b-tag B score"],
    "j1_btagB": [50, (0, 1), False, "Jet 1 b-tag B score"],
    "j2_btagB": [50, (0, 1), False, "Jet 2 b-tag B score"],
    "j3_btagB": [50, (0, 1), False, "Jet 3 b-tag B score"],
    
    # Electron variables
    "Ele0_pt": [50, (0, 200), False, "Electron 0 $p_T$ [GeV]"],
    "Ele1_pt": [50, (0, 200), False, "Electron 1 $p_T$ [GeV]"],
    "Ele0_eta": [50, (0, 10), False, "Electron 0 $\\eta$"],
    "Ele1_eta": [50, (0, 10), False, "Electron 1 $\\eta$"],
    "Ele0_phi": [50, (0, 10), False, "Electron 0 $\\phi$"],
    "Ele1_phi": [50, (0, 10), False, "Electron 1 $\\phi$"],
    "Ele0_charge": [3, (-1, 1), False, "Electron 0 charge"],
    "Ele1_charge": [3, (-1, 1), False, "Electron 1 charge"],
    "Ele0_id": [50, (0, 1), False, "Electron 0 ID"],
    "Ele1_id": [50, (0, 1), False, "Electron 1 ID"],
    "n_electrons": [10, (0, 10), False, "Number of electrons"],
    
    # Muon variables
    "Muo0_pt": [50, (0, 200), False, "Muon 0 $p_T$ [GeV]"],
    "Muo1_pt": [50, (0, 200), False, "Muon 1 $p_T$ [GeV]"],
    "Muo0_eta": [50, (0, 10), False, "Muon 0 $\\eta$"],
    "Muo1_eta": [50, (0, 10), False, "Muon 1 $\\eta$"],
    "Muo0_phi": [50, (0, 10), False, "Muon 0 $\\phi$"],
    "Muo1_phi": [50, (0, 10), False, "Muon 1 $\\phi$"],
    "Muo0_charge": [3, (-1, 1), False, "Muon 0 charge"],
    "Muo1_charge": [3, (-1, 1), False, "Muon 1 charge"],
    "Muo0_id": [50, (0, 1), False, "Muon 0 ID"],
    "Muo1_id": [50, (0, 1), False, "Muon 1 ID"],
    "n_muons": [10, (0, 10), False, "Number of muons"],
    
    # Tau variables
    "Tau0_pt": [50, (0, 200), False, "Tau 0 $p_T$ [GeV]"],
    "Tau1_pt": [50, (0, 200), False, "Tau 1 $p_T$ [GeV]"],
    "Tau0_eta": [50, (0, 10), False, "Tau 0 $\\eta$"],
    "Tau1_eta": [50, (0, 10), False, "Tau 1 $\\eta$"],
    "Tau0_phi": [50, (0, 10), False, "Tau 0 $\\phi$"],
    "Tau1_phi": [50, (0, 10), False, "Tau 1 $\\phi$"],
    "Tau0_charge": [3, (-1, 1), False, "Tau 0 charge"],
    "Tau1_charge": [3, (-1, 1), False, "Tau 1 charge"],
    "n_taus": [10, (0, 10), False, "Number of taus"],
    
    # Lepton variables
    "Lep0_pt": [50, (0, 200), False, "Lepton 0 $p_T$ [GeV]"],
    "Lep1_pt": [50, (0, 200), False, "Lepton 1 $p_T$ [GeV]"],
    "Lep0_eta": [50, (0, 10), False, "Lepton 0 $\\eta$"],
    "Lep1_eta": [50, (0, 10), False, "Lepton 1 $\\eta$"],
    "Lep0_phi": [50, (0, 10), False, "Lepton 0 $\\phi$"],
    "Lep1_phi": [50, (0, 10), False, "Lepton 1 $\\phi$"],
    "Lep0_charge": [3, (-1, 1), False, "Lepton 0 charge"],
    "Lep1_charge": [3, (-1, 1), False, "Lepton 1 charge"],
    "Lep0_flav": [5, (0, 5), False, "Lepton 0 flavour"],
    "Lep1_flav": [5, (0, 5), False, "Lepton 1 flavour"],
    "n_leptons": [10, (0, 10), False, "Number of leptons"],
    
    # Additional variables
    "cosDeltaPhi": [50, (0, 1), False, "Cosine of $\\Delta \\phi$"],
    "deltaPhiJ0GG": [50, (0, 10), False, "$\\Delta \\phi(J0, \\gamma\\gamma)$"],
    "deltaPhiJ1GG": [50, (0, 10), False, "$\\Delta \\phi(J1, \\gamma\\gamma)$"],
    "deltaPhiJ2GG": [50, (0, 10), False, "$\\Delta \\phi(J2, \\gamma\\gamma)$"],
    "deltaEtaJ0GG": [50, (0, 10), False, "$\\Delta \\eta(J0, \\gamma\\gamma)$"],
    "deltaEtaJ1GG": [50, (0, 10), False, "$\\Delta \\eta(J1, \\gamma\\gamma)$"],
    "deltaEtaJ2GG": [50, (0, 10), False, "$\\Delta \\eta(J2, \\gamma\\gamma)$"],
    "centrality": [50, (0, 1), False, "Centrality"],
    "dilepton_mass": [50, (0, 300), False, "Dilepton mass [GeV]"],
    "deltaR_L0G0": [50, (0, 10), False, "$\\Delta R(L0, G0)$"],
    "deltaR_L0G1": [50, (0, 10), False, "$\\Delta R(L0, G1)$"],
    "deltaR_L1G0": [50, (0, 10), False, "$\\Delta R(L1, G0)$"],
    "deltaR_L1G1": [50, (0, 10), False, "$\\Delta R(L1, G1)$"],
    "theta_ll_gg": [50, (0, 10), False, "$\\theta_{ll, \\gamma\\gamma}$"],
    "cosThetaStar": [50, (0, 1), False, "Cosine of $\\theta^*$"],
    "deltaPhiMetGG": [50, (0, 10), False, "$\\Delta \\phi(MET, \\gamma\\gamma)$"],
    "minDeltaPhiJMET": [50, (0, 10), False, "Minimum $\\Delta \\phi(J, MET)$"],
    "pt_balance": [50, (0, 10), False, "$p_T$ balance"],
    "helicity_angle": [50, (0, 1), False, "Helicity angle"]
}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Useful function definitions
# Function to extract 2NLL from array of NLL values
def TwoDeltaNLL(x):
    x = np.array(x)
    return 2*(x-x.min())


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
    num_truth_cats = conf_matrix.shape[1]  # Number of truth categories, based on confusion matrix
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
                        scaled_yield = bin_yields[bin_idx] * conf_matrix[j, truth_cat_idx]
                        combined_histogram[f"{signal}_{truth_cat_idx}"][combined_bin_idx] += scaled_yield
                    else:
                        combined_histogram[proc][combined_bin_idx] += bin_yields[bin_idx]

    return combined_histogram


def calc_NLL(combined_histogram, mus, signal='ttH'):
    """
    Calculate the NLL using the combined 25-bin histogram with variable \mu parameters.

    Parameters:
        combined_histogram (dict): Combined histogram for each process across 25 bins.
        mus (list or array): Signal strength modifiers, one for each truth category.
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
        #breakpoint()
        for proc, yields in combined_histogram.items():
            if signal in proc:
                # Extract the truth category index from the signal label, e.g., "ttH_0"
                truth_cat_idx = int(proc.split('_')[1])
                mu = mus[truth_cat_idx]  # Apply the appropriate \mu for this truth category
                expected_total += mu * yields[bin_idx]
            else:
                expected_total += yields[bin_idx]

            observed_count += yields[bin_idx]  # Observed count from all processes in this bin

        # Avoid division by zero in log calculation
        expected_total = max(expected_total, 1e-10)
        
        # Calculate NLL contribution for this bin
        NLL_total += observed_count * np.log(expected_total) - expected_total

    return -NLL_total



def add_val_label(val):
    return "$%.2f^{+%.2f}_{-%.2f}$"%(val[0],abs(val[1]),abs(val[2]))

def find_crossings(graph, yval, spline_type="cubic", spline_points=1000, remin=True, return_all_intervals=False):
    #breakpoint()
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
    #bestfit = bestfit[0] # NOT SURE WHY THIS NEEDS TO HAPPEN
    #print(bestfit)

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
        if (interval['lo']<=bestfit)&(interval['hi']>=bestfit): interval['contains_bf'] = True

    for interval in intervals:
        if interval['contains_bf']:
            val = (bestfit, interval['hi']-bestfit, interval['lo']-bestfit)

    if return_all_intervals:
        return val, intervals
    else:
        return val

def calc_Hessian(combined_histogram, mus, signal='ttH'):
    """
    Calculate the Hessian matrix for signal strength parameters.
    
    Parameters:
        combined_histogram (dict): Combined histogram for each process across bins.
        mus (list or array): Signal strength modifiers for each truth category.
        signal (str): The signal process (default 'ttH').
    
    Returns:
        numpy.ndarray: Hessian matrix for signal strength parameters.
    """
    # Determine the number of signal truth categories
    signal_categories = [proc for proc in combined_histogram.keys() if proc.startswith(signal + '_')]
    num_categories = len(signal_categories)
    
    # Initialize Hessian matrix
    Hessian = np.zeros((num_categories, num_categories))
    
    # Total number of bins
    num_bins = len(next(iter(combined_histogram.values())))  # Assume all histograms have the same length
    
    # Calculate Hessian matrix elements
    for i in range(num_categories):
        for m in range(num_categories):
            hessian_sum = 0.0
            
            for bin_idx in range(num_bins):
                # Get background contribution for this bin
                background = sum(
                    combined_histogram[proc][bin_idx]
                    for proc in combined_histogram
                    if not proc.startswith(signal + '_')
                )
                
                # Signal yields for this bin
                s_i = combined_histogram[signal_categories[i]][bin_idx]
                s_m = combined_histogram[signal_categories[m]][bin_idx]
                
                # Expected total: scaled signal + background
                expected_total = sum(
                    mus[k] * combined_histogram[signal_categories[k]][bin_idx]
                    for k in range(num_categories)
                ) + background
                
                # Ensure positive expected total to avoid division by zero
                expected_total = max(expected_total, 1e-10)
                
                # Total observed count in this bin
                observed_count = sum(combined_histogram[proc][bin_idx] for proc in combined_histogram)
                
                # Hessian contribution for this bin
                hessian_sum += (observed_count * s_i * s_m) / (expected_total ** 2)
            
            Hessian[i, m] = hessian_sum
    
    return Hessian

def covariance_to_correlation(cov_matrix):
    # Get the standard deviations (square root of diagonal elements)
    std_devs = np.sqrt(np.diag(cov_matrix))
    # Outer product of standard deviations to create the normalization matrix
    normalization_matrix = np.outer(std_devs, std_devs)
    # Element-wise division to compute the correlation matrix
    correlation_matrix = cov_matrix / normalization_matrix
    # Fill diagonal with 1s to ensure it's properly normalized
    np.fill_diagonal(correlation_matrix, 1.0)
    return correlation_matrix

def plot_matrix(matrix, x_labels=None, y_labels=None, title=None, cmap='viridis', colorbar=False):
    """
    Plots a given matrix with values labeled to 2 decimal places and optional x and y axis labels and a title.

    Parameters:
        matrix (2D array-like): The matrix to be plotted.
        x_labels (list, optional): Labels for the x-axis.
        y_labels (list, optional): Labels for the y-axis.
        title (str, optional): Title of the plot.
        cmap (str, optional): Colormap for the plot. Default is 'viridis'.
        colorbar (bool, optional): Whether to include a colorbar. Default is True.
    """
    matrix = np.array(matrix)  # Ensure the input is a NumPy array
    
    if colorbar:
        plt.figure(figsize=(10, 8))
    else:
        plt.figure(figsize=(8, 8))
    
    
    plt.imshow(matrix, cmap=cmap, aspect='auto')
    
    if colorbar:
        plt.colorbar()

    # Annotate the values in the matrix
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, f"{matrix[i, j]:.2f}", ha='center', va='center', color='white' if matrix[i, j] < np.max(matrix) / 2 else 'black')

    if x_labels is not None:
        plt.xticks(ticks=np.arange(matrix.shape[1]), labels=x_labels, rotation=90)
    else:
        plt.xticks([])  # Remove ticks if no labels provided

    if y_labels is not None:
        plt.yticks(ticks=np.arange(matrix.shape[0]), labels=y_labels)
    else:
        plt.yticks([])  # Remove ticks if no labels provided

    if title:
        plt.title(title)
    
    plt.tight_layout()
    plt.show()


def plot_combined_histogram(combined_histogram, categories, mass_bins=5):
    """
    Plot a 25-bin stacked histogram with bins labeled according to their respective categories.

    Parameters:
        combined_histogram (dict): Combined histogram with contributions from all processes.
        categories (list): List of category labels corresponding to each group of bins.
        mass_bins (int): Number of bins per category (default 5).
    """
    num_bins = len(categories) * mass_bins  # Total number of bins
    processes = list(combined_histogram.keys())  # Processes to plot
    bin_indices = np.arange(num_bins)  # X-axis bin indices

    # Extract contributions for each process
    contributions = np.array([combined_histogram[proc] for proc in processes])
    
    # Use a colormap to assign unique colors to each process
    cmap = get_cmap("tab20c")  # Use a larger color palette for distinct colors
    colors = [cmap(i / len(processes)) for i in range(len(processes))]
    
    # Create the stacked bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(num_bins)  # Tracks the cumulative height of the bars

    for i, proc in enumerate(processes):
        ax.bar(bin_indices, contributions[i], label=proc, bottom=bottom, color=colors[i])
        bottom += contributions[i]

    # Create custom x-axis labels
    x_labels = [f"{category}" if i % mass_bins == 2 else "" for i, category in enumerate(np.repeat(categories, mass_bins))]
    ax.set_xticks(bin_indices)
    ax.set_xticklabels(x_labels, rotation=0, fontsize=20)

    # Add vertical separators between categories for clarity
    for i in range(1, len(categories)):
        ax.axvline(i * mass_bins - 0.5, color='black', linestyle='--', linewidth=0.5)

    # Customize the plot
    ax.set_ylabel("Yields", fontsize=15)
    ax.set_xlabel("Categories", fontsize=15)
    ax.set_title("Combined Histogram: Process Contributions", fontsize=24)

    # Adjust the legend
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.4, 0.8), # Positioned (x%, y%)
        fontsize=10,
        ncol=2,
        title="Processes",
        title_fontsize=10,
        frameon=True
    )

    plt.tight_layout()
    plt.show()