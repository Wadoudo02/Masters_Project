import numpy as np
from scipy.interpolate import interp1d

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

def calc_NLL(hists, mus, conf_matrix, signal='ttH', mass_bins=5):
    """
    Calculate the NLL using 5 signal strength modifiers (mus), with confusion matrix integration.
    
    Parameters:
        hists (dict): Observed histogram data with categories.
        mus (list or array): Signal strength modifiers, one for each truth category (length 5).
        conf_matrix (ndarray): Confusion matrix for adjusting expected yields.
        signal (str): Name of the signal process (default 'ttH').
        mass_bins (int): Number of mass bins.
        
    Returns:
        float: Total NLL.
    """
    # Ensure conf_matrix is a numpy array for indexing
    conf_matrix = np.array(conf_matrix)
    NLL_total = 0.0

    # Loop over reconstructed categories (j)
    for j, (recon_cat, yields) in enumerate(hists.items()):
        e = np.zeros(mass_bins)
        n = np.zeros(mass_bins)

        # Convert `recon_cat` to an integer if it’s not already
        recon_cat_idx = j  # or use mapping if `recon_cat` is not numeric
        
        # Loop over processes (signal or background)
        for proc, bin_yields in yields.items():
            bin_yields = np.clip(bin_yields, 0, None)  # Ensure no negative yields

            if proc == signal:
                # Sum over truth categories (i) for signal contributions
                for truth_cat_idx in range(5):
                    mu = mus[truth_cat_idx]
                    # Ensure `truth_cat` and `recon_cat_idx` are valid for conf_matrix
                    e += mu * np.array(bin_yields) * conf_matrix[recon_cat_idx][truth_cat_idx]
            else:
                e += bin_yields

            n += bin_yields

        # Calculate NLL contribution for this category
        e = np.where(e > 0, e, 1e-10)
        NLL_total += np.sum(n * np.log(e) - e)

    return -NLL_total




def add_val_label(val):
    return "$%.2f^{+%.2f}_{-%.2f}$"%(val[0],abs(val[1]),abs(val[2]))

def find_crossings(graph, yval, spline_type="cubic", spline_points=1000, remin=True, return_all_intervals=False):

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





def calc_Hessian(hists, mus, conf_matrix, signal='ttH', mass_bins=5):
    """
    Calculate the Hessian matrix of the NLL with respect to signal strength parameters (mus).
    
    Parameters:
        hists (dict): Observed histogram data with categories.
        mus (list or array): Signal strength modifiers, one for each truth category.
        conf_matrix (ndarray): Confusion matrix for adjusting expected yields.
        signal (str): Name of the signal process (default 'ttH').
        mass_bins (int): Number of mass bins.
        
    Returns:
        ndarray: A diagonal Hessian matrix of the second derivatives of the NLL, with size based on the number of truth categories.
    """
    # Ensure conf_matrix is a numpy array for indexing
    conf_matrix = np.array(conf_matrix)
    num_categories = conf_matrix.shape[1]  # Number of truth categories
    hessian = np.zeros((num_categories, num_categories))  # Initialize Hessian matrix
    
    # Loop over each truth category to compute diagonal elements H_ii
    for i in range(num_categories):
        H_ii = 0.0  # Diagonal element for the i-th truth category
        
        # Loop over reconstructed categories (j)
        for j, (recon_cat, yields) in enumerate(hists.items()):
            # Convert `recon_cat` to an integer if it’s not already
            # recon_cat_idx = j  
            
            # Loop over processes (signal or background)
            for proc, bin_yields in yields.items():
                bin_yields = np.clip(bin_yields, 0, None)  # Ensure no negative yields

                if proc == signal:
                    # Apply only for the i-th truth category in this loop
                    mu_i = mus[i]
                    
                    # Calculate expected yield `e_ijk` and observed yield `n`
                    s_ijk = np.array(bin_yields) * conf_matrix[j][i]  # Signal yield for i-th category
                    e_ijk = mu_i * s_ijk + bin_yields  # Total expected yield
                    n = np.array(bin_yields)  # Observed yield
                    
                    # Avoid division by zero in the Hessian calculation
                    e_ijk = np.where(e_ijk > 0, e_ijk, 1e-10)
                    
                    # Calculate H_ii based on the provided formula
                    H_ii += np.sum(s_ijk**2 * n / (e_ijk**2))
        
        # Store the diagonal element in the Hessian matrix
        hessian[i, i] = H_ii

    return hessian