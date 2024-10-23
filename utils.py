import numpy as np
from scipy.interpolate import interp1d

sample_path = "/vols/cms/jl2117/icrf/hgg/MSci_projects/samples/Pass0"
plot_path = "all_plots"
analysis_path = "stat_anal"

vars_plotting_dict = {
    # var_name : [nbins, range, log-scale]
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
def TwoDeltaNLL(x):
    x = np.array(x)
    return 2*(x-x.min())

def calc_NLL(hists, mu, signal='ttH'):
    NLL_vals = []
    # Loop over categories
    for yields in hists.values():
        n_bins = len(list(yields.values())[0])
        e = np.zeros(n_bins)
        n = np.zeros(n_bins)
        for proc, bin_yields in yields.items():
            if proc == signal:
                e += mu*bin_yields
            else:
                e += bin_yields
            n += bin_yields
        nll = e-n*np.log(e)
        NLL_vals.append(nll)
    return np.array(NLL_vals).sum()

def add_val_label(val):
    return "$%.2f^{+%.2f}_{-%.2f}$"%(val[0],abs(val[1]),abs(val[2]))

def find_crossings(graph, yval, spline_type="cubic", spline_points=1000, remin=True, return_all_intervals=False):
    #print("graph 1: NLL vals = ", graph[1])
    
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
