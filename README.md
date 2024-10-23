# ttH EFT analysis (Run 3)

Repository to store scripts for the MSci project looking at constraining EFT operators using ttH in the diphoton decay channel.

The latest samples to begin the project are stored in:
```
/vols/cms/jl2117/icrf/hgg/MSci_projects/samples/Pass0
```

The current scripts are very basic. The following command used to plot the 1D variable distributions:
```
python3 plot_hist.py <comma-sep list of variables> <1/0>
```
where the first argument should be a comma-separated list of variables e.g. `mass,n_jets,Muo0_pt`, and the second argument should be 0 to plot the absolute number of events for each process, or 1 to plot the fractional number of events (area under histogram is unity). In the 0 setting, the signal histograms are multiplied by a factor of 10 to allow to compare with the much larger backgrounds. The variable plotting details are configured in `utils.py`.

The `plot_hist.py` script gives an example on how to construct new features, which are the highest and second-highest jet b-tagging scores. You can implement similar code to build new features. A useful tip is for debugging you can run python in the interpreter mode:
```
python3 -i plot_hist.py <comma-sep list of variables> <1/0>
```
which will give you access to the script variables in the interpreter.

A second script can be used to perform a very simple analysis:
```
python3 simple_stat_analysis.py
```
In this script we assume a target integrated luminosity of 300fb^-1. The steps are as follows:

* Perform a selection to discriminate ttH from background and other H production modes. Right now we perform simple cuts on the the number of jets, and the (second-)max b-tagging scores. This could be replaced by a cut on a more powerful ML classifier that uses a larger set of input features.
* Categorise events. For now we define two categories: zero lepton (0) and  at-least one lepton (1), which have different signal purities.
* Fit the diphoton mass distribution (around the signal peak) in each of the categories using a binned likelihood fit. Extract 2NLL as a function of the ttH signal strength to calculate the constraints as the 68% confidence interval where 2NLL cross the y=1 line. 

The output of this script are diphoton mass distributions in each analysis category, and the 2NLL curve. 

### Tasks
* Feature exploration: understand which features in the dataframes can be used to separate ttH from background and the other H production modes.
* Apply simple selection cuts and see if you can improve the ttH signal strength sensitivity.
* Use Paul's XGBoost classifier to separate ttH from background. Optimise a cut on this classifier to maximise the sensitivity to the ttH signal strength (minimize the 68% confidence interval).
* Swap to using data sidebands for the background estimate in the signal region.
* Replace the ttH signal strength with a SMEFT parametrisation and extract one-at-a-time constraints on the SMEFT Wilson coefficients.
* Categorise events according to the STXS stage 1.2 framework (in bins of the Higgs boson transverse momentum) and see how this affects the SMEFT constraints.
* Use [EFT2Obs](https://github.com/ajgilbert/EFT2Obs) tool to derive SMEFT parametrisation for a difference choice of categorisation e.g. m(ttH) instead of pT(H).
* Propagate through statistical analysis to compare SMEFT constraints for different categorisation choices.
* Add EFT effects to the current set of samples using nanoAOD reweighting tools.
* Train Machine Learning algorithm the categorise events according to SMEFT effects and maximise our sensitivity to Wilson coefficients.
