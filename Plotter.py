import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mplhep as hep

plt.style.use(hep.style.CMS)

# Define a class for plotting particle physics-related data
class Plotter:
    def __init__(self, style='whitegrid', font_scale=1.5):
        """Initialize the plotter with a default seaborn style."""
        self.colors = {"blue":"#0200FB",
                       "red":"#FF0F17",
                       "green":"#58D354",
                       "black":"#000000",
                       "purple":"mediumorchid",
                       "orange":"darkorange"}
        # Set the global DPI for all plots
        plt.rcParams['figure.dpi'] = 300
        #sns.set_theme(style=style, font_scale=font_scale)

    def histogram(self, data, bins=30, weights = None,title='', xlabel='', ylabel='', legend_label='', color='blue', alpha=0.7, density=False, axes = None):
        """Create a histogram with custom labels and formatting."""
        if not axes:
            fig, axes = plt.subplots(figsize=(10, 7))
        #plt.figure(figsize=(8, 6))
        
        # Create the histogram
        #sns.histplot(data, bins=bins, kde=False, color=color, alpha=alpha, stat='density' if density else 'count', label=legend_label, element="step")
        #hep.histplot(data, bins=bins,weights=weights if weights else np.ones(len(data)), label=legend_label, color=color, ax=axes,alpha=alpha, density=density)
        sns.histplot(x=data,
                         bins=bins,
                         weights=list(weights) if weights else np.ones(len(data)),
                         kde=False,
                         color=color,
                         alpha=alpha,
                         stat='density' if density else 'count', label=legend_label, 
                         ax=axes, 
                         element="step",
                         fill=False)
        # Customize the plot
        axes.set_title(title, fontsize=18)
        axes.set_xlabel(xlabel, fontsize=20)
        axes.set_ylabel(ylabel, fontsize=20)
        if legend_label:
            axes.legend(fontsize=20)
        hep.cms.label("", com="13.6", lumi=300, lumi_format="{0:.2f}", ax=axes)
        # Improve layout
        plt.tight_layout()
        
        #plt.show()

    def line_plot(self, x, y, title='', xlabel='', ylabel='', legend_label='', color='red', linewidth=2, linestyle='-', axes = None):

        if not axes:
            fig, axes = plt.subplots(figsize=(8, 6))
        
        # Create the line plot
        sns.lineplot(x=x, y=y, linewidth=linewidth, linestyle=linestyle, label=legend_label, ax=axes, color=color)

        # Customize the plot
        axes.set_title(title, fontsize=18)
        axes.set_xlabel(xlabel, fontsize=16)
        axes.set_ylabel(ylabel, fontsize=16)
        if legend_label:
            axes.legend(fontsize=14)

        # Improve layout
        # plt.tight_layout()
        
        # plt.show()

    def overlay_histograms(self, datasets, bins=30, weights=None,title='', xlabel='', ylabel='', labels=None, colors=None, alpha=0.6, density=False, axes = None, type="bars", fill=False):
        """Create overlaid histograms for multiple datasets."""
        # plt.figure(figsize=(8, 6))
        default_figsize = (10, 7)
        if not axes:
            fig, axes = plt.subplots(figsize=default_figsize)
            current_figsize = fig.get_size_inches()
        else:
            current_figsize = axes.figure.get_size_inches()

        # Calculate scaling factor
        scaling_factor = (current_figsize[0] / default_figsize[0] + current_figsize[1] / default_figsize[1]) / 2

        # Create overlaid histograms
        for i, data in enumerate(datasets):
            label = labels[i] if labels else None
            color = self.colors[colors[i]] if colors else None
            sns.histplot(x=data,
                         bins=bins,
                         weights=list(weights[i]) if weights else np.ones(len(data)),
                         kde=False,
                         color=color,
                         alpha=alpha,
                         stat='density' if density else 'count', label=label, 
                         ax=axes, 
                         element=type,
                         fill=fill,
                         linewidth=3)
        # Customize the plot
        #axes.set_title(title, fontsize=18)
        axes.set_xlabel(xlabel, fontsize=16*scaling_factor)
        axes.set_ylabel(ylabel, fontsize=16*scaling_factor)
        if labels:
            axes.legend(fontsize=12*scaling_factor, loc="best")
        hep.cms.label(title, com="13.6", lumi=300, lumi_format="{0:.2f}", ax=axes, fontsize=10*scaling_factor)
        # Improve layout
        plt.tight_layout()

        #plt.show()

    def scatter_plot(self, x, y, title='', xlabel='', ylabel='', legend_label='', color='green', marker='o', alpha=0.8):
        """Create a scatter plot with custom labels and formatting."""
        plt.figure(figsize=(8, 6))
        
        # Create the scatter plot
        plt.scatter(x, y, color=color, marker=marker, alpha=alpha, label=legend_label)

        # Customize the plot
        plt.title(title, fontsize=18)
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        if legend_label:
            plt.legend(fontsize=14)

        # Improve layout
        plt.tight_layout()

        plt.show()
    def overlay_line_plots(self, x, y_datasets, title='', xlabel='', ylabel='', labels=None, colors=None, linewidth=2, linestyles=None, axes=None, xlim = None, ylim = None, base_fontsize=14):
        """Create overlaid line plots for multiple datasets."""
        #axes.figure(figsize=(8, 6))
        if axes is None:
            fig, axes = plt.subplots(figsize=(8, 6))
        # Get figure size and calculate scale factor
        fig_width = axes.figure.get_size_inches()[0]
        scale_factor = fig_width / 8.0  # Scale relative to default width of 8
        
        # Calculate font sizes
        title_size = base_fontsize * scale_factor  # Title slightly larger
        label_size = base_fontsize * scale_factor
        legend_size = base_fontsize * scale_factor * 0.7  # Legend slightly smaller
        
        # Plot each dataset
        for i, y in enumerate(y_datasets):
            label = labels[i] if labels else None
            color = colors[i] if colors else None
            linestyle = linestyles[i] if linestyles else '-'
            axes.plot(x, y, color=color, linewidth=linewidth, linestyle=linestyle, label=label)

        # Customize the plot
        axes.set_title(title, fontsize=title_size)
        axes.set_xlabel(xlabel, fontsize=label_size)
        axes.set_ylabel(ylabel, fontsize=label_size)
        if xlim:
            axes.set_xlim(xlim)
        if ylim:
            axes.set_ylim(ylim)
        if labels:
            axes.legend(fontsize=legend_size, loc="best",frameon=True,  # Enable the frame
                   edgecolor='black',  # Set edge color
                   fancybox=True,  # Rounded corners
                   framealpha=1)  # Solid background)
        hep.cms.label("", com="13.6", lumi=300, lumi_format="{0:.2f}", ax=axes, fontsize=title_size)
        plt.tight_layout()

        #plt.show()
    def set_global_style(self, style='whitegrid', font_scale=1.5):
        """Set a global style for all plots."""
        sns.set_theme(style=style, font_scale=font_scale)
