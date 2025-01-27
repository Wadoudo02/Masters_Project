import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mplhep as hep

plt.style.use(hep.style.CMS)

# Define a class for plotting particle physics-related data
class Plotter:
    def __init__(self, style='whitegrid', font_scale=1.5):
        """Initialize the plotter with a default seaborn style."""
        sns.set_theme(style=style, font_scale=font_scale)

    def histogram(self, data, bins=30, title='', xlabel='', ylabel='', legend_label='', color='blue', alpha=0.7, density=False):
        """Create a histogram with custom labels and formatting."""
        plt.figure(figsize=(8, 6))
        
        # Create the histogram
        sns.histplot(data, bins=bins, kde=False, color=color, alpha=alpha, stat='density' if density else 'count', label=legend_label, element="step")

        # Customize the plot
        plt.title(title, fontsize=18)
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        if legend_label:
            plt.legend(fontsize=14)
        
        # Improve layout
        plt.tight_layout()
        
        plt.show()

    def line_plot(self, x, y, title='', xlabel='', ylabel='', legend_label='', color='red', linewidth=2, linestyle='-'):
        plt.figure(figsize=(8, 6))
        
        # Create the line plot
        sns.lineplot(x=x, y=y, linewidth=linewidth, linestyle=linestyle, label=legend_label)

        # Customize the plot
        plt.title(title, fontsize=18)
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        if legend_label:
            plt.legend(fontsize=14)

        # Improve layout
        plt.tight_layout()
        
        plt.show()

    def overlay_histograms(self, datasets, bins=30, title='', xlabel='', ylabel='', labels=None, colors=None, alpha=0.6, density=False):
        """Create overlaid histograms for multiple datasets."""
        plt.figure(figsize=(8, 6))

        # Create overlaid histograms
        for i, data in enumerate(datasets):
            label = labels[i] if labels else None
            color = colors[i] if colors else None
            sns.histplot(data, bins=bins, kde=False, color=color, alpha=alpha, stat='density' if density else 'count', label=label)

        # Customize the plot
        plt.title(title, fontsize=18)
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        if labels:
            plt.legend(fontsize=14)
        
        # Improve layout
        plt.tight_layout()

        plt.show()

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
    def overlay_line_plots(self, x, y_datasets, title='', xlabel='', ylabel='', labels=None, colors=None, linewidth=2, linestyles=None):
        """Create overlaid line plots for multiple datasets."""
        plt.figure(figsize=(8, 6))

        # Plot each dataset
        for i, y in enumerate(y_datasets):
            label = labels[i] if labels else None
            color = colors[i] if colors else None
            linestyle = linestyles[i] if linestyles else '-'
            plt.plot(x, y, color=color, linewidth=linewidth, linestyle=linestyle, label=label)

        # Customize the plot
        plt.title(title, fontsize=18)
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        if labels:
            plt.legend(fontsize=14)

        # Improve layout
        plt.tight_layout()

        plt.show()
    def set_global_style(self, style='whitegrid', font_scale=1.5):
        """Set a global style for all plots."""
        sns.set_theme(style=style, font_scale=font_scale)
