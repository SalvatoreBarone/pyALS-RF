"""
Copyright 2021-2023 Salvatore Barone <salvatore.barone@unina.it>

This is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation; either version 3 of the License, or any later version.

This is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License along with
RMEncoder; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA.
"""
import matplotlib.pyplot as plt, numpy as np
from matplotlib.ticker import MaxNLocator

def scatterplot(paretos, legend_markers, xlabel, ylabel, outfile, figsize = (4,4)):
    plt.figure(figsize = figsize)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for pareto, l in zip(paretos, legend_markers):
        plt.scatter(pareto[:,0], pareto[:,1], c = l.get_color(), marker = l.get_marker())
    plt.legend(handles=legend_markers, frameon=False, loc='upper right', ncol=1)
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight", pad_inches=0)
    
def boxplot(data, xlabel, ylabel, outfile, figsize = (4,4), annotate = True, float_format = "%.2f", fontsize = 14, integer_only = False):
    plt.figure(figsize=figsize)
    bp_dict = plt.boxplot(data, showmeans=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax = plt.gca()
    ax.spines[['right', 'top', 'bottom']].set_visible(False)
    if annotate:
        for median, box, mean in zip(bp_dict['medians'], bp_dict['boxes'], bp_dict['means']):
            median_x, median_y = median.get_xydata()[1] # top of median line
            plt.text(1.03 * median_x, median_y, float_format % median_y, horizontalalignment='left', verticalalignment='center', fontsize = fontsize)
            
            box_left, q1_y = box.get_xydata()[0]
            plt.text(0.97 * box_left, q1_y, float_format % q1_y, horizontalalignment='right', verticalalignment='center', fontsize = fontsize)
            _, q3_y = box.get_xydata()[3]
            plt.text(0.97 * box_left, q3_y, float_format % q3_y, horizontalalignment='right', verticalalignment='center', fontsize = fontsize)
            
            _, mean_y = mean.get_xydata()[0]
            plt.text(0.97 * box_left, mean_y, float_format % mean_y, horizontalalignment='right', verticalalignment='center', fontsize = fontsize)
    #test for nested list. If data is not a list of list, disable the ticks            
    if not any(isinstance(sub, list) for sub in data):
        plt.tick_params(bottom = False)
        plt.xticks([1], [""])
    if integer_only:
        y = np.array(data)
        yint = range(int(np.min(y)), int(np.ceil(np.max(y))+1))
        plt.yticks(yint)
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches='tight', pad_inches=0)
    