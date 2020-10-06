
import matplotlib.pyplot as plt
import numpy as np

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')
import csv
import urllib.request
from collections import namedtuple
IrisRecord = namedtuple('IrisRecord', ['sepal_length','sepal_width','petal_length','petal_width','species'])
with open('iris_data.csv','r') as file:
    read=csv.reader(file)
    data = []
    for row in read:
        try:
            data.append(IrisRecord(sepal_length = float(row[0]),sepal_width=float(row[1]),petal_length= float(row[2]),petal_width= float(row[3]), species= row[4]))
        except IndexError:
            continue

from itertools import cycle
from collections import defaultdict


def scatterplot_matrix(m, target=True):
    """Takes a MxN matrix and draws a scatterplot matrix

    This function assumes that each row in the matrix is
    organized as features first, followed by the target
    variable unless the target parameter is set to False.
    In that case, each row is considered to contain only
    features of the data.

    Keyword arguments:
    target -- if True, the last column in m is the target variable
    """
    try:
        # If each record is a namedtuple, get the list of fields;
        # we'll use those for the x- and y-axis labels of the
        # scatterplot matrix. If target is True, don't get the
        # last field name.
        features = m[0]._fields[:-1] if target else m[0]._fields
    except:
        features = range(len(m[0]) - 1) if target else range(len(m[0]))

    # If the matrix contains a target variables, create a list of classes
    if target:
        classes = list(set(r[-1] for r in m))

    # Create a color map of species names to colors
    color_cycler = cycle(plt.rcParams['axes.prop_cycle'])
    cmap = defaultdict(lambda: next(color_cycler)['color'])

    # Set the size of the plot
    fig = plt.figure(figsize=(12, 12))

    # Loop through every feature and plot it against every feature
    for i, feature_row in enumerate(features):
        for j, feature_col in enumerate(features):
            # Create a new subplot
            plt.subplot(len(features), len(features), i * len(features) + j + 1)

            # Plot the scatter plot for the current pair of features
            if i != j:
                x = [r[j] for r in m]
                y = [r[i] for r in m]
                if target:
                    c = [cmap[r[-1]] for r in m]
                else:
                    c = 'b'
                plt.scatter(x, y, edgecolors='w', c=c, linewidths=0.5)

            # Plot the histogram on the diagonal
            if target and i == j:
                df = [[r[i] for r in m if r[-1] == cls] for cls in classes]
                colors = [cmap[cls] for cls in classes]
                plt.hist(df, color=colors, histtype='barstacked')
            elif i == j:
                plt.hist([r[i] for r in m], color='b', histtype='barstacked')

            # Turn off the x-axis labels for everything but the last row
            if i < len(features) - 1:
                plt.tick_params(labelbottom='off')
            else:
                plt.xlabel(feature_col)

            # Turn off the y-axis labels for everything but the last column
            if j > 0:
                plt.tick_params(labelleft='off')
            else:
                plt.ylabel(feature_row)

            # Turn off all tick marks and make the label size
            # a bit smaller than the default
            plt.tick_params(top='off', bottom='off', left='off', right='off', labelsize=8)

scatterplot_matrix(data)
scatterplot_matrix([r[:-1] for r in data], target=False)