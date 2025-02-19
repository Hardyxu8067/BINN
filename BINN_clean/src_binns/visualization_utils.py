import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import torch
from matplotlib.colors import Normalize 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, euclidean_distances
from scipy.interpolate import interpn
import subprocess


# Get the hash of the latest Git commit.
# TODO - this is not a visualization method, but temporarily putting it here for convenience
def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def plot_losses(filename, losses, labels, min_val=None, max_val=None):
    """
    Plots all loss curves to the given filename.
    'losses' should be a list of lists, where each inner list represents
    a particular loss (at each epoch).
    'labels' should contain a string for each loss type.
    """
    for loss_idx, loss_curve in enumerate(losses):
        plt.plot(np.arange(len(loss_curve)), loss_curve, label=labels[loss_idx])
    plt.xlabel('Epoch #')
    plt.ylabel('Value')
    if min_val is not None and max_val is not None:
        plt.ylim([min_val, max_val])
    plt.legend()
    plt.savefig(filename)
    plt.close()


def density_scatter(x, y, ax=None, sort=True, bins=20, **kwargs):
    """Plot a scatter between x/y with density coloring (with 2d histogram).
    
    Code from https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
    """

    if ax is None :
        fig, ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )
    norm = Normalize(vmin = np.min(z), vmax = np.max(z))

    return ax


def plot_single_scatter(ax, x, y, x_label, y_label, title, should_align=True):
    # Convert PyTorch Tensor or list into Numpy array
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    elif type(x) == list:
        x = np.array(x)
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()
    elif type(y) == list:
        y = np.array(y)
    not_nan = ~np.isnan(x) & ~np.isnan(y)
    y = y[not_nan]
    x = x[not_nan]

    # If there are at least 2 datapoints, fit linear regression,
    # plot trendline, compute stats
    if x.size >= 2:
        x = x.reshape(-1, 1)
        regression = LinearRegression(fit_intercept=True).fit(x, y)
        slope = regression.coef_[0]
        intercept = regression.intercept_
        regression_line = slope * x + intercept
        regression_equation = 'y={:.2f}x+{:.2f}'.format(slope, intercept)
        identity_line = x
        y_pred = regression.predict(x)

        # Compute statistics
        y = y.ravel()
        x = x.ravel()
        if should_align:
            r2 = r2_score(y, x)
        else:
            r2 = r2_score(y, y_pred)
        corr = np.corrcoef(x, y)[0, 1]
        mae = mean_absolute_error(x, y)
        rmse =  math.sqrt(mean_squared_error(x, y))

        # Plot stats, regression line, labels
        if should_align:
            stats_string = '(R^2={:.3f}, RMSE={:.3f}, Corr={:.3f})'.format(r2, rmse, corr)
        else:
            stats_string = '(R^2={:.3f}, Corr={:.3f})'.format(r2, corr)
        ax.plot(x, regression_line, 'r', label=regression_equation + ' ' + stats_string) # ' (R^2={:.2f}, Corr={:.2f}, MAPE={:.2f})'.format(r2, corr, mape))
        if should_align:
            ax.plot(x, identity_line, 'g', label='Identity function')
        ax.legend(fontsize=10)

    # Plot scatterplot for this crop type
    if x.size > 500:
        density_scatter(x, y, ax=ax, s=8) #, s=2, color='black')
    elif x.size > 100:
        ax.scatter(x, y, color="k", s=30)
    else:
        ax.scatter(x, y, color="k", s=50)

    ax.tick_params(labelsize=12)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    if should_align and x.size >= 1:
        min_value = min(np.min(x), np.min(y))
        max_value = max(np.max(x), np.max(y))
        margin = (max_value - min_value) * 0.02
        ax.set_xlim(min_value - margin, max_value + margin)
        ax.set_ylim(min_value - margin, max_value + margin)

    ax.set_title(title + " (num datapoints: " + str(len(x)) + ")", fontsize=12)


def plot_true_vs_predicted(filename, y_hat, y):
    """
    Plot a scatter of y vs. y_hat to filename.
    Assumes y and y_hat are single-dimensional and have no nans"""
    assert y.shape == y_hat.shape
    plot_single_scatter(plt.gca(), y_hat, y, "Predicted", "True", "True vs predicted SOC")
    plt.savefig(filename)
    plt.close()


def plot_true_vs_predicted_multiple(filename, y_hats, ys, titles, cols=None):
    """
    Plots multiple scatters to a single filename.
    Scatter i plots y_hats[i] as prediction and ys[i] as true values,
    and will have title titles[i].

    y_hats is a list of 1D numpy arrays or Tensors
    ys is a list of 1D numpy arrays or Tensors
    titles is a list of strings
    All lists must have same length
    """
    n_plots = len(y_hats)
    assert len(y_hats) == len(ys)
    assert len(y_hats) == len(titles)

    if cols is None:
        cols = min(4, n_plots)
    rows = math.ceil(n_plots / cols)
    fig, axeslist = plt.subplots(rows, cols, figsize=(7*cols, 7*rows), squeeze=False)
    fig.suptitle('True vs predicted', fontsize=13)
    for i in range(n_plots):
        ax = axeslist.ravel()[i]
        plot_single_scatter(ax, y_hats[i], ys[i], "Predicted", "True", titles[i])
    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    plt.savefig(filename)
    plt.close()


def plot_observations_world_map(lons, lats, values, plot_dir, var_name, title=None, us_only=False, ax=None):
    if title is None:
        title = var_name
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    # If data were tensors, first convert to numpy
    if torch.is_tensor(lons):
        lons = lons.detach().cpu().numpy()
    if torch.is_tensor(lats):
        lats = lats.detach().cpu().numpy()
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()

    # Exclude NaNs and Infs
    df = pd.DataFrame({"lon": lons,
                       "lat": lats,
                        var_name: values})
    df = df[~np.isnan(df[var_name])]
    df = df[~np.isinf(df[var_name])]

    # Plot world map
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world.boundary.plot(ax=ax, color='gray')
    if us_only:
        ax.set_xlim(-124.8, -66.9)
        ax.set_ylim(24.5, 49.4)
    gdf.plot(column=var_name, ax=ax, marker='o', markersize=8, legend=True, zorder=10)  # legend_kwds={'shrink': 0.7},
    ax.set_title(title)

    if plot_dir is not None:
        plt.savefig(os.path.join(plot_dir, "map_{}.png".format(var_name)), bbox_inches='tight')
        plt.close()

    # # Plot histograms of the raw values
    # # filtered_values = values[~np.isnan(values)]
    # # filtered_values = filtered_values[~np.isinf(filtered_values)]
    # values = values[~np.isnan(values)]
    # values = values[~np.isinf(values)]
    # plt.hist(values, bins=30)
    # plt.title(title)
    # plt.savefig(os.path.join(plot_dir, "histogram_{}.png".format(var_name)))
    # plt.close()

def plot_map_grid(filename, lons_list, lats_list, values_list, vars_list, us_only=False, cols=None):
    n_plots = len(lons_list)
    assert len(lons_list) == len(lats_list)
    assert len(lons_list) == len(values_list)
    assert len(lons_list) == len(vars_list)

    if cols is None:
        cols = min(4, n_plots)
    rows = math.ceil(n_plots / cols)
    fig, axeslist = plt.subplots(rows, cols, figsize=(12*cols, 6*rows), squeeze=False)
    for i in range(n_plots):
        ax = axeslist.ravel()[i]
        plot_observations_world_map(lons_list[i], lats_list[i], values_list[i], plot_dir=None, 
                                    var_name=vars_list[i], title=None, us_only=us_only, ax=ax)
    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    plt.savefig(filename)
    plt.close()

