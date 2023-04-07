# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 19:51:50 2019

@author: weld
"""

# Third-party libraries:
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
# My libraries:


# Basic config:
plt.rcParams.update({'font.size': 14, 'figure.figsize': (6, 4.5)})


def plot_gaussians(mu_array=[], sigma_array=[], prob_thold=0, test_points=[], n=1000):
    '''
    This definition plots the sunspot data.
    '''

    # Auxiliar variables:
    labels = []
    # Generating the curves data:
    y_data = []
    # Definning the x bins:
    xmin = min(mu_array) - 3*max(sigma_array)
    xmax = max(mu_array) + 3*max(sigma_array)
    ymax = 0
    bins = [xmin]
    while bins[-1] < xmax:
        bins.append(bins[-1] + 0.001)
    bins = np.array(bins)

    plt.figure("IMG_001_probability_score_multifeatures")
    i = 1
    for mu, sigma in zip(mu_array, sigma_array):
        labels.append("$p_{" + str(i) + "}(x)$")
        y_data = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2))
        ymax = max([ymax, y_data.max()])
        plt.plot(bins, y_data, linewidth=2, zorder=5)
        thold_adjust = get_thold_adjust(mu, sigma, prob_thold)
        print('p(mu)' + str(i) + ":", round(y_data.max(), 4))
        print('Density threshold line ' + str(i) + ":", round(thold_adjust, 4))
#        labels.append("Rejection line " + str(i))
#        plt.plot([xmin, xmax], [thold_adjust, thold_adjust], linestyle='--', linewidth=2, zorder=8)
        i += 1
    
    for point in test_points:
        x = []
        y = []
        prob_score = []
        for mu, sigma in zip(mu_array, sigma_array):
            p_val = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp( - (point - mu)**2 / (2 * sigma**2))
            mu_val = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp( - (mu - mu)**2 / (2 * sigma**2))
            x.append(point)
            y.append(round(p_val, 4))
            prob_score.append(round(p_val/mu_val, 4))
        print('x:', point, 'p_vals:', y,'prob_score:', prob_score)
        plt.scatter(x, y, marker='s', color='k', s=25, zorder=10)
        line_plot_x = [xmin] + x + [point]
        y.sort(reverse=True)
        line_plot_y = [max(y)] + y + [0]
        plt.plot(line_plot_x, line_plot_y, linestyle=':', linewidth=1, color='k',zorder=8)

    plt.xlim([xmin, xmax])
    plt.ylim([0, 1.05*ymax])
    plt.xlabel("x")
    plt.ylabel("p(x)")
#    plt.grid(True)
    plt.legend(labels=labels, loc='best', prop={'size': 14})
    plt.tight_layout()
    plt.show()


def get_thold_adjust(mu, sigma, thold):
    """
    """

    x = mu
    max_density = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (mu - mu)**2 / (2 * sigma**2))
    prob_density = max_density
    while prob_density/max_density > thold:
        x += 0.001
        prob_density = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2))

    return prob_density


if __name__ == "__main__":
    '''
    TODO: implement this docstring.
    '''

    # Defining the values of mu and sigma for each curve:
    mu_array = [0, 10]
    sigma_array = [4, 8]
    prob_thold = 0.3
    test_points = [-20, -10, 0, 10, 20, 30]
    plot_gaussians(mu_array, sigma_array, prob_thold, test_points)
