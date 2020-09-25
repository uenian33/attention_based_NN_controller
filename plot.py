import os
import numpy as np
import random

from glob import glob

import math


import matplotlib.pyplot as plt
from sklearn import preprocessing
import pickle

from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn.preprocessing import Normalizer
import pandas as pd

from bitstring import BitArray

from utils import *


def plot_states_trajs(obs_mode='autumn', pressure_mode='telescope', len=1000):
    import matplotlib.pylab as plt
    from mpl_toolkits.mplot3d.axes3d import Axes3D

    fig = plt.figure()
    # ax = Axes3D(fig)

    if obs_mode == "autumn":
        traj_files = glob('data/autumn_data_trajs/*')
    elif obs_mode == "summer":
        traj_files = glob('data/summer_data_trajs/*')

    for idx, sensor_f in enumerate(traj_files[0:len]):
        dataset = load_exps(exp_f=sensor_f, rough=True)
        if obs_mode == "autumn":
            if pressure_mode == 'wheel':
                dx = dataset[:, [19, 20, 35, 36]]
                # dx = preprocessing.scale(dx, axis=1)
                dy = dataset[:, [52, 53, 41]]
            elif pressure_mode == 'telescope':
                dx = dataset[:, [19, 20, 27, 28]]
                # dx = preprocessing.scale(dx, axis=1)
                dy = dataset[:, [52, 53, 41]]
            elif pressure_mode == 'wheel_telescope':
                dx = dataset[:, [1, 2, 27, 28, 35, 36]]  # boom, bucket, telescope7,telescope8, wheel, wheel
                # dx = preprocessing.scale(dx, axis=1)
                dy = dataset[:, [52, 53, 41]]

        elif obs_mode == "summer":
            if pressure_mode == 'wheel':
                dx = dataset[:, [1, 2, 16, 17]]
                # dx = preprocessing.scale(dx, axis=1)
                dy = dataset[:, [34, 35, 22]]
            elif pressure_mode == 'telescope':
                dx = dataset[:, [1, 2, 8, 9]]
                # dx = preprocessing.scale(dx, axis=1)
                dy = dataset[:, [34, 35, 22]]
            elif pressure_mode == 'wheel_telescope':
                dx = dataset[:, [1, 2, 8, 9, 16, 17]]  # boom, bucket, telescope7,telescope8, wheel, wheel
                # dx = preprocessing.scale(dx, axis=1)
                dy = dataset[:, [34, 35, 22]]

        inde = np.linspace(0, 1, dx.shape[0])
        plt.plot(inde, dx[:, 3], alpha=0.8)
    plt.show()


def plot_validations():
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    fig, ax = plt.subplots()
    clrs = sns.color_palette("husl", 5)
    with sns.axes_style("darkgrid"):
        epochs = list(range(101))
        for i in range(5):
            meanst = np.array(means.ix[i].values[3:-1], dtype=np.float64)
            sdt = np.array(stds.ix[i].values[3:-1], dtype=np.float64)
            ax.plot(epochs, meanst, label=means.ix[i]["label"], c=clrs[i])
            ax.fill_between(epochs, meanst - sdt, meanst + sdt, alpha=0.3, facecolor=clrs[i])
        ax.legend()
        ax.set_yscale('log')

plot_states_trajs()
# plot_validations()
