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


action_scale = [128., 128., 50.]


def show_figures(y_rf, dy, mode='compare'):
    from scipy.signal import savgol_filter
    print(y_rf.shape)

    if mode == 'single':
        a = y_rf[:, 0]
        b = dy[:, 0]
        # yhat = savgol_filter(b, 51, 3)  # window size 51, polynomial order 3

        inde = np.linspace(0, 1, len(a))
        plt.subplot(321)
        plt.plot(inde, a, label='predict')
        plt.subplot(322)
        plt.plot(inde, b, label='gt')

        a = y_rf[:, 1]
        b = dy[:, 1]
        # yhat = savgol_filter(b, 51, 3)  # window size 51, polynomial order 3

        inde = np.linspace(0, 1, len(a))
        plt.subplot(323)
        plt.plot(inde, a, label='predict')
        plt.subplot(324)
        plt.plot(inde, b, label='gt')

        a = y_rf[:, 2]
        b = dy[:, 2]
        # yhat = savgol_filter(b, 51, 3)  # window size 51, polynomial order 3

        inde = np.linspace(0, 1, len(a))
        plt.subplot(325)
        plt.plot(inde, a, label='predict')
        plt.subplot(326)
        plt.plot(inde, b, label='gt')

        plt.legend()
        plt.show()

    elif mode == 'compare':
        if len(y_rf.shape) > 1:

            a = y_rf[:, 0]
            b = dy[:, 0]
            # yhat = savgol_filter(b, 51, 3)  # window size 51, polynomial order 3

            inde = np.linspace(0, 1, len(a))
            plt.subplot(311)
            plt.plot(inde, b, label='gt')
            plt.plot(inde, a, label='predict')

            a = y_rf[:, 1]
            b = dy[:, 1]
            # yhat = savgol_filter(b, 51, 3)  # window size 51, polynomial order 3
            inde = np.linspace(0, 1, len(a))
            plt.subplot(312)
            plt.plot(inde, b, label='gt')
            plt.plot(inde, a, label='predict')

            a = y_rf[:, 2]
            b = dy[:, 2]
            # yhat = savgol_filter(b, 51, 3)  # window size 51, polynomial order 3
            inde = np.linspace(0, 1, len(a))
            plt.subplot(313)
            plt.plot(inde, b, label='gt')
            plt.plot(inde, a, label='predict')
            #plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

            plt.legend()
            plt.show()
        else:
            a = y_rf
            b = dy
            # yhat = savgol_filter(b, 51, 3)  # window size 51, polynomial order 3

            inde = np.linspace(0, 1, len(a))
            plt.legend()
            plt.show()


def convert_csv_2_array_summer_files(output_file='data/summer_data_trajs/'):
    # act boom: sensor_items[34],
    # act bucket: sensor_items[35]
    # act gas: sensor_items[22]

    # state boom: [1]
    # state bucket: [2]
    # state pressure: [16]
    # state pressure: [17]
    task = 'scoop'  #
    summer_path = '/media/wenyan/data_nataliya/00-original/trajectories/sensors_to_process/'
    summer_sensor_files = sorted(glob('/media/wenyan/data_nataliya/00-original/trajectories/sensors_to_process/*scoop*'))
    sumer_video_folders = sorted(glob('/media/wenyan/data_nataliya/00-original/zed_output/*'))
    traj_idx = 0
    for idx, sensor_f in enumerate(summer_sensor_files):
        regression_data = []
        basename = os.path.splitext(os.path.basename(sensor_f))[0]
        traj_file_name = output_file + basename
        with open(sensor_f) as sf:
            ses = sf.readlines()

            for line in ses:
                data = line.replace('\n', '').split(',')
                data = np.array(data)
                data = data.astype(np.float)
                regression_data.append(data)
                with open(traj_file_name, 'a') as outf:
                    out_line = ''
                    for t in data:
                        out_line += str(t) + " "
                    out_line += '\n'
                    outf.write(out_line)

                # print(data)
        # print(ses)
        # if len(ses) > 0:
    return  # regression_data


def convert_csv_2_array_autumn_files(output_file='data/autumn_data_trajs/'):
    # act boom: sensor_items[52],
    # act bucket: sensor_items[53]
    # act gas: sensor_items[41]

    # state boom: [19]
    # state bucket: [20]
    # state pressure: [35]
    # state pressure: [36]
    autumn_folders = sorted(glob('/media/wenyan/data_nataliya/avant/demos_autumn_winter/2019-12-03-demo/*.csv'))
    traj_idx = 0
    for idx, demo in enumerate(autumn_folders):
        regression_data = []
        basename = os.path.splitext(os.path.basename(demo))[0]
        traj_file_name = output_file + basename
        print(demo)
        with open(demo) as sf:
            ses = sf.readlines()
            print(demo, len(ses))

            for line in ses:
                data = line.replace('\n', '').split(',')
                data = np.array(data)
                data = data.astype(np.float)
                regression_data.append(data)
                with open(traj_file_name, 'a') as outf:
                    out_line = ''
                    for t in data:
                        out_line += str(t) + " "
                    out_line += '\n'
                    outf.write(out_line)

                # print(len(data))
                # print(ses)
                # if len(ses) > 0:
    return  # regression_data


def convert_csv_2_array_summer(output_file='data/summer_data'):
    # act boom: sensor_items[34],
    # act bucket: sensor_items[35]
    # act gas: sensor_items[22]

    # state boom: [1]
    # state bucket: [2]
    # state pressure: [16]
    # state pressure: [17]
    task = 'scoop'  #
    summer_path = '/media/wenyan/data_nataliya/00-original/trajectories/sensors_to_process/'
    summer_sensor_files = sorted(glob('/media/wenyan/data_nataliya/00-original/trajectories/sensors_to_process/*scoop*'))
    sumer_video_folders = sorted(glob('/media/wenyan/data_nataliya/00-original/zed_output/*'))
    traj_idx = 0
    regression_data = []
    for idx, sensor_f in enumerate(summer_sensor_files):
        with open(sensor_f) as sf:
            ses = sf.readlines()

            for line in ses:
                data = line.replace('\n', '').split(',')
                data = np.array(data)
                data = data.astype(np.float)
                regression_data.append(data)
                with open(output_file, 'a') as outf:
                    out_line = ''
                    for t in data:
                        out_line += str(t) + " "
                    out_line += '\n'
                    outf.write(out_line)

                # print(data)
        # print(ses)
        # if len(ses) > 0:
    return regression_data


def convert_csv_2_array_autumn(output_file='data/autumn_data'):
    # act boom: sensor_items[52],
    # act bucket: sensor_items[53]
    # act gas: sensor_items[41]

    # state boom: [19]
    # state bucket: [20]
    # state pressure: [35]
    # state pressure: [36]
    autumn_folders = sorted(glob('/media/wenyan/data_nataliya/avant/demos_autumn_winter/2019-12-03-demo/*.csv'))
    traj_idx = 0
    regression_data = []
    for idx, demo in enumerate(autumn_folders):
        print(demo)
        with open(demo) as sf:
            ses = sf.readlines()
            print(demo, len(ses))

            for line in ses:
                data = line.replace('\n', '').split(',')
                data = np.array(data)
                data = data.astype(np.float)
                regression_data.append(data)
                with open(output_file, 'a') as outf:
                    out_line = ''
                    for t in data:
                        out_line += str(t) + " "
                    out_line += '\n'
                    outf.write(out_line)

                # print(len(data))
                # print(ses)
                # if len(ses) > 0:
    return regression_data


def load_exps(exp_f, rough=True):
        # prepare dataset
    print("loading dataset...")
    dataset = []
    dataset_rough = []
    #""
    with open(exp_f) as fp:
        exp_lines = fp.readlines()
        print(len(exp_lines))
        exp_lines = exp_lines[:75000]

    for i in exp_lines:
        exp = i.split(' ')
        exp_single = np.array(exp[:len(exp) - 1])
        exp_single_load = np.hstack(exp_single.astype(np.float32))
        # print(exp_single_load)

        dataset_rough.append(exp_single_load)
        if rough:
            dataset.append(exp_single_load)
        else:
            if abs(exp_single_load[11] > 0.1) or abs(exp_single_load[12] > 0.1) or abs(exp_single_load[13] > 0.1):
                # print(exp_single)
                dataset.append(exp_single_load)

    return np.array(dataset)


def generate_dataset(obs_mode='summer', pressure_abs=True, rough=True):

    # plot_v_field(dataset_rough)
    # print(exp_single_load)
    if obs_mode == "summer":
        dataset = load_exps(exp_f='data/summer_data', rough=rough)
        # dataset_rough = load_exps(exp_f='/home/wenyan/Desktop/exps/ddpg_v0/ddpg_exps', rough=True)  # load_exps(rough=True)
        print(len(dataset[0]))
        print(dataset.shape)
        dx = dataset[:, [1, 2, 16, 17]]
        #dx = preprocessing.scale(dx, axis=1)
        dy = dataset[:, [34, 35, 22]]

    elif obs_mode == "autumn":
        dataset = load_exps(exp_f='data/autumn_data', rough=rough)
        # dataset_rough = load_exps(exp_f='/home/wenyan/Desktop/exps/all_obs/vis+sens', rough=True)  # load_exps(rough=True)

        dx = dataset[:, [19, 20, 35, 36]]
        #dx = preprocessing.scale(dx, axis=1)
        dy = dataset[:, [52, 53, 41]]

    if pressure_abs:
        nx = np.zeros((len(dataset), 3))
        nx[:, 0] = dx[:, 0]
        nx[:, 1] = dx[:, 1]
        nx[:, 2] = abs(dx[:, 2] - dx[:, 3])
    dy = dy.astype(np.int8)
    print(dy)
    dy[:, 0] = convert_uint_sensor(dy[:, 0])
    dy[:, 1] = convert_uint_sensor(dy[:, 1])
    dy = dy / action_scale
    print(dy)
    return dx, dy


def convert_uint_sensor(sens_array):
    convereted_arr = []
    for idx, sens in enumerate(sens_array):
        if sens > 128:
            print(sens)
            s = bin(int(sens)).replace("0b", "")
            num = '0' + s[1:]
            b = -1 * BitArray(bin=num).int
        else:
            b = sens
        convereted_arr.append(b)
    return np.array(convereted_arr)


def plot_states_trajs(obs_mode='autumn', len=1):
    import matplotlib.pylab as plt
    from mpl_toolkits.mplot3d.axes3d import Axes3D

    fig = plt.figure()
    ax = Axes3D(fig)

    if obs_mode == "autumn":
        traj_files = glob('data/autumn_data_trajs/*')
    elif obs_mode == "summer":
        traj_files = glob('data/summer_data_trajs/*')

    for idx, sensor_f in enumerate(traj_files[10:len]):
        if obs_mode == "autumn":
            dataset = load_exps(exp_f=sensor_f, rough=True)
            dx = dataset[:, [19, 20, 35, 36]]
            dy = dataset[:, [52, 53, 41]]

        elif obs_mode == "summer":
            dataset = load_exps(exp_f=sensor_f, rough=True)
            dx = dataset[:, [1, 2, 16, 17]]
            dy = dataset[:, [34, 35, 22]]

        X = dx[:, 0]
        Y = dx[:, 1]
        Z = abs(dx[:, 2] - dx[:, 3])
        print(dy)
        dy[:, 0] = convert_uint_sensor(dy[:, 0])
        dy[:, 1] = convert_uint_sensor(dy[:, 1])

        nx = np.zeros((X.size, 3))

        mode = 'single'
        if mode == 'single':
            nx[:, 0] = X
            nx[:, 1] = Y
            nx[:, 2] = Z
            show_figures(dy, nx, mode=mode)

        if mode == 'compare':
            nx[:, 0] = X / (max(X) - min(X))
            nx[:, 1] = Y / (max(Y) - min(Y))
            nx[:, 2] = Z / (max(Z) - min(Z))
            print(dy.shape, nx.shape)

            dy[:, 0] = dy[:, 0] / (max(dy[:, 0]) - min(dy[:, 0]))
            dy[:, 1] = dy[:, 1] / (max(dy[:, 1]) - min(dy[:, 1]))
            dy[:, 2] = dy[:, 2] / (max(dy[:, 2]) - min(dy[:, 2]))

            #xscaler = Normalizer().fit(nx)
            #nx = xscaler.transform(nx)
            #yscaler = Normalizer().fit(dy)
            #dy = yscaler.transform(dy)
            show_figures(dy, nx, mode=mode)
        """
        N = X.size
        for i in range(N):
            if i == N - 1:
                print(i)
            ax.set_xlabel('\n' + 'boom', linespacing=4)
            ax.set_ylabel('bucket')
            ax.plot(X[:i], Y[:i], Z[:i])
            plt.pause(0.001)
            ax.cla()
        """
    # plt.show()


def annimate_states_trajs(obs_mode='autumn', len=1):
    from matplotlib import pyplot as plt
    from matplotlib import animation
    from mpl_toolkits.mplot3d import Axes3D

    #fig = plt.figure()
    #ax = Axes3D(fig)

    if obs_mode == "autumn":
        traj_files = glob('data/autumn_data_trajs/*')
    elif obs_mode == "summer":
        traj_files = glob('data/summer_data_trajs/*')

    for idx, sensor_f in enumerate(traj_files[:len]):
        if obs_mode == "autumn":
            dataset = load_exps(exp_f=sensor_f, rough=True)
            dx = dataset[:, [19, 20, 35, 36]]
        elif obs_mode == "summer":
            dataset = load_exps(exp_f=sensor_f, rough=True)
            dx = dataset[:, [1, 2, 16, 17]]

        X = dx[:, 0]
        Y = dx[:, 1]
        Z = dx[:, 2] - dx[:, 3]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        seg, = ax.plot([], [], [], lw=1)
        ax.set_xlim3d(min(X), max(X))
        ax.set_ylim3d(min(Y), max(Y))
        ax.set_zlim3d(min(Y), max(Z))

        def init():
            return seg,

        def updateFigure(t):
            p1 = X[t]
            p2 = Y[t]
            p3 = Z[t]
            seg._verts3d = (p1, p2, p3)
            print(p1, p2, p3)
            return seg,

        ani = animation.FuncAnimation(
            fig, updateFigure, init_func=init, frames=500, interval=5, blit=True)
        plt.show()
        # Save
        #anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    # plt.show()

# convert_csv_2_array_summer()
# generate_dataset(obs_mode='autumn')
# convert_csv_2_array_summer()
# convert_csv_2_array_autumn_files()
# convert_csv_2_array_summer_files()
#plot_states_trajs(obs_mode='autumn', len=100)
#annimate_states_trajs(obs_mode='autumn', len=1)
