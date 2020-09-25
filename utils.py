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
import matplotlib as mpl

action_scale = [128., 128., 50.]


def show_figures(y_rf, dy, mode='compare', path=None):
    from scipy.signal import savgol_filter
    print(y_rf.shape)

    if mode == 'single':
        a = y_rf[:, 0]
        b = dy[:, 0]
        # yhat = savgol_filter(b, 51, 3)  # window size 51, polynomial order 3

        inde = np.linspace(0, len(a))
        plt.subplot(321)
        plt.plot(inde, a, 's', label='predict')
        plt.subplot(322)
        plt.plot(inde, b, label='gt')

        a = y_rf[:, 1]
        b = dy[:, 1]
        # yhat = savgol_filter(b, 51, 3)  # window size 51, polynomial order 3

        plt.subplot(323)
        plt.plot(inde, a, label='predict')
        plt.subplot(324)
        plt.plot(inde, b, label='gt')

        a = y_rf[:, 2]
        b = dy[:, 2]
        # yhat = savgol_filter(b, 51, 3)  # window size 51, polynomial order 3

        plt.subplot(325)
        plt.plot(inde, a, label='predict')
        plt.subplot(326)
        plt.plot(inde, b, label='gt')

        plt.legend()
        plt.show()

    elif mode == 'compare':
        mpl.style.use('seaborn')
        if len(y_rf.shape) > 1:

            a = y_rf[:, 0]
            b = dy[:, 0]
            # yhat = savgol_filter(b, 51, 3)  # window size 51, polynomial order 3

            inde = np.linspace(0, len(a), len(a))
            plt.subplot(311)
            plt.plot(inde, b, label='boom human', c='olive')
            plt.plot(inde, a, label='boom prediction', c='salmon')
            plt.scatter(50, 0, s=10, c='black')
            plt.scatter(114, 0, s=10, c='black')
            plt.scatter(170, 0, s=10, c='black')
            plt.scatter(213, 0, s=10, c='black')
            plt.legend()
            plt.xticks([])

            a = y_rf[:, 1]
            b = dy[:, 1]
            # yhat = savgol_filter(b, 51, 3)  # window size 51, polynomial order 3
            inde = np.linspace(0, len(a), len(a))
            plt.subplot(312)
            plt.plot(inde, b, label='bucket human', c='olive')
            plt.plot(inde, a, label='bucket prediction', c='salmon')
            plt.scatter(50, 0, s=10, c='black')
            plt.scatter(114, 0, s=10, c='black')
            plt.scatter(170, 0, s=10, c='black')
            plt.scatter(213, 0, s=10, c='black')
            plt.legend()
            plt.xticks([])

            a = y_rf[:, 2]
            b = dy[:, 2]
            # yhat = savgol_filter(b, 51, 3)  # window size 51, polynomial order 3
            inde = np.linspace(0, len(a), len(a))
            plt.subplot(313)
            plt.plot(inde, b, label='gas human', c='olive')
            plt.plot(inde, a, label='gas prediction', c='salmon')
            plt.scatter(50, 0, s=10, c='black')
            plt.scatter(114, 0, s=10, c='black')
            plt.scatter(170, 0, s=10, c='black')
            plt.scatter(213, 0, s=10, c='black')
            # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.legend()
            plt.xticks([])

            if path == None:
                plt.show()
            else:
                plt.savefig(path)
                plt.show(block=False)
                plt.pause(1)
                plt.close("all")
        else:
            a = y_rf
            b = dy
            # yhat = savgol_filter(b, 51, 3)  # window size 51, polynomial order 3

            inde = np.linspace(0, 1, len(a))
            plt.legend()

            if path == None:
                plt.show()
            else:
                plt.savefig(path)
                plt.show(block=False)
                plt.pause(1)
                plt.close("all")
    return plt


# narvi/vis_reward/t3d
# sensor[1:12], [13:15], [16:18], [20:22], [24:27], [29,31]


import os
import numpy as np
import random

from glob import glob

import math


def parse_sensor(sensor_data, parse_mode='simple'):
    s_data = np.array(sensor_data[:len(sensor_data) - 1]).astype(float)
    if parse_mode == "simple":
        s = np.hstack((s_data[1],  s_data[2], abs(s_data[16] - s_data[17])))
    elif parse_mode == "all":
        s = np.hstack((s_data[1:12],  s_data[13:15], s_data[16:18], s_data[20:22], s_data[24:27], s_data[29:31]))
    else:
        s = None
    return s


def generate_image_dataset(mode='vis+sens', out_folder='', np_version=False, sparse_reward=True):
    sub_tasks = ['scoop']  # sub_tasks = ['back', 'bucket', 'scoop']  #
    video_folders = sorted(glob('/media/wenyan/data_nataliya/00-original/zed_output/*'))
    # print(video_folders)

    traj_idx = 0
    for idx, test_f in enumerate(video_folders):
        # print(idx, test_f)
        for task in sub_tasks:

            sub_folders = test_f + '/' + task + '*'
            sub_folders_l = sorted(glob(sub_folders))
            for i, f in enumerate(sub_folders_l):
                ims = sorted(glob(f + '/depth*'))
                vis_f = 'trajs/vis_' + os.path.basename(test_f) + '_' + task + str(i + 1)
                tmp = os.path.basename(test_f)
                tmp_s = tmp.replace('camera', 'sensors')
                sensor_f = '../sensors_to_process/' + tmp_s + task + str(i + 1) + '.csv'

                print(os.path.isfile(vis_f))
                print(os.path.isfile(sensor_f))

                if os.path.isfile(vis_f) and os.path.isfile(sensor_f):
                    print(vis_f)
                    print(sensor_f)
                    with open(vis_f) as vf:
                        vis = vf.readlines()
                    with open(sensor_f) as sf:
                        ses = sf.readlines()

                    for idx, l in enumerate(vis):
                        if idx != len(vis) - 1 and len(ses) > 0:
                            vis_items = l.split(' ')

                            # current visual states and reward
                            if sparse_reward:
                                reward = float(vis_items[len(vis_items) - 2])
                            else:
                                reward = float(vis_items[len(vis_items) - 1])
                            visual_features = np.array(vis_items[:8])
                            visual_features = visual_features.astype(float)
                            # current sensor states and reward
                            frame_gap = int(len(ses) / len(vis))
                            sensor_idx = idx * frame_gap + random.randint(0, frame_gap - 1)
                            sensor_items = ses[sensor_idx].split(',')
                            actions_boom_bucket = np.array((sensor_items[34], sensor_items[35])).astype(np.int8)
                            actions_boom_bucket = actions_boom_bucket / 128.
                            action_gas = float(sensor_items[22]) / 50.
                            actions = np.hstack((actions_boom_bucket, action_gas))
                            print(actions)

                            # angle + pressure
                            sensor_state = parse_sensor(sensor_items, parse_mode="all")

                            # next sensor states
                            next_idx = idx + 1
                            next_vis_items = vis[next_idx].split(' ')
                            next_visual_features = np.array(next_vis_items[:8])
                            next_visual_features = next_visual_features.astype(float)
                            print(vis[next_idx].split(' '), next_visual_features)
                            # current sensor states and reward
                            frame_gap = int(len(ses) / len(vis))
                            next_sensor_idx = next_idx * frame_gap + random.randint(0, frame_gap - 1)
                            next_sensor_items = ses[next_sensor_idx].split(',')
                            next_sensor_state = parse_sensor(next_sensor_items, parse_mode="all")

                            #    print('sensor states:', sensor_state, sensor_state.size)
                            #    print('n sensor states:', next_sensor_state, next_sensor_state.size)
                            """
                            print('reward:', reward, idx)
                            print('vis:', visual_features)
                            print('actions:', actions)
                            print('states:', sensor_state, sensor_idx)
                            print('n vis:', visual_features)
                            print('n states:', next_sensor_state, next_sensor_idx)
                            """

                            import time
                            # time.sleep(1000)
                            done = False
                            if mode == 'vis+sens':
                                print(mode)
                                states = np.hstack((sensor_state, visual_features))
                                next_states = np.hstack((next_sensor_state, next_visual_features))

                                print('next states:', next_states)
                                traj = np.hstack((states, actions, [reward], next_states))
                                if np_version:
                                    outfile = out_folder + 'all_obs/np' + str(traj_idx)
                                    np.save(outfile, traj)
                                else:
                                    outfile = out_folder + 'all_obs/vis+sens'
                                    with open(outfile, 'a') as outf:
                                        out_line = ''
                                        for t in traj:
                                            out_line += str(t) + " "
                                        out_line += '\n'
                                        outf.write(out_line)

                                    subf = out_folder + 'all_obs/' + os.path.basename(test_f) + '_' + task + str(i + 1)
                                    print(subf)
                                    with open(subf, 'a') as sub_outf:
                                        sub_out_line = ''
                                        for t in traj:
                                            sub_out_line += str(t) + " "
                                        sub_out_line += '\n'
                                        sub_outf.write(sub_out_line)
                                print(traj)
                                # import time
                                # time.sleep(1000)
                            elif mode == 'vis':
                                states = visual_features
                                next_states = next_visual_features
                                traj = np.hstack((states, actions, [reward], next_states))
                                if np_version:
                                    outfile = out_folder + 'vis/_' + str(traj_idx)
                                    np.save(outfile, traj)
                                else:
                                    outfile = out_folder + 'all_obs/vis'
                                    with open(outfile, 'a') as outf:
                                        out_line = ''
                                        for t in traj:
                                            out_line += str(t) + " "
                                        out_line += '\n'
                                        outf.write(out_line)
                            elif mode == 'sens':
                                states = sensor_state
                                next_states = next_sensor_state
                                traj = np.hstack((states, actions, [reward], next_states))
                                if np_version:
                                    outfile = out_folder + 'sensor/_' + str(traj_idx)
                                    np.save(outfile, traj)
                                else:
                                    outfile = out_folder + 'all_obs/sens'
                                    with open(outfile, 'a') as outf:
                                        out_line = ''
                                        for t in traj:
                                            out_line += str(t) + " "
                                        out_line += '\n'
                                        outf.write(out_line)

                            # print('states:', states, states.size)
                            # print('n states:', next_states, next_states.size)
                            print('traj size', traj, traj.size)
                            print('-----------------------------------------------------')

                            traj_idx += 1

                       # print(len(vis), len(ses), (len(ses) / len(vis)))


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


def load_exps(exp_f, use_npy=True, rough=True):
        # prepare dataset
    print("loading dataset...")
    dataset = []
    dataset_rough = []
    #""
    if use_npy:
        train_list = np.load('data/im_attention_dataset/dataset_list.npy', allow_pickle=True)
        exp_lines = train_list[:, 1]
        for i in exp_lines:
            exp_single_load = i
            # print(exp_single_load)

            dataset_rough.append(exp_single_load)
            if rough:
                dataset.append(exp_single_load)
            else:
                if abs(exp_single_load[34] > 0.) or abs(exp_single_load[35] > 0.) or abs(exp_single_load[22] > 0.):
                    # print(exp_single)
                    dataset.append(exp_single_load)
                else:
                    p = random.random()
                    # print(p)
                    if p > 0.35:
                        dataset.append(exp_single_load)

    else:
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
                if abs(exp_single_load[34] > 0.) or abs(exp_single_load[35] > 0.) or abs(exp_single_load[22] > 0.):
                    # print(exp_single)
                    dataset.append(exp_single_load)
                else:
                    p = random.random()
                    # print(p)
                    if p > 0.65:
                        dataset.append(exp_single_load)

    return np.array(dataset)


def generate_dataset(obs_mode='summer', pressure_mode='telescope', pressure_abs=True, rough=True):

    # plot_v_field(dataset_rough)
    # print(exp_single_load)
    if obs_mode == "summer":
        dataset = load_exps(exp_f='data/summer_data', rough=rough, use_npy=True)
        # dataset_rough = load_exps(exp_f='/home/wenyan/Desktop/exps/ddpg_v0/ddpg_exps', rough=True)  # load_exps(rough=True)
        print(len(dataset[0]))
        print(dataset.shape)
        if pressure_mode == 'wheel':
            dx = dataset[:, [1, 2, 16, 17]]
            # dx = preprocessing.scale(dx, axis=1)
            dy = dataset[:, [34, 35, 22]]
        elif pressure_mode == 'telescope':
            dx = dataset[:, [1, 2, 8, 9]]
            # dx = preprocessing.scale(dx, axis=1)
            dy = dataset[:, [34, 35, 22]]
        elif pressure_mode == 'wheel_telescope':
            dx = dataset[:, [1, 2, 8,  16, 17]]  # boom, bucket, telescope7,telescope8, wheel, wheel
            # dx = preprocessing.scale(dx, axis=1)
            dy = dataset[:, [34, 35, 22]]
        elif pressure_mode == 'attention':
            dx_a = dataset[:, [1, 2, 8,  16, 17, 6, 7, 13, 25, 30]]
            # boom, bucket, telescope7, wheel, wheel, WorkHydraulicsPressureSensorLift5,
            # WorkHydraulicsPressureSensorLift6, TransmissionPressureSensor14,
            # HSTPumpAngle, PumpFeedbackUp
            dx = dataset[:, [1, 2, 8,  16, 17]]  # boom, bucket, telescope7,telescope8, wheel, wheel
            dy = dataset[:, [34, 35, 22]]
            print(dy)

    elif obs_mode == "autumn":
        dataset = load_exps(exp_f='data/autumn_data', rough=rough, use_npy=False)
        # dataset_rough = load_exps(exp_f='/home/wenyan/Desktop/exps/all_obs/vis+sens', rough=True)  # load_exps(rough=True)

        if pressure_mode == 'wheel':
            dx = dataset[:, [19, 20, 35, 36]]
            # dx = preprocessing.scale(dx, axis=1)
            dy = dataset[:, [52, 53, 41]]
        elif pressure_mode == 'telescope':
            dx = dataset[:, [19, 20, 27, 28]]
            # dx = preprocessing.scale(dx, axis=1)
            dy = dataset[:, [52, 53, 41]]
        elif pressure_mode == 'wheel_telescope':
            dx = dataset[:, [19, 20, 27,  35, 36]]  # boom, bucket, telescope7,telescope8, wheel, wheel
            # dx = preprocessing.scale(dx, axis=1)
            dy = dataset[:, [52, 53, 41]]

        elif pressure_mode == 'attention':
            dx_a = dataset[:, [19, 20, 27, 35, 36, 25, 26, 32, 44, 49]]  # boom, bucket, telescope7,telescope8, wheel, wheel
            dx = dataset[:, [19, 20, 27,  35, 36]]  # boom, bucket, telescope7,telescope8, wheel, wheel
            # dx = preprocessing.scale(dx, axis=1)
            dy = dataset[:, [52, 53, 41]]

    if pressure_abs:
        if pressure_mode == 'wheel':
            nx = np.zeros((len(dataset), 3))
            nx[:, 0] = dx[:, 0]
            nx[:, 1] = dx[:, 1]
            nx[:, 2] = (abs(dx[:, 2] - dx[:, 3])) / 100.
            dx = nx
        elif pressure_mode == 'wheel_telescope':
            nx = np.zeros((len(dataset), 4))
            nx[:, 0] = dx[:, 0]
            nx[:, 1] = dx[:, 1]
            nx[:, 2] = (dx[:, 2]) / 100.
            nx[:, 3] = (abs(dx[:, 4] - dx[:, 3])) / 100.
            dx = nx
        elif pressure_mode == 'attention':
            nx = np.zeros((len(dataset), 9))
            nx[:, 0] = dx_a[:, 0]
            nx[:, 1] = dx_a[:, 1]
            nx[:, 2] = (dx_a[:, 2]) / 100.
            nx[:, 3] = (abs(dx_a[:, 4] - dx_a[:, 3])) / 100.
            nx[:, 4:] = nx[:, 4:] / 100.
            dx_a = nx

            nx = np.zeros((len(dataset), 4))
            nx[:, 0] = dx[:, 0]
            nx[:, 1] = dx[:, 1]
            nx[:, 2] = (dx[:, 2]) / 100.
            nx[:, 3] = (abs(dx[:, 4] - dx[:, 3])) / 100.
            dx = nx

    dy = dy.astype(np.int8)
    dy[:, 0] = convert_uint_sensor(dy[:, 0])
    dy[:, 1] = convert_uint_sensor(dy[:, 1])
    dy = dy / action_scale
    print(dy)

    if pressure_mode == 'attention':
        return dx_a, dx, dy

    else:
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


def plot_states_trajs(obs_mode='autumn', pressure_mode='telescope', len=1):
    import matplotlib.pylab as plt
    from mpl_toolkits.mplot3d.axes3d import Axes3D

    fig = plt.figure()
    ax = Axes3D(fig)

    if obs_mode == "autumn":
        traj_files = glob('data/autumn_data_trajs/*')
    elif obs_mode == "summer":
        traj_files = glob('data/summer_data_trajs/*')

    for idx, sensor_f in enumerate(traj_files[10:len]):
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

            # xscaler = Normalizer().fit(nx)
            # nx = xscaler.transform(nx)
            # yscaler = Normalizer().fit(dy)
            # dy = yscaler.transform(dy)
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


def annimate_states_trajs(obs_mode='summer', pressure_mode='telescope', len=2):
    from matplotlib import pyplot as plt
    from matplotlib import animation
    from mpl_toolkits.mplot3d import Axes3D

    # fig = plt.figure()
    # ax = Axes3D(fig)

    if obs_mode == "autumn":
        traj_files = glob('data/autumn_data_trajs/*')
    elif obs_mode == "summer":
        traj_files = glob('data/summer_data_trajs/*')

    for idx, sensor_f in enumerate(traj_files[:len]):
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

        X = dx[:, 0]
        Y = dx[:, 1]
        Z = dx[:, 3]  # abs(dx[:, 2] - dx[:, 3])
        print(X, Y, Z)
        a = X.size
        print(a)
        for _ in range(Z.size):
            print(Z[_])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        seg, = ax.plot([], [], [], lw=1)
        """
        ax.set_xlim3d(min(X), max(X))
        ax.set_ylim3d(min(Y), max(Y))
        ax.set_zlim3d(min(Z), max(Z))
        """
        #ax.plot3D(X, Y, Z)

        x, y, z = np.meshgrid(np.arange(min(X), max(X), (min(X) - max(X)) / X.size),
                              np.arange(min(Y), max(Y), (min(Y) - max(Y)) / Y.size),
                              np.arange(min(Z), max(Z), (min(Z) - max(Z)) / Z.size))
        X = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
        Y = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
        Z = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
             np.sin(np.pi * z))

        ax.quiver(x, y, z, X, Y, Z, length=0.1, normalize=True)

        """def init():
            return seg,

        def updateFigure(t):
            p1 = X[t]
            p2 = Y[t]
            p3 = Z[t]
            seg._verts3d = (p1, p2, p3)
            print(t, p1, p2, p3)
            return seg,

        ani = animation.FuncAnimation(
            fig, updateFigure, init_func=init, frames=500, interval=5, blit=True)"""
        plt.show()
        # Save
        # anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    # plt.show()


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

# convert_csv_2_array_summer()
# generate_dataset(obs_mode='autumn')
# convert_csv_2_array_summer()
# convert_csv_2_array_autumn_files()
# convert_csv_2_array_summer_files()
# plot_states_trajs(obs_mode='autumn', len=100)

#annimate_states_trajs(obs_mode='summer', len=10)
