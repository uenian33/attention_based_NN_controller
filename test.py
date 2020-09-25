#!/usr/bin/env python3

import tensorflow as tf
import shutil

import sys
import numpy as np
import cv2
from threading import Event
import time
import numpy
import csv
import socket
import struct
import argparse


import os
import sklearn
from sklearn.preprocessing import *

from regression_models import *
from lstm_nn_rf_utils import *
import pickle
from mpl_toolkits import mplot3d


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state_type', type=str,
                        default="3_input")  # 3_sens, raw_input
    parser.add_argument('--model', type=str,
                        default="rf")  # train, test, rf-train, rf-test
    parser.add_argument('--load_exp', type=str,
                        default="False")
    return parser.parse_args()


def test_load_scaler():
    args = get_args()

    if args.state_type == '3_input':
        normalizer = pickle.load(open('0122_weights/' + args.state_type + '/scalerfile', 'rb'))
        x_dim = 3
    elif args.state_type == 'raw_input':
        normalizer = pickle.load(open('0122_weights/' + args.state_type + '/scalerfile', 'rb'))
        x_dim = 4

    if args.model == 'lstm':
        prediction_model = LSTM_model(x_dim=x_dim)
        lstm_model.model = lstm_model.load_keras_model('0122_weights/' + args.state_type + '/LSTM/')
    elif args.model == 'nn':
        prediction_model = TF_model(x_dim)
        prediction_model.restore_model('0122_weights/' + args.state_type + '/NN/')
    elif args.model == 'rf':
        rf_model = RF_model()
        rf_model.load_rf('0122_weights/' + args.state_type + '/RF/rf.pkl')
        prediction_model = rf_model.model

    test = np.array([[-0.8, 1.4, 100, 1]])
    print(test)
    test = normalizer.transform(test)[0]
    print(test)
    print(prediction_model.predict(test))


def mannually_generate_traj():

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for _ in range(1):
        tele_traj = []
        boom_traj = []
        bucket_traj = []

        tele_init = random.uniform(7, 15)
        boom_init = random.uniform(48, 56)  # degree
        bucket_init = random.uniform(-0.2, 6)  # degree

        forward_steps = random.randint(45, 65)
        # forward stage
        for i in range(forward_steps):
            tele = tele_init + random.uniform(-2, 2)
            boom = boom_init + random.uniform(-2, 2)
            bucket = bucket_init + random.uniform(-2, 2)

            tele_traj.append(tele)
            boom_traj.append(boom)
            bucket_traj.append(bucket)

        # hit and insert stage
        hit_steps = random.randint(1, 5)
        tele_hit = random.uniform(20, 80)
        boom_movement = random.uniform(0, 5)
        bucket_movement = random.uniform(0, 5)
        for i in range(hit_steps):
            tele = tele_hit + random.uniform(-2, 5)
            boom = boom_init + random.uniform(0.2, 1)
            bucket = bucket_init + random.uniform(0.2, 1)
            tele_traj.append(tele)
            boom_traj.append(boom)
            bucket_traj.append(bucket)

        # rising stage
        tele = tele_traj[-1]
        boom = boom_traj[-1]
        bucket = bucket_traj[-1]

        tele_end = random.randint(10, 18)
        tele_end_min = tele_end + random.randint(-2, 3)
        tele_change = abs(tele_end - tele)
        tele_change_steps = random.randint(8, 25)

        boom_target = random.uniform(50, 65)
        bucket_target = random.uniform(15, 35)

        rising_steps = random.randint(55, 65)
        for i in range(rising_steps):
            tele = tele - (tele_change / tele_change_steps + random.uniform(-1, 3))
            if tele < tele_end_min:
                tele = tele_end_min
            boom = boom + boom_target / rising_steps + random.uniform(-1, 1)
            bucket = bucket + bucket_target / rising_steps + random.uniform(-1, 1)
            tele_traj.append(tele + random.uniform(-2, 2))
            boom_traj.append(boom)
            bucket_traj.append(bucket)

        total_length = forward_steps + hit_steps + rising_steps

        #ax = plt.axes(projection='3d')

        # Data for a three-dimensional line
        zline = bucket_traj
        xline = tele_traj
        yline = boom_traj
        ax.plot3D(xline, yline, zline)
        #inde = np.linspace(0, 1, total_length)
        #plt.plot(inde, tele_traj, alpha=0.8)
    ax.set_xlabel('tele')
    ax.set_ylabel('boom')
    ax.set_zlabel('bucket')
    plt.show()

mannually_generate_traj()
