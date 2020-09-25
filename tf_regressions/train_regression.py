import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pickle
import os
from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn.preprocessing import Normalizer

from sklearn.metrics import mean_squared_error
import pandas as pd

from regression_models import *
from utils import *
import pickle


def train_test_NN(type='train', dataset_type='summer', sensor_type='wheel_telescope', pressure_abs=True, iteration=100):
    if sensor_type == 'attention':
        x_batch, _, y_batch = generate_dataset(obs_mode=dataset_type, pressure_mode=sensor_type, rough=True)
        x_dim = np.shape(x_batch)[1]
    else:
        x_batch, y_batch = generate_dataset(obs_mode=dataset_type, pressure_abs=pressure_abs, pressure_mode=sensor_type, rough=True)
        x_dim = np.shape(x_batch)[1]

    print("generate normalizer")
    channel = False
    if channel:
        x_batch_tmp = np.transpose(x_batch)
        scaler = Normalizer().fit(x_batch_tmp)
        # scaler = pickle.load(open('scalerfile', 'rb'))
        x_batch_tmp = scaler.transform(x_batch_tmp)
        x_batch = np.transpose(x_batch_tmp)
    else:
        scaler = Normalizer().fit(x_batch)
        # scaler = pickle.load(open('scalerfile', 'rb'))
        x_batch = scaler.transform(x_batch)

    #########################################
    NN = TF_model(x_dim)

    print("start training")

    test_len = 10000
    if type == 'train':
        NN.train(x_batch[test_len:], y_batch[test_len:], iteration=iteration)
        NN.save_model('weights/NN/test.ckpt')
    else:
        NN.restore_model('weights/NN/')
    # NN.restore_model('tmp/')
    y_pred_batch = NN.sess.run(NN.y_pred, {NN.X: x_batch[:test_len]})
    yps = []
    yps.append(y_pred_batch)
    # testScore = math.sqrt(mean_squared_error(y_batch[:test_len], yps[:]))
    # print('Test Score: %.2f RMSE' % (testScore))
    testPredict = y_pred_batch

    testScore = math.sqrt(mean_squared_error(y_batch[:test_len], testPredict[:]))
    print('Test Score: %.2f RMSE' % (testScore))
    show_figures(y_pred_batch[:], y_batch[:test_len])

    # tmp_d = np.array([1, 1, 1])
    # tmp = NN.prediction(tmp_d)
    # print(tmp)


def train_test_LSTM(type="train", dataset_type='summer', sensor_type='wheel_telescope', pressure_abs=True):
    x_batch, y_batch = generate_dataset(obs_mode=dataset_type, pressure_abs=pressure_abs, pressure_mode=sensor_type,  rough=True)
    x_dim = np.shape(x_batch)[1]
    print(x_dim)

    print("generate normalizer")
    scaler = Normalizer().fit(x_batch)

    # scaler = pickle.load(open('scalerfile', 'rb'))
    x_batch = scaler.transform(x_batch)

    test_len = 10000
    #########################################

    lstm_model = LSTM_model(x_dim=x_dim)
    if type == "train":
        lstm_model.train(x_batch[test_len:], y_batch[test_len:], x_dim=x_dim,
                         test_len=test_len)
    else:
        lstm_model.model = lstm_model.load_keras_model('weights/LSTM/')
    ##############################################
    x_batch, y_batch = lstm_model.create_lstm_dataset(x_batch, y_batch)
    testPredict = lstm_model.model.predict(x_batch[:test_len])

    testScore = math.sqrt(mean_squared_error(y_batch[:test_len], testPredict[:]))
    print('Test Score: %.2f RMSE' % (testScore))
    show_figures(testPredict[:], y_batch[:test_len])


def train_test_RF(type="train", dataset_type='summer', sensor_type='wheel_telescope', pressure_abs=True):
    x_batch, y_batch = generate_dataset(obs_mode=dataset_type, pressure_abs=pressure_abs, pressure_mode=sensor_type,  rough=True)
    x_dim = np.shape(x_batch)[1]
    print(x_dim)

    test_len = 1000
    #########################################

    rf_model = RF_model()
    if type == "train":
        rf_model.train(x_batch[test_len:], y_batch[test_len:])
    else:
        rf_model.load_rf()
    #############################################
    testPredict = rf_model.model.predict(x_batch[:test_len])

    testScore = math.sqrt(mean_squared_error(y_batch[:test_len], testPredict[:]))
    print('Test Score: %.2f RMSE' % (testScore))
    show_figures(testPredict[:], y_batch[:test_len])
    print(rf_model.model.get_params(deep=True))

    return


def train_test_AttentionMLP(type="train", dataset_type='summer', sensor_type='wheel_telescope', backend='tf', pressure_abs=True, iteration=100):
    x_batch, y_batch = generate_dataset(obs_mode=dataset_type, pressure_abs=pressure_abs, pressure_mode=sensor_type,  rough=True)
    x_dim = np.shape(x_batch)[1]
    print(x_dim)

    print("generate normalizer")
    scaler = Normalizer().fit(x_batch)

    # scaler = pickle.load(open('scalerfile', 'rb'))
    x_batch = scaler.transform(x_batch)

    test_len = 10000
    #########################################
    if backend == 'keras':
        attention_model = Attention_MLP_keras(x_dim=x_dim)
        if type == "train":
            attention_model.train(x_batch[test_len:], y_batch[test_len:], x_dim=x_dim,
                                  test_len=test_len)
        else:
            attention_model.model = attention_model.load_keras_model('weights/Attention_MLP_keras/')
        ##############################################
        testPredict = attention_model.model.predict(x_batch[:test_len])

        testScore = math.sqrt(mean_squared_error(y_batch[:test_len], testPredict[:]))
        print('Test Score: %.2f RMSE' % (testScore))
        show_figures(testPredict[:], y_batch[:test_len])

    elif backend == 'tf':
        NN = Attention_MLP_tf(x_dim)
        if type == 'train':
            NN.train(x_batch[test_len:], y_batch[test_len:], iteration=iteration)
            NN.save_model('weights/Attention_MLP_tf/test.ckpt')
        else:
            NN.restore_model('weights/Attention_MLP_tf/')
        # NN.restore_model('tmp/')
        y_pred_batch = NN.sess.run(NN.y_pred, {NN.X: x_batch[:test_len]})
        yps = []
        yps.append(y_pred_batch)
        # testScore = math.sqrt(mean_squared_error(y_batch[:test_len], yps[:]))
        # print('Test Score: %.2f RMSE' % (testScore))
        testPredict = y_pred_batch

        testScore = math.sqrt(mean_squared_error(y_batch[:test_len], testPredict[:]))
        print('Test Score: %.2f RMSE' % (testScore))
        show_figures(y_pred_batch[:], y_batch[:test_len])


def train_test_Extra_Attention_MLP(type="train", dataset_type='summer', sensor_type='attention', iteration=100):
    x_a_batch, x_batch, y_batch = generate_dataset(obs_mode=dataset_type, pressure_mode=sensor_type, rough=True)
    x_a_dim = np.shape(x_a_batch)[1]
    x_dim = np.shape(x_batch)[1]
    print(x_a_dim, x_dim)

    print("generate normalizer")
    scaler_a = Normalizer().fit(x_a_batch)
    # scaler = pickle.load(open('scalerfile', 'rb'))
    x_a_batch = scaler_a.transform(x_a_batch)

    scaler = Normalizer().fit(x_batch)
    # scaler = pickle.load(open('scalerfile', 'rb'))
    x_batch = scaler.transform(x_batch)

    test_len = 10000

    NN = Extra_Attention_MLP_tf(x_a_dim, x_dim)
    if type == 'train':
        NN.train(x_a_batch[test_len:], x_batch[test_len:], y_batch[test_len:], iteration=iteration)
        NN.save_model('weights/Attention_MLP_tf/test.ckpt')
    else:
        NN.restore_model('weights/Attention_MLP_tf/')
    # NN.restore_model('tmp/')
    # NN.sess.run(NN.y_pred, {NN.X_A: x_a_batch[:test_len], NN.X: x_batch[:test_len]})
    y_pred_batch = NN.predict(x_a_batch[:test_len], x_batch[:test_len])
    yps = []
    yps.append(y_pred_batch)
    # testScore = math.sqrt(mean_squared_error(y_batch[:test_len], yps[:]))
    # print('Test Score: %.2f RMSE' % (testScore))
    testPredict = y_pred_batch

    testScore = math.sqrt(mean_squared_error(y_batch[:test_len], testPredict[:]))
    print('Test Score: %.2f RMSE' % (testScore))
    show_figures(y_pred_batch[:], y_batch[:test_len])


def test_tmp():
    """
    x_batch, y_batch = generate_dataset(obs_mode='summer', pressure_abs=False, rough=True)
    normalizer_0 = Normalizer().fit([x_batch[:, 0]])
    normalizer_1 = Normalizer().fit([x_batch[:, 1]])
    normalizer_2 = Normalizer().fit([x_batch[:, 2]])
    normalizer_3 = Normalizer().fit([x_batch[:, 3]])

    import pickle
    pickle.dump(normalizer_0, open('scalerfile', 'wb'))
    # scaler = pickle.load(open('scalerfile', 'rb'))
    """

    import argparse

    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--state_type', type=str,
                            default="3_input")  # 3_sens, raw_input
        parser.add_argument('--model', type=str,
                            default="rf")  # train, test, rf-train, rf-test
        parser.add_argument('--load_exp', type=str,
                            default="False")
        return parser.parse_args()

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
        print('rf')
        rf_model = RF_model()
        rf_model.load_rf('0122_weights/' + args.state_type + '/RF/rf.pkl')
        prediction_model = rf_model.model

    test = np.array([[-1.2, 0.2, 100, 1]])
    print(test)
    # test = normalizer.transform(test)
    """
    normalizer_0.transform([[1]])
    test[:, 0] = normalizer_0.transform([test[:, 0]])[0]
    test[:, 1] = normalizer_1.transform([test[:, 1]])[0]
    test[:, 2] = normalizer_2.transform([test[:, 2]])[0]
    test[:, 3] = normalizer_3.transform([test[:, 3]])[0]
    """
    print(test)
    print(prediction_model.predict([test[0]]))


def gen_scaler(pressure_mode='wheel'):
    if pressure_mode != 'attention':
        x_batch, y_batch = generate_dataset(obs_mode='summer', pressure_abs=True,  pressure_mode=pressure_mode, rough=True)
        scaler = Normalizer().fit(x_batch)
        scalerfile = 'scalerfile' + '_' + pressure_mode
        pickle.dump(scaler, open(scalerfile, 'wb'))
    else:
        x_a_batch, x_batch, y_batch = generate_dataset(obs_mode='summer', pressure_abs=True,  pressure_mode=pressure_mode, rough=True)
        scaler = Normalizer().fit(x_a_batch)
        scalerfile = 'scalerfile' + '_' + pressure_mode
        pickle.dump(scaler, open(scalerfile, 'wb'))


def load_scaler():
    scalerfile = 'scalerfile'
    scaler = pickle.load(open(scalerfile, 'rb'))
# test_tmp()

sensors = ['wheel', 'telescope', 'wheel_telescope', 'attention']

train_iteration = 125
#train_test_NN('train', "summer", pressure_abs=True, sensor_type=sensors[2], iteration=train_iteration)
# train_test_NN('test', "summer", pressure_abs=True, sensor_type=sensors[1])
#train_test_NN('test', "autumn", pressure_abs=True, sensor_type=sensors[3])
# train_test_LSTM('train', "summer", pressure_abs=True, sensor_type=sensors[2], iteration=train_iteration)

#"""
# train_test_LSTM('test', "summer", pressure_abs=True, sensor_type=sensors[2])
# train_test_LSTM('test', "autumn", pressure_abs=True, sensor_type=sensors[2])


#train_test_RF('train', "summer", pressure_abs=True, sensor_type=sensors[2])
#train_test_RF('test', "autumn", pressure_abs=True, sensor_type=sensors[2])

#"""
#train_test_AttentionMLP('train', "summer", pressure_abs=True, sensor_type=sensors[2],  backend='tf', iteration=train_iteration)
train_test_AttentionMLP('train', "summer", pressure_abs=True, sensor_type=sensors[2], iteration=train_iteration)
#train_test_AttentionMLP('test', "autumn", pressure_abs=True, sensor_type=sensors[1],  backend='tf')

#train_test_Extra_Attention_MLP('train', "summer", sensor_type=sensors[3],  iteration=train_iteration)
#train_test_Extra_Attention_MLP('test', "autumn", sensor_type=sensors[3],  iteration=train_iteration)

# gen_scaler(pressure_mode=sensors[3])
# load_scaler()
