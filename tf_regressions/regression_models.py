import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pickle
import os
from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn.preprocessing import Normalizer
import pandas as pd
from sklearn.metrics import mean_squared_error

from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM, Dropout, Lambda, Multiply
from keras.models import model_from_json

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import time
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from AdaBound import AdaBoundOptimizer
from AMSGrad import AMSGrad


class Extra_Attention_MLP_tf(object):
    """docstring for TF_model"""

    def __init__(self, input_size_a, input_size):
        super(Extra_Attention_MLP_tf, self).__init__()
        print("init NN model...")

        self.saver = None
        self.sess = tf.Session()

        self.input_size = input_size
        self.input_size_a = input_size_a
        self.X = tf.placeholder(tf.float32, shape=(None, input_size), name='x')
        self.X_A = tf.placeholder(tf.float32, shape=(None, input_size_a), name='x_a')
        self.Y = tf.placeholder(tf.float32, shape=(None, 3), name='y')

        with tf.variable_scope('NN'):
            self.y_pred = self._build_net(self.X_A, self.X, scope='eval_net', trainable=True)
            self.supervised_loss = tf.reduce_mean(tf.square(self.Y - self.y_pred))

            self.supervised_optimizer = tf.train.AdamOptimizer(learning_rate=0.01)  # define optimizer # play around with learning rate
            # self.supervised_train_op = self.supervised_optimizer.minimize(self.supervised_loss)  # minimize losss
            # self.supervised_train_op = AdaBoundOptimizer(learning_rate=0.01, final_lr=0.1,
            #                                             beta1=0.9, beta2=0.999, amsbound=False).minimize(self.supervised_loss)
            self.supervised_train_op = AMSGrad(learning_rate=0.01, beta1=0.9, beta2=0.99,
                                               epsilon=1e-8).minimize(self.supervised_loss)
            #self.supervised_train_op = tf.train.GradientDescentOptimizer(0.01).minimize(self.supervised_loss)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.graph = tf.get_default_graph()

    def _build_net(self, input_a, input_x, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)

            # attention mask learning
            mask_x = tf.layers.dense(input_a, 64, activation=tf.nn.relu, name='l1',
                                     trainable=True)
            mask_x = tf.layers.dense(mask_x, 128, activation=tf.nn.relu, name='l2',
                                     trainable=True)
            mask_x = tf.layers.dense(mask_x, 64, activation=tf.nn.relu, name='l3',
                                     trainable=True)
            with tf.variable_scope('mask'):
                mask_x = tf.layers.dense(mask_x, self.input_size, activation=tf.nn.sigmoid, kernel_initializer=init_w,
                                         name='mask', trainable=True)

            # multiply the attention with initial inputs
            # Fusing
            with tf.name_scope("fusing"), tf.variable_scope("fusing"):
                weighted_x = tf.multiply(input_x, mask_x, name="fuse_mul")
                #weighted_x = tf.add(input_x, weighted_x, name="fuse_add")

            weighted_x = tf.layers.dense(weighted_x, 100, activation=tf.nn.relu, name='l5',
                                         trainable=True)
            weighted_x = tf.layers.dense(weighted_x, 100, activation=tf.nn.relu, name='l6',
                                         trainable=True)
            weighted_x = tf.layers.dense(weighted_x, 10, activation=tf.nn.relu, name='l7',
                                         trainable=True)
            with tf.variable_scope('a'):
                y_pred = tf.layers.dense(weighted_x, 3, activation=tf.nn.tanh, kernel_initializer=init_w,
                                         name='a', trainable=True)

        return y_pred

    def train(self, train_x_a, train_x, train_y, iteration=90):
        batch_size = 1024
        for epoch in range(iteration):
            total_batch = int(len(train_x) / batch_size)
            # Loop over all batches

            for i in range(total_batch):
                ids = np.random.choice(train_x.shape[0], batch_size, replace=False)
                batch_xs = train_x[ids]
                batch_xas = train_x_a[ids]
                batch_ys = train_y[ids]

                # print(batch_ys[0], batch_ys[0])
                # Run optimization op (backprop) and cost op (to get loss value)
                feed_dict = {self.X_A: batch_xas, self.X: batch_xs, self.Y: batch_ys}
                self.sess.run(self.supervised_train_op, feed_dict)
                print(epoch, i, "loss:", self.supervised_loss.eval(feed_dict, session=self.sess))

        print("Optimization Finished!")

    def predict(self, s_a, s):
        s = s[np.newaxis, :]    # single state
        s_a = s_a[np.newaxis, :]    # single state
        return self.sess.run(self.y_pred, feed_dict={self.X_A: s_a, self.X: s})[0]  # single action

    def save_model(self, model_dir):
        self.saver.save(
            self.sess, model_dir, write_meta_graph=False)

    def restore_model(self, model_dir):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))

    def get_mask(self):
        mask = graph.get_tensor_by_name('output:0')
        return mask


class Attention_MLP_keras(object):
    """docstring for  Attention_MLP"""

    def __init__(self, x_dim):
        super(Attention_MLP_keras, self).__init__()

        self.model = self.define_model(x_dim)

    def define_model(self, x_dim):
        # specify input shape
        input_x = Input(shape=(x_dim,))
        #"""
        # the first mask branch generate attention
        mask_x = Dense(128, activation="relu")(input_x)  # x = BatchNormalization(axis=chanDim)(x)
        #mask_x = Dropout(0.1)(mask_x)
        mask_x = Dense(256, activation="relu")(mask_x)  # x = BatchNormalization(axis=chanDim)(x)
        mask_x = Dropout(0.1)(mask_x)
        mask_x = Dense(128, activation="relu")(mask_x)  # x = BatchNormalization(axis=chanDim)(x)
        mask_x = Dropout(0.1)(mask_x)
        mask_x = Dense(x_dim, activation="sigmoid")(mask_x)
        #mask_x = Model(inputs=inputA, outputs=mask_x)

        # multiply the attention with initial inputs
        # Attention: (1 + output_soft_mask) * output_trunk
        mask_x = Lambda(lambda x: x + 1)(mask_x)
        weighted_x = Multiply()([mask_x, input_x])  #
        #"""
        x = Dense(200, activation="relu")(weighted_x)  # x = BatchNormalization(axis=chanDim)(x)
        x = Dropout(0.1)(x)
        x = Dense(100, activation="relu")(x)  # x = BatchNormalization(axis=chanDim)(x)
        x = Dropout(0.1)(x)
        x = Dense(10, activation="relu")(x)  # x = BatchNormalization(axis=chanDim)(x)
        #x = Dropout(0.25)(x)
        x = Dense(3, activation="tanh")(x)  # x = BatchNormalization(axis=chanDim)(x)
        model = Model(inputs=input_x, outputs=x)

        return model

    def train(self, x_batch, y_batch, x_dim=30,
              test_len=2000):

        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.fit(x_batch[test_len:], y_batch[test_len:], epochs=200, batch_size=128, verbose=2)

        self.save_model(self.model)

    def save_model(self, model):
        # serialize model to JSON
        model_json = model.to_json()
        with open("weights/Attention_ML_kerasP/attention_mlp_avant_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("weights/Attention_MLP_keras/attention_mlp_avant_model.h5")
        print("Saved model to disk")
        return

    def load_keras_model(self, path):
        # load json and create model
        json_file = open(path + 'attention_mlp_avant_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(path + "attention_mlp_avant_model.h5")
        print("Loaded model from disk")
        return loaded_model


class Attention_MLP_tf(object):
    """docstring for TF_model"""

    def __init__(self, input_size):
        super(Attention_MLP_tf, self).__init__()
        print("init NN model...")

        self.saver = None
        self.sess = tf.Session()

        self.input_size = input_size
        self.X = tf.placeholder(tf.float32, shape=(None, input_size), name='x')
        self.Y = tf.placeholder(tf.float32, shape=(None, 3), name='y')

        with tf.variable_scope('NN'):
            self.y_pred = self._build_net(self.X, scope='eval_net', trainable=True)
            self.supervised_loss = tf.reduce_mean(tf.square(self.Y - self.y_pred))

            self.supervised_optimizer = tf.train.AdamOptimizer(learning_rate=0.01)  # define optimizer # play around with learning rate
            # self.supervised_train_op = self.supervised_optimizer.minimize(self.supervised_loss)  # minimize losss
            # self.supervised_train_op = AdaBoundOptimizer(learning_rate=0.01, final_lr=0.1,
            #                                             beta1=0.9, beta2=0.999, amsbound=False).minimize(self.supervised_loss)
            self.supervised_train_op = AMSGrad(learning_rate=0.01, beta1=0.9, beta2=0.99,
                                               epsilon=1e-8).minimize(self.supervised_loss)
            #self.supervised_train_op = tf.train.GradientDescentOptimizer(0.01).minimize(self.supervised_loss)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.graph = tf.get_default_graph()

    def _build_net(self, input_x, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)

            # attention mask learning
            mask_x = tf.layers.dense(input_x, 64, activation=tf.nn.relu, name='l1',
                                     trainable=True)
            mask_x = tf.layers.dense(mask_x, 128, activation=tf.nn.relu, name='l2',
                                     trainable=True)
            mask_x = tf.layers.dense(mask_x, 64, activation=tf.nn.relu, name='l3',
                                     trainable=True)
            with tf.variable_scope('mask'):
                mask_x = tf.layers.dense(mask_x, self.input_size, activation=tf.nn.sigmoid,  # kernel_initializer=init_w,
                                         name='mask', trainable=True)

            # multiply the attention with initial inputs
            # Fusing
            with tf.name_scope("fusing"), tf.variable_scope("fusing"):
                weighted_x = tf.multiply(input_x, mask_x, name="fuse_mul")
                #weighted_x = tf.add(input_x, weighted_x, name="fuse_add")

            weighted_x = tf.layers.dense(weighted_x, 100, activation=tf.nn.relu, name='l5',
                                         trainable=True)
            weighted_x = tf.layers.dense(weighted_x, 100, activation=tf.nn.relu, name='l6',
                                         trainable=True)
            weighted_x = tf.layers.dense(weighted_x, 10, activation=tf.nn.relu, name='l7',
                                         trainable=True)
            with tf.variable_scope('a'):
                y_pred = tf.layers.dense(weighted_x, 3, activation=tf.nn.tanh,  # kernel_initializer=init_w,
                                         name='a', trainable=True)

        return y_pred

    def train(self, train_x, train_y, iteration=90):
        batch_size = 1024
        for epoch in range(iteration):
            total_batch = int(len(train_x) / batch_size)
            # Loop over all batches

            for i in range(total_batch):
                ids = np.random.choice(train_x.shape[0], batch_size, replace=False)
                batch_xs = train_x[ids]
                batch_ys = train_y[ids]

                # print(batch_ys[0], batch_ys[0])
                # Run optimization op (backprop) and cost op (to get loss value)
                feed_dict = {self.X: batch_xs, self.Y: batch_ys}
                self.sess.run(self.supervised_train_op, feed_dict)
                print(epoch, i, "loss:", self.supervised_loss.eval(feed_dict, session=self.sess))

        print("Optimization Finished!")

    def predict(self, s):
        s = s[np.newaxis, :]    # single state
        return self.sess.run(self.y_pred, feed_dict={self.X: s})[0]  # single action

    def save_model(self, model_dir):
        self.saver.save(
            self.sess, model_dir, write_meta_graph=False)

    def restore_model(self, model_dir):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))

    def get_mask(self):
        mask = graph.get_tensor_by_name('output:0')
        return mask


class LSTM_model(object):
    """docstring for LSTM_model"""

    def __init__(self, x_dim):
        super(LSTM_model, self).__init__()

        self.timesteps = 5
        self.model = self.define_model(x_dim)

    def define_model(self, x_dim):
        # create and fit the LSTM network
        model = Sequential()
        """
	    model.add(LSTM(5, input_shape=(x_dim, look_back), return_sequences=False))
	    model.add(Dense(3))
	    """
        model.add(LSTM(20,  # return_sequences=True,
                       input_shape=(self.timesteps, x_dim)))  # returns a sequence of vectors of dimension 32
       # model.add(LSTM(5))  # returns a sequence of vectors of dimension 32
        model.add(Dense(3))

        return model

    def train(self, x_batch, y_batch, x_dim=30,
              test_len=2000):

        x_batch, y_batch = self.create_lstm_dataset(x_batch, y_batch)

        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.fit(x_batch[test_len:], y_batch[test_len:], epochs=60, batch_size=128, verbose=2)

        self.save_model(self.model)

    def create_lstm_dataset(self, X, Y, look_back=5):
        dataX, dataY = [], []
        for i in range(len(X) - look_back - 1):
            y = Y[i + look_back - 1]
            x = X[i:(i + look_back)]

            # print(y)
            dataX.append(x)
            dataY.append(y)
        return np.array(dataX), np.array(dataY)

    def save_model(self, model):
        # serialize model to JSON
        model_json = model.to_json()
        with open("weights/LSTM/lstm_avant_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("weights/LSTM/lstm_avant_model.h5")
        print("Saved model to disk")
        return

    def load_keras_model(self, path):
        # load json and create model
        json_file = open(path + 'lstm_avant_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(path + "lstm_avant_model.h5")
        print("Loaded model from disk")
        return loaded_model


class TF_model(object):
    """docstring for TF_model"""

    def __init__(self, input_size):
        super(TF_model, self).__init__()
        print("init NN model...")

        self.saver = None
        self.sess = tf.Session()

        self.X = tf.placeholder(tf.float32, shape=(None, input_size), name='x')
        self.Y = tf.placeholder(tf.float32, shape=(None, 3), name='y')

        with tf.variable_scope('NN'):
            self.y_pred = self._build_net(self.X, scope='eval_net', trainable=True)
            self.supervised_loss = tf.reduce_mean(tf.square(self.Y - self.y_pred))

            self.supervised_optimizer = tf.train.AdamOptimizer(learning_rate=0.01)  # define optimizer # play around with learning rate
            # self.supervised_train_op = self.supervised_optimizer.minimize(self.supervised_loss)  # minimize losss
            self.supervised_train_op = AdaBoundOptimizer(learning_rate=0.01, final_lr=0.1,
                                                         beta1=0.9, beta2=0.999, amsbound=False).minimize(self.supervised_loss)
            # self.supervised_train_op = AMSGrad(learning_rate=0.01, beta1=0.9, beta2=0.99,
            #                                  epsilon=1e-8).minimize(self.supervised_loss)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def _build_net(self, x, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)

            # net = tf.layers.dense(x, 5, activation=tf.nn.relu, name='l3',
            #                      trainable=True)
            net = tf.layers.dense(x, 100, activation=tf.nn.relu, name='l5',
                                  trainable=True)
            net = tf.layers.dense(net, 100, activation=tf.nn.relu, name='l6',
                                  trainable=True)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu, name='l7',
                                  trainable=True)
            with tf.variable_scope('a'):
                y_pred = tf.layers.dense(net, 3, activation=tf.nn.tanh, kernel_initializer=init_w,
                                         name='a', trainable=True)
                # Scale output to -action_bound to action_bound

        return y_pred

    def train(self, train_x, train_y, iteration=90):
        batch_size = 1024
        for epoch in range(iteration):
            total_batch = int(len(train_x) / batch_size)
            # Loop over all batches

            for i in range(total_batch):
                ids = np.random.choice(train_x.shape[0], batch_size, replace=False)
                batch_xs = train_x[ids]
                batch_ys = train_y[ids]

                # print(batch_ys[0], batch_ys[0])
                # Run optimization op (backprop) and cost op (to get loss value)
                feed_dict = {self.X: batch_xs, self.Y: batch_ys}
                self.sess.run(self.supervised_train_op, feed_dict)
                print(i, "loss:", self.supervised_loss.eval(feed_dict, session=self.sess))

        print("Optimization Finished!")

    def predict(self, s):
        s = s[np.newaxis, :]    # single state
        return self.sess.run(self.y_pred, feed_dict={self.X: s})[0]  # single action

    def save_model(self, model_dir):
        self.saver.save(
            self.sess, model_dir, write_meta_graph=False)

    def restore_model(self, model_dir):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))


class RF_model(object):
    """docstring for RF_model"""

    def __init__(self):
        super(RF_model, self).__init__()
        self.model = RandomForestRegressor(n_estimators=20, max_leaf_nodes=30)

    def train(self, dx, dy):
        self.model.fit(dx[:], dy[:])
        self.save_rf()

        # Extract single tree
        print(self.model.estimators_)
        estimator = self.model.estimators_[0]
        from sklearn.tree import export_graphviz  # Export as dot file
        export_graphviz(estimator,
                        out_file='tree.dot',
                        rounded=True, proportion=False,
                        precision=2, filled=True)
        # Convert to png
        from subprocess import call
        call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

        # Display in jupyter notebook
        from IPython.display import Image
        Image(filename='tree.png')
        return

    def save_rf(self, pkl_filename='weights/RF/rf.pkl'):
        # Save to file in the current working directory
        with open(pkl_filename, 'wb') as file:
            pickle.dump(self.model, file)
        return

    def load_rf(self, pkl_filename='weights/RF/rf.pkl'):
        print('load file')
        with open(pkl_filename, 'rb') as file:
            self.model = pickle.load(file)
        return


def test_TF_NN():
    NN = TF_model(3)
    NN.supervised_learn(x_batch[test_len:], y_batch[test_len:])
    NN.save_model('tmp/')
    NN.restore_model('tmp/')
    y_pred_batch = NN.sess.run(NN.y_pred, {NN.X: x_batch[:test_len]})
    yps = []
    yps.append(y_pred_batch)
    show_figures(y_pred_batch, y_batch[:test_len])

    tmp_d = np.array([1, 1, 1])
    tmp = NN.prediction(tmp_d)
    print(tmp)
