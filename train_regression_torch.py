import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn.preprocessing import Normalizer
from sklearn.metrics import mean_squared_error
import pandas as pd
from matplotlib import ticker

import torch
import os
import torch.optim as optim

from regression_models_torch import *
from utils import *
from Optimizer_torch import *
from matplotlib.pyplot import cm
import argparse
# torch.set_default_tensor_type('torch.DoubleTensor')


class customized_mse(nn.Module):
    """docstring for customized_mse"""

    def __init__(self):
        super(customized_mse, self).__init__()

    def forward(self, x, y):
        boom = torch.mean(torch.pow((x[:, 0] - y[:, 0]), 2))
        bucket = torch.mean(torch.pow((x[:, 1] - y[:, 1]), 2))
        gas = torch.mean(torch.pow((x[:, 2] - y[:, 2]), 2))

        loss = 0.44 * boom + 0.44 * bucket + 0.22 * gas
        return loss


class SENSOR(Dataset):

    def __init__(self, X, y=None):
        self.X = X
        self.Y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index].copy()), torch.from_numpy(self.Y[index].copy())


class EXTARA_SENSOR(Dataset):

    def __init__(self, XA, X, y=None):
        self.XA = XA
        self.X = X
        self.Y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return torch.from_numpy(self.XA[index].copy()), torch.from_numpy(self.X[index].copy()), torch.from_numpy(self.Y[index].copy())


def train(train=True,
          dataset='summer',
          sensor_type='all',
          model_type='attention',
          use_abs_pressure=True,
          train_epochs=15,
          dual=False,
          ith=None):

    # define save path
    if dual:
        pth_path = 'weights_torch/' + model_type + '_dual_' + sensor_type + '/'
    else:
        pth_path = 'weights_torch/' + model_type + '_' + sensor_type + '/'
    if not os.path.exists(pth_path):
        os.makedirs(pth_path)
    # load all dataset
    if sensor_type == 'attention':
        x_a_batch, x_batch, y_batch = generate_dataset(obs_mode=dataset, pressure_mode=sensor_type, rough=False)
        x_a_dim = np.shape(x_a_batch)[1]
        x_dim = np.shape(x_batch)[1]
        scaler_a = Normalizer().fit(x_a_batch)
        x_a_batch = scaler_a.transform(x_a_batch)
        scaler = Normalizer().fit(x_batch)
        x_batch = scaler.transform(x_batch)

        x_batch = x_a_batch
        x_dim = x_a_dim

        scalerfile = sensor_type
        pickle.dump(scaler, open(scalerfile, 'wb'))
        scalerfile = sensor_type + '_a'
        pickle.dump(scaler, open(scalerfile, 'wb'))
    else:
        x_batch, y_batch = generate_dataset(obs_mode=dataset, pressure_abs=use_abs_pressure, pressure_mode=sensor_type, rough=False)
        x_dim = np.shape(x_batch)[1]
        scaler = Normalizer().fit(x_batch)
        x_batch = scaler.transform(x_batch)

        scalerfile = sensor_type
        pickle.dump(scaler, open(scalerfile, 'wb'))

    split_length = int(len(x_batch) / 20)

    # generate dataloader
    train_dataset = SENSOR(X=x_batch[split_length:], y=y_batch[split_length:])
    valid_dataset = SENSOR(X=x_batch[:split_length], y=y_batch[:split_length])
    # test_dataset = SENSOR(X=x_batch, y=y_batch)

    test_seg = 2500
    test_dataset = SENSOR(X=x_batch[test_seg:], y=y_batch[test_seg:])

    try:
        train_loader = DataLoader(dataset=train_dataset, batch_size=1024, shuffle=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=len(valid_dataset), shuffle=False)
    except:
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=len(valid_dataset), shuffle=False)

    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
    # init model
    if model_type == 'NN':
        model = NN(x_dim).cuda()
    elif model_type == 'attention':
        if dual:
            model = Attention_NN_dual(x_dim).cuda()
        else:
            model = Attention_NN(x_dim).cuda()

    # init optimizer
    # optimizer = AdaBound(model.parameters(), lr=1e-3, final_lr=0.001)
    # optimizer = RAdam(model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    loss_fn = customized_mse()  # nn.MSELoss()
    if train:
        mean_train_losses = []
        mean_valid_losses = []
        valid_acc_list = []
        epochs = train_epochs
        epoch = 0

        while epoch < epochs:  # for epoch in range(epochs):
            print(model_type, sensor_type, epoch)
            model.train()

            train_losses = []
            valid_losses = []
            for i, (x, labels) in enumerate(train_loader):
                x = x.float().cuda()
                labels = labels.float().cuda()

                optimizer.zero_grad()
                outputs = model(x)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            print(np.mean(train_losses), epoch, epochs)
            if np.mean(train_losses) > 0.01200 and epoch > (train_epochs - 2):
                print("continue")
                epochs += 1
            model.eval()

            total = 0
            best_valid = 1000
            with torch.no_grad():
                for i, (test_x, test_labels) in enumerate(valid_loader):
                    test_x = test_x.float().cuda()
                    test_labels = test_labels.float().cuda()
                    test_outputs = model(test_x)
                    loss = loss_fn(test_outputs, test_labels)

                    valid_losses.append(loss.item())

            mean_train_losses.append(np.mean(train_losses))
            mean_valid_losses.append(np.mean(valid_losses))

            if np.mean(valid_losses) < best_valid:
                best_valid = np.mean(valid_losses)
                if ith == None:
                    torch.save(model.state_dict(), pth_path + 'model_best.pth')
                else:
                    torch.save(model.state_dict(), pth_path + 'model_best' + str(ith) + '.pth')

            print('epoch : {}, train loss : {:.4f}, valid loss : {:.4f}'
                  .format(epoch + 1, np.mean(train_losses), np.mean(valid_losses)))
            epoch += 1

        if ith == None:
            torch.save(model.state_dict(), pth_path + 'model.pth')
        else:
            torch.save(model.state_dict(), pth_path + 'model' + str(ith) + '.pth')
    else:
        if ith == None:
            model.load_state_dict(torch.load(pth_path + 'model.pth'))
        else:
            model.load_state_dict(torch.load(pth_path + 'model' + str(ith) + '.pth'))

        model.eval()
        with torch.no_grad():
            for i, (x, labels) in enumerate(test_loader):
                print(i)
                x = x.float().cuda()
                labels = labels.float().cuda()
                test_outputs = model(x)
        if ith == None:
            pdf_path = pth_path + "validation.pdf"
        else:
            pdf_path = pth_path + "validation" + str(ith) + ".pdf"
            print(pdf_path)
        show_figures(test_outputs.cpu().detach().numpy(), y_batch[test_seg:], path=pdf_path)

        if ith == None:
            model.load_state_dict(torch.load(pth_path + 'model_best.pth'))
        else:
            model.load_state_dict(torch.load(pth_path + 'model_best' + str(ith) + '.pth'))
        model.eval()
        with torch.no_grad():
            for i, (x, labels) in enumerate(test_loader):
                print(i)
                x = x.float().cuda()
                labels = labels.float().cuda()
                test_outputs = model(x)

        if ith == None:
            best_pdf_path = pth_path + "best_validation.pdf"
        else:
            best_pdf_path = pth_path + "best_validation" + str(ith) + ".pdf"
        show_figures(test_outputs.cpu().detach().numpy(), y_batch[test_seg:], path=best_pdf_path)

        return model

    # show_figures(test_outputs.cpu().detach().numpy(), y_batch[:split_length])


def train_extra(train=True,
                dataset='summer',
                sensor_type='all',
                model_type='attention',
                use_abs_pressure=True,
                train_epochs=15,
                dual=False,
                ith=None):

    # define save path
    if dual:
        pth_path = 'weights_torch/' + model_type + '_dual_' + sensor_type + '/'
    else:
        pth_path = 'weights_torch/' + model_type + '_' + sensor_type + '/'
    if not os.path.exists(pth_path):
        os.makedirs(pth_path)

    # load all dataset
    x_a_batch, x_batch, y_batch = generate_dataset(obs_mode=dataset, pressure_mode=sensor_type, rough=False)
    x_a_dim = np.shape(x_a_batch)[1]
    x_dim = np.shape(x_batch)[1]
    scaler_a = Normalizer().fit(x_a_batch)
    x_a_batch = scaler_a.transform(x_a_batch)
    scaler = Normalizer().fit(x_batch)
    x_batch = scaler.transform(x_batch)

    split_length = int(len(x_batch) / 20)
    # generate dataloader
    train_dataset = EXTARA_SENSOR(XA=x_a_batch[split_length:], X=x_batch[split_length:], y=y_batch[split_length:])
    valid_dataset = EXTARA_SENSOR(XA=x_a_batch[:split_length], X=x_batch[:split_length], y=y_batch[:split_length])
    test_seg = 2500
    test_dataset = EXTARA_SENSOR(XA=x_a_batch[test_seg:], X=x_batch[test_seg:], y=y_batch[test_seg:])
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

    try:
        train_loader = DataLoader(dataset=train_dataset, batch_size=1024, shuffle=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=len(valid_dataset), shuffle=False)
    except:
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=len(valid_dataset), shuffle=False)

    # init model
    if dual:
        model = Extra_Attention_NN_dual(x_a_dim, x_dim).cuda()
    else:
        model = Extra_Attention_NN(x_a_dim, x_dim).cuda()

    # init optimizer
    # optimizer = AdaBound(model.parameters(), lr=1e-3, final_lr=0.001)
    # optimizer = RAdam(model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    loss_fn = customized_mse()  # nn.MSELoss()
    if train:
        mean_train_losses = []
        mean_valid_losses = []
        valid_acc_list = []
        epochs = train_epochs
        epoch = 0

        while epoch < epochs:  # for epoch in range(epochs):
            model.train()

            train_losses = []
            valid_losses = []
            for i, (xa, x, labels) in enumerate(train_loader):
                xa = xa.float().cuda()
                x = x.float().cuda()
                labels = labels.float().cuda()

                optimizer.zero_grad()
                outputs = model(xa, x)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            if np.mean(train_losses) > 0.01210 and epoch > (train_epochs - 2):
                print("continue", train_losses[-1])
                epochs += 1

            model.eval()

            total = 0
            best_valid = 1000
            with torch.no_grad():
                for i, (test_xa, test_x, test_labels) in enumerate(valid_loader):
                    test_xa = test_xa.float().cuda()
                    test_x = test_x.float().cuda()
                    test_labels = test_labels.float().cuda()
                    test_outputs = model(test_xa, test_x)
                    loss = loss_fn(test_outputs, test_labels)

                    valid_losses.append(loss.item())

            mean_train_losses.append(np.mean(train_losses))
            mean_valid_losses.append(np.mean(valid_losses))

            if np.mean(valid_losses) < best_valid:
                best_valid = np.mean(valid_losses)
                if ith == None:
                    torch.save(model.state_dict(), pth_path + 'model_best.pth')
                else:
                    torch.save(model.state_dict(), pth_path + 'model_best' + str(ith) + '.pth')

            print('epoch : {}, train loss : {:.4f}, valid loss : {:.4f}'
                  .format(epoch + 1, np.mean(train_losses), np.mean(valid_losses)))
            epoch += 1

        if ith == None:
            torch.save(model.state_dict(), pth_path + 'model.pth')
        else:
            torch.save(model.state_dict(), pth_path + 'model' + str(ith) + '.pth')
    else:
        if ith == None:
            model.load_state_dict(torch.load(pth_path + 'model.pth'))
        else:
            model.load_state_dict(torch.load(pth_path + 'model' + str(ith) + '.pth'))
        model.eval()
        with torch.no_grad():
            for i, (test_xa, test_x, test_labels) in enumerate(test_loader):
                test_xa = test_xa.float().cuda()
                test_x = test_x.float().cuda()
                test_labels = test_labels.float().cuda()
                test_outputs = model(test_xa, test_x)

        if ith == None:
            pdf_path = pth_path + "validation.pdf"
        else:
            pdf_path = pth_path + "validation" + str(ith) + ".pdf"
        show_figures(test_outputs.cpu().detach().numpy(), y_batch[test_seg:], path=pdf_path)

        if ith == None:
            model.load_state_dict(torch.load(pth_path + 'model_best.pth'))
        else:
            model.load_state_dict(torch.load(pth_path + 'model_best' + str(ith) + '.pth'))
        model.eval()
        with torch.no_grad():
            for i, (test_xa, test_x, test_labels) in enumerate(test_loader):
                test_xa = test_xa.float().cuda()
                test_x = test_x.float().cuda()
                test_labels = test_labels.float().cuda()
                test_outputs = model(test_xa, test_x)

        if ith == None:
            best_pdf_path = pth_path + "best_validation.pdf"
        else:
            best_pdf_path = pth_path + "best_validation" + str(ith) + ".pdf"
        show_figures(test_outputs.cpu().detach().numpy(), y_batch[test_seg:], path=best_pdf_path)

        return model

    # show_figures(test_outputs.cpu().detach().numpy(), y_batch[:split_length])


def train_test():

    parser = argparse.ArgumentParser("simple_example")
    parser.add_argument('mode', type=str, help='[NN, single_attention, single_extra_attention, dual_attention, dual_extra_attention]')
    args = parser.parse_args()

    simple = False

    """
    for tmp in range(4):

        train(train=False, dataset='summer', sensor_type=sensors[tmp], model_type=models[0], train_epochs=0,)

    import time
    time.sleep(100)
    """

    model_num = 10
    if simple:
        #"""
        train(train=True, dataset='summer', sensor_type=sensors[0], model_type=models[0], train_epochs=200,)
        train(train=False, dataset='autumn', sensor_type=sensors[0], model_type=models[0])
        #"""

        # single attention
        train(dual=False, train=True, dataset='summer', sensor_type=sensors[0], model_type=models[1], train_epochs=200,)
        train(dual=False, train=False, dataset='autumn', sensor_type=sensors[0], model_type=models[1])

        train_extra(dual=False, train=True, dataset='summer', sensor_type=sensors[3], model_type=models[2], train_epochs=200,)
        train_extra(dual=False, train=False, dataset='autumn', sensor_type=sensors[3], model_type=models[2])

        # dual attention
        train(dual=True, train=True, dataset='summer', sensor_type=sensors[0], model_type=models[1], train_epochs=200,)
        train(dual=True, train=False, dataset='autumn', sensor_type=sensors[0], model_type=models[1])

        train_extra(dual=True, train=True, dataset='summer', sensor_type=sensors[3], model_type=models[2], train_epochs=200,)
        train_extra(dual=True, train=False, dataset='autumn', sensor_type=sensors[3], model_type=models[2])
    else:
        if args.mode == 'NN':
            for i in range(model_num):
                for j in range(4):
                    train(ith=i, train=True, dataset='summer', sensor_type=sensors[j], model_type=models[0], train_epochs=200,)
                    train(ith=i, train=False, dataset='autumn', sensor_type=sensors[j], model_type=models[0])

        elif args.mode == 'single_attention':  # single attention
            for i in range(model_num):
                for j in range(4):
                    train(ith=i, dual=False, train=True, dataset='summer', sensor_type=sensors[j], model_type=models[1], train_epochs=200,)
                    train(ith=i, dual=False, train=False, dataset='autumn', sensor_type=sensors[j], model_type=models[1])

        elif args.mode == 'single_extra_attention':
            for i in range(model_num):
                train_extra(ith=i, dual=False, train=True, dataset='summer',
                            sensor_type=sensors[3], model_type=models[2], train_epochs=200,)
                train_extra(ith=i, dual=False, train=False, dataset='autumn', sensor_type=sensors[3], model_type=models[2])

        # dual attention
        elif args.mode == 'dual_attention':
            for i in range(model_num):
                for j in range(4):
                    train(ith=i, dual=True, train=True, dataset='summer', sensor_type=sensors[j], model_type=models[1], train_epochs=200,)
                    train(ith=i, dual=True, train=False, dataset='autumn', sensor_type=sensors[j], model_type=models[1])

        elif args.mode == 'dual_extra_attention':
            for i in range(model_num):
                train_extra(ith=i, dual=True, train=True, dataset='summer', sensor_type=sensors[3], model_type=models[2], train_epochs=200,)
                train_extra(ith=i, dual=True, train=False, dataset='autumn', sensor_type=sensors[3], model_type=models[2])
        else:

            for i in range(model_num):
                for j in range(4):
                    # train(ith=i, train=True, dataset='summer', sensor_type=sensors[j], model_type=models[0], train_epochs=200,)
                    train(ith=i, train=False, dataset='autumn', sensor_type=sensors[j], model_type=models[0])

            for i in range(model_num):
                for j in range(4):
                    # train(ith=i, dual=False, train=True, dataset='summer',
                    # sensor_type=sensors[j], model_type=models[1], train_epochs=200,)
                    train(ith=i, dual=False, train=False, dataset='autumn', sensor_type=sensors[j], model_type=models[1])

            for i in range(model_num):
                # train_extra(ith=i, dual=False, train=True, dataset='summer',
                # sensor_type=sensors[3], model_type=models[2], train_epochs=200,)
                train_extra(ith=i, dual=False, train=False, dataset='autumn', sensor_type=sensors[3], model_type=models[2])

            for i in range(model_num):
                for j in range(4):
                    # train(ith=i, dual=True, train=True, dataset='summer', sensor_type=sensors[j], model_type=models[1], train_epochs=200,)
                    train(ith=i, dual=True, train=False, dataset='autumn', sensor_type=sensors[j], model_type=models[1])

            for i in range(model_num):
                # train_extra(ith=i, dual=True, train=True, dataset='summer',
                # sensor_type=sensors[3], model_type=models[2], train_epochs=200,)
                train_extra(ith=i, dual=True, train=False, dataset='autumn', sensor_type=sensors[3], model_type=models[2])

    # train_extra(ith=1, dual=True, train=False, dataset='autumn', sensor_type=sensors[3], model_type=models[2])
    # train_extra(ith=1, dual=False, train=False, dataset='autumn', sensor_type=sensors[3], model_type=models[0])
    # train(ith=1, train=False, dataset='autumn', sensor_type=sensors[0], model_type=models[1])


def test_mask(model_type='dual_extra_attention'):

    if model_type == 'single_attention':
        x_batch, y_batch = generate_dataset(obs_mode='autumn', pressure_abs=True, pressure_mode=sensors[2], rough=False)
        test_dataset = SENSOR(X=x_batch[8809:9039], y=y_batch[8809:9039])
        test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

        model = Attention_NN(4).cuda()
        pth_path = 'figures/raw/single_attention_wheel_telescope/'
        model.load_state_dict(torch.load(pth_path + 'model35.pth'))

        model.eval()
        with torch.no_grad():
            for i, (test_x, test_labels) in enumerate(test_loader):
                test_x = test_x.float().cuda()
                test_labels = test_labels.float().cuda()
                y_pred, s_m = model(test_x)
        y_pred = y_pred.cpu().detach().numpy()
        s_m = s_m.cpu().detach().numpy()

    elif model_type == 'dual_attention':
        x_batch, y_batch = generate_dataset(obs_mode='autumn', pressure_abs=True, pressure_mode=sensors[2], rough=False)
        test_dataset = SENSOR(X=x_batch[8809:9039], y=y_batch[8809:9039])
        test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

        model = Attention_NN_dual(4).cuda()
        pth_path = 'figures/raw/dual_attention_wheel_telescope/'
        model.load_state_dict(torch.load(pth_path + 'model5.pth'))

        model.eval()
        with torch.no_grad():
            for i, (test_x, test_labels) in enumerate(test_loader):
                test_x = test_x.float().cuda()
                test_labels = test_labels.float().cuda()
                y_pred, s_m, a_m = model(test_x)
        y_pred = y_pred.cpu().detach().numpy()
        s_m = s_m.cpu().detach().numpy()
        a_m = a_m.cpu().detach().numpy()

    elif model_type == 'single_extra_attention':
        x_a_batch, x_batch, y_batch = generate_dataset(obs_mode='autumn', pressure_mode=sensors[3], rough=False)
        test_dataset = EXTARA_SENSOR(XA=x_a_batch[8809:9039], X=x_batch[8809:9039], y=y_batch[8809:9039])
        test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

        model = Extra_Attention_NN(9, 4).cuda()
        pth_path = 'figures/raw/single_extra_attention_attention/'
        model.load_state_dict(torch.load(pth_path + 'model5.pth'))

        model.eval()
        with torch.no_grad():
            for i, (test_xa, test_x, test_labels) in enumerate(test_loader):
                test_xa = test_xa.float().cuda()
                test_x = test_x.float().cuda()
                test_labels = test_labels.float().cuda()
                y_pred, s_m = model(test_xa, test_x)
        y_pred = y_pred.cpu().detach().numpy()
        s_m = s_m.cpu().detach().numpy()

    elif model_type == 'dual_extra_attention':
        x_a_batch, x_batch, y_batch = generate_dataset(obs_mode='autumn', pressure_mode=sensors[3], rough=False)
        test_dataset = EXTARA_SENSOR(XA=x_a_batch[8809:9039], X=x_batch[8809:9039], y=y_batch[8809:9039])
        test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

        model = Extra_Attention_NN_dual(9, 4).cuda()
        pth_path = 'figures/raw/dual_extra_attention_attention/'
        model.load_state_dict(torch.load(pth_path + 'model1.pth'))

        model.eval()
        with torch.no_grad():
            for i, (test_xa, test_x, test_labels) in enumerate(test_loader):
                test_xa = test_xa.float().cuda()
                test_x = test_x.float().cuda()
                test_labels = test_labels.float().cuda()
                y_pred, s_m, a_m = model(test_xa, test_x)
        y_pred = y_pred.cpu().detach().numpy()
        s_m = s_m.cpu().detach().numpy()
        a_m = a_m.cpu().detach().numpy()
    show_figures(y_pred,  y_batch[8809:9039])
    # print(s_m)
    # print(a_m)

    anchors = [50, 114, 170, 213]
    sensor_names = ['boom angle', 'bucket angle', 'pressure W', 'pressure T']
    y_names = ['T1', 'T2', 'T3', 'T4']
    act_names = ['boom action', 'bucket action', 'gas action']

    vis_sens = np.array([s_m[50], s_m[114], s_m[170], s_m[213]])
    vis_act = np.array([a_m[50], a_m[114], a_m[170], a_m[213]])
    valfmt = ticker.StrMethodFormatter('{x:.3f}')
    for idx, anc in enumerate(anchors):
        fig, ax = plt.subplots()
        a = vis_sens
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(sensor_names)
        ax.set_yticks(np.arange(4))
        ax.set_yticklabels(y_names)
        # Rotate the tick labels and set their alignment.
        #plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
        # Loop over data dimensions and create text annotations.
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if a[i, j] > 0.5:
                    text = ax.text(j, i, valfmt(a[i, j]), ha="center", va="center", color="w")
                else:
                    text = ax.text(j, i, valfmt(a[i, j]), ha="center", va="center", color="black")

        im = ax.imshow(a, cmap=cm.Blues)
        plt.axis('off')
        plt.savefig('sensor_mask_' + str(idx) + '.pdf')
        plt.show()

        fig, ax = plt.subplots()
        a = vis_act
        ax.set_xticks(np.arange(len(act_names)))
        ax.set_xticklabels(act_names)
        # Rotate the tick labels and set their alignment.
        #plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
        # Loop over data dimensions and create text annotations.
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if a[i, j] > 0.5:
                    text = ax.text(j, i, valfmt(a[i, j]), ha="center", va="center", color="w")
                else:
                    text = ax.text(j, i, valfmt(a[i, j]), ha="center", va="center", color="black")

        im = ax.imshow(a, cmap=cm.Oranges)
        plt.axis('off')
        # plt.yticks([])
        plt.savefig('act_mask_' + str(idx) + '.pdf')
        plt.show()


sensors = ['wheel', 'telescope', 'wheel_telescope', 'attention']
models = ['NN', 'attention', 'extra_attention']

x_batch, y_batch = generate_dataset(obs_mode='summer', pressure_abs=True, pressure_mode=sensors[2], rough=False)
gt = y_batch
count = 0
dual_count = 0
three_count = 0
zero_count = 0
for idx, g in enumerate(gt):
    if (abs(gt[idx, 2]) > 0.01 and abs(gt[idx, 1]) > 0.01) or (abs(gt[idx, 0]) > 0.01 and abs(gt[idx, 1]) > 0.01) or (abs(gt[idx, 0]) > 0.01 and abs(gt[idx, 2]) > 0.01) or (abs(gt[idx, 0]) > 0.01 and abs(gt[idx, 2]) > 0.01 and abs(gt[idx, 1]) > 0.01):
        count += 1
    if (abs(gt[idx, 2]) > 0.01 and abs(gt[idx, 1]) > 0.01) or (abs(gt[idx, 0]) > 0.01 and abs(gt[idx, 1]) > 0.01) or (abs(gt[idx, 0]) > 0.01 and abs(gt[idx, 2]) > 0.01):
        dual_count += 1
    if (abs(gt[idx, 0]) > 0.01 and abs(gt[idx, 2]) > 0.01 and abs(gt[idx, 1]) > 0.01):
        three_count += 1
    if (abs(gt[idx, 0]) == 0. and abs(gt[idx, 2]) == 0. and abs(gt[idx, 1]) == 0.):
        zero_count += 1
print(zero_count, len(gt[:, 0]) - count, dual_count, three_count)
train_extra(ith=1, dual=True, train=False, dataset='autumn', sensor_type=sensors[3], model_type=models[2])

# test_mask()
"""
train(train=True, dataset='summer', sensor_type=sensors[0], model_type=models[0], train_epochs=200,)
train(train=False, dataset='autumn', sensor_type=sensors[0], model_type=models[0])
"""
