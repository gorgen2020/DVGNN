# -*- coding: utf-8 -*-
""" Created on Mon May  5 15:14:25 2023
@author: Gorgen @Fuction：
（1）“Dynamic Causal Explanation Based Diffusion-Variational Graph Neural Network for Spatio-temporal Forecasting”；
# """

import os
from time import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

from lib.utils import compute_val_loss, evaluate, predict
from lib.data_preparation import read_and_generate_dataset
from lib.utils import scaled_Laplacian

from model import DVGCN as model



print(torch.__version__)
print(torch.cuda.is_available())

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--max_epoch', type=int, default=40, help='Epoch to run [default: 40]')

parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: Tdrive: 8, los_loop: 16, PEMS08: 16]')

parser.add_argument('--learning_rate', type=float, default=0.0005, help='Initial learning rate [default: 0.0005]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--length', type=int, default=60, help='Size of temporal : 12')
parser.add_argument("--force", type=str, default=False,
                    help="remove params dir", required=False)
parser.add_argument('--decay', type=float, default=0.92, help='decay rate of learning rate ')

parser.add_argument("--data_name", type=str, default='PEMS08',
                    help="The dataset name:Tdirve, Los_loop, PEMS08 ", required=False)

parser.add_argument('--num_point', type=int, default=170,
                        help='road or grid Point Number 64,207,170 ', required=False)

FLAGS = parser.parse_args()
decay = FLAGS.decay
f = FLAGS.data_name

graph_signal_matrix_filename = 'data/%s/%s.npz' % (f, f)
Length = FLAGS.length
batch_size = FLAGS.batch_size
num_nodes = FLAGS.num_point
epochs = FLAGS.max_epoch
learning_rate = FLAGS.learning_rate
optimizer = FLAGS.optimizer

if f == 'Tdrive':
    points_per_hour = 12
    num_of_features = 1
    merge = False
    beijing_adj = pd.read_csv(r'data/Tdrive/Tdrive_adj.csv', header=None)
    adj = np.mat(beijing_adj, dtype=float)
    num_for_predict = 12
    num_of_weeks = 2
    num_of_days = 1
    num_of_hours = 2
if f == 'Los_loop':
    points_per_hour = 6
    num_of_features = 1
    merge = False
    Los_loop_adj = pd.read_csv(r'data/Los_loop/los_adj.csv', header=None)
    adj = np.mat(Los_loop_adj, dtype=float)
    num_for_predict = 12
    num_of_weeks = 2
    num_of_days = 1
    num_of_hours = 2
if f == 'PEMS08':
    points_per_hour = 12
    num_of_features = 3
    merge = False
    PEMS08_adj = pd.read_csv(r'data/PEMS08/pems08_adj.csv', header=None)
    adj = np.mat(PEMS08_adj, dtype=float)
    num_for_predict = 12
    num_of_weeks = 2
    num_of_days = 1
    num_of_hours = 2

num_of_vertices = FLAGS.num_point



model_name = 'DVGCN_%s' % f

wdecay = 0.000


device = torch.device(FLAGS.device)

adjs = scaled_Laplacian(adj)
supports = (torch.tensor(adjs)).type(torch.float32).to(device)



print('Model is %s' % (model_name))

prediction_path = 'DVGCN_prediction_%s' % f


if f == 'Tdrive':
    generated_adj = 'generated_adj/dynamic_Tdrive_adj.npy'
if f == 'Los_loop':
    generated_adj = 'generated_adj/dynamic_Los_loop_adj.npy'
if f == 'PEMS08':
    generated_adj = 'generated_adj/dynamic_PEMS08_adj.npy'

if __name__ == "__main__":
    # read all data from graph signal matrix file
    print("Reading data...")
    # Input: train / valid  / test : length x 3 x NUM_POINT x 12
    all_data = read_and_generate_dataset(graph_signal_matrix_filename, generated_adj, num_of_weeks,  num_of_days, num_of_hours,   num_for_predict, points_per_hour,
                                         merge)
    # test set ground truth
    true_value = all_data['test']['target']
    print(true_value.shape)

    # training set data loader
    train_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['train']['week']),
            torch.Tensor(all_data['train']['day']),
            torch.Tensor(all_data['train']['recent']),
            torch.Tensor(all_data['train']['target']),
            torch.Tensor(all_data['train']['recent_adj']),
        ),
        batch_size=batch_size,
        shuffle=True
    )

    # validation set data loader
    val_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['val']['week']),
            torch.Tensor(all_data['val']['day']),
            torch.Tensor(all_data['val']['recent']),
            torch.Tensor(all_data['val']['target']),
            torch.Tensor(all_data['val']['recent_adj']),
        ),
        batch_size=batch_size,
        shuffle=False
    )

    # testing set data loader
    test_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['test']['week']),
            torch.Tensor(all_data['test']['day']),
            torch.Tensor(all_data['test']['recent']),
            torch.Tensor(all_data['test']['target']),
            torch.Tensor(all_data['test']['recent_adj']),
        ),
        batch_size=batch_size,
        shuffle=False
    )

    # loss function MSE
    loss_function = nn.MSELoss()

    # get model's structure
    net = model(c_in=num_of_features, c_out=64,
                num_nodes=num_nodes, week=24,
                day=12, recent=24,
                K=3, Kt=3)
    net.to(device)  # to cuda

    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=wdecay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay)


    # calculate origin loss in epoch 0

    compute_val_loss(net, val_loader, loss_function, supports, device, epoch=0)

    # compute testing set MAE, RMSE, MAPE before training
    evaluate(net, test_loader, true_value, supports, device, epoch=0)

    clip = 5
    his_loss = []
    train_time = []
    rmse = []
    mae = []
    mape = []
    for epoch in range(1, epochs + 1):
        train_l = []
        start_time_train = time()
        for train_w, train_d, train_r, train_t, train_adj_r in train_loader:
            train_w = train_w.to(device)
            train_d = train_d.to(device)
            train_r = train_r.to(device)
            train_t = train_t.to(device)
            train_adj_r = train_adj_r.to(device)

            net.train()  # train pattern
            optimizer.zero_grad()  # grad to 0


            output, _, A = net(train_w, train_d, train_r, supports,train_adj_r)

            loss = loss_function(output, train_t)
            # backward p
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(net.parameters(), clip)

            # update parameter
            optimizer.step()

            training_loss = loss.item()
            train_l.append(training_loss)
        scheduler.step()
        end_time_train = time()
        train_l = np.mean(train_l)
        print('epoch step: %s, training loss: %.2f, time: %.2fs'
              % (epoch, train_l, end_time_train - start_time_train))
        train_time.append(end_time_train - start_time_train)


        # compute validation loss
        valid_loss = compute_val_loss(net, val_loader, loss_function, supports, device, epoch)
        his_loss.append(valid_loss)


        # evaluate the model on testing set
        rmse1,mae1,mape1 = evaluate(net, test_loader, true_value, supports, device, epoch)

        rmse1 = round(rmse1, 4)
        mae1 = round(mae1, 4)
        mape1 = round(mape1, 4)

        rmse.append(rmse1)
        mae.append(mae1)
        mape.append(mape1)


    print("Training finished")
    print("Training time/epoch: %.2f secs/epoch" % np.mean(train_time))


    start_time_test = time()
    prediction, spatial_at, parameter_adj = predict(net, test_loader, supports, device)

    end_time_test = time()

    evaluate(net, test_loader, true_value, supports, device, epoch)

    test_time = np.mean(end_time_test - start_time_test)
    # print("Test time: %.2f" % test_time)

    print("The min rmse is : " + str(min(rmse)))
    print("The min rmse epoch is : " + str(rmse.index(min(rmse))))

    print("The min mae is : " + str(min(mae)))
    print("The min mae epoch is : " + str(mae.index(min(mae))))

    print("The min mape is : " + str(min(mape)))
    print("The min mape epoch is : " + str(mape.index(min(mape))))
    #绘制折线图
    fig = plt.figure()
    fig = plt.figure(figsize=(15, 8))  # 画柱形图
    ax1 = fig.add_subplot(111)
    yerr = np.linspace(0.05, 0.2, 10)
    x = np.linspace(1, epochs, epochs)
    plt.errorbar(x, rmse, marker='H', markersize=12, yerr=yerr[0], uplims=True, lolims=True,
                 label='rmse')
    plt.errorbar(x, mae, marker='D', markersize=10, yerr=yerr[1], uplims=True, lolims=True,
                 label='mae')

    plt.show()
    np.savez_compressed(
        os.path.normpath(prediction_path),
        prediction=prediction,
        spatial_at=spatial_at,
        parameter_adj=parameter_adj,
        ground_truth=all_data['test']['target']
    )














