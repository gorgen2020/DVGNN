# -*- coding:utf-8 -*-
# pylint: disable=no-member

import csv
import numpy as np
from scipy.sparse.linalg import eigs

from .metrics import mean_absolute_error, mean_squared_error, masked_mape_np
import torch
from time import time
import scipy.sparse as sp
import tensorflow as tf


def normalized_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    normalized_adj = normalized_adj.astype(np.float32)
    return normalized_adj


def sparse_to_tuple(mx):
    mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col)).transpose()
    L = tf.SparseTensor(coords, mx.data, mx.shape)
    return tf.sparse_reorder(L)


def calculate_laplacian(adj, lambda_max=1):
    L = normalized_adj(adj + np.eye(adj.shape[0]))
    L=L+L**2+L**3+L**4+L**5

    L = sp.csr_matrix(L)
    L = L.astype(np.float32)
    return L


def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)

    return tf.Variable(initial, name=name)


def search_data(sequence_length, num_of_batches, label_start_idx,
                num_for_predict, units, points_per_hour):
    '''
    Parameters
    ----------
    sequence_length: int, length of all history data

    num_of_batches: int, the number of batches will be used for training

    label_start_idx: int, the first index of predicting target

    num_for_predict: int,
                     the number of points will be predicted for each sample

    units: int, week: 7 * 24, day: 24, recent(hour): 1

    points_per_hour: int, number of points per hour, depends on data

    Returns
    ----------
    list[(start_idx, end_idx)]
    '''

    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    for i in range(1, num_of_batches + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_for_predict
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_batches:
        return None

    return x_idx[::-1]#倒叙输出,符合时间的 顺序输出,这里不占用多少空间


def get_sample_indices(data_sequence,adj, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=1):
    '''
    Parameters
    ----------
    data_sequence: np.ndarray
                   shape is (sequence_length, num_of_vertices, num_of_features)

    num_of_weeks, num_of_days, num_of_hours: int

    label_start_idx: int, the first index of predicting target

    num_for_predict: int,
                     the number of points will be predicted for each sample

    points_per_hour: int, default 12, number of points per hour

    Returns
    ----------
    week_sample: np.ndarray
                 shape is (num_of_weeks * points_per_hour,
                           num_of_vertices, num_of_features)

    day_sample: np.ndarray
                 shape is (num_of_days * points_per_hour,
                           num_of_vertices, num_of_features)

    hour_sample: np.ndarray
                 shape is (num_of_hours * points_per_hour,
                           num_of_vertices, num_of_features)

    target: np.ndarray
            shape is (num_for_predict, num_of_vertices, num_of_features)
    '''

    week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                               label_start_idx, num_for_predict,
                               1, points_per_hour)
    if not week_indices:
        return None

    day_indices = search_data(data_sequence.shape[0], num_of_days,
                              label_start_idx, num_for_predict,
                              1, points_per_hour)
    if not day_indices:
        return None

    hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                               label_start_idx, num_for_predict,
                               1, points_per_hour)
    if not hour_indices:
        return None
    c = [data_sequence[i: j] for i, j in week_indices]
    week_sample = np.concatenate([data_sequence[i: j]
                                  for i, j in week_indices], axis=0)
    day_sample = np.concatenate([data_sequence[i: j]
                                 for i, j in day_indices], axis=0)
    hour_sample = np.concatenate([data_sequence[i: j]
                                  for i, j in hour_indices], axis=0)
    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

    hour_sample_adj = np.concatenate([adj[i: j]
                                  for i, j in hour_indices], axis=0)



    return week_sample, day_sample, hour_sample, target, hour_sample_adj


def get_adjacency_matrix(distance_df_filename, num_of_vertices):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''

    with open(distance_df_filename, 'r') as f:
        reader = csv.reader(f)
        header = f.__next__()
        edges = [(int(i[0]), int(i[1])) for i in reader]

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    for i, j in edges:
        A[i, j] = 1

    return A


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W
    
    lambda_max = eigs(L, k=1, which='LR')[0].real
    
    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list[np.ndarray], length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(
            2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


def compute_val_loss(net, val_loader, loss_function, supports, device, epoch):
    '''
    compute mean loss on validation set

    Parameters
    ----------
    net: model

    val_loader: gluon.data.DataLoader

    loss_function: func

    epoch: int, current epoch

    '''
    net.eval()
    with torch.no_grad():
        tmp = []
        for index, (val_w, val_d, val_r, val_t, val_adj_r) in enumerate(val_loader):
            val_w=val_w.to(device)
            val_d=val_d.to(device)
            val_r=val_r.to(device)
            val_t=val_t.to(device)
            shape = val_t.shape

            val_adj_r = val_adj_r.to(device)
            output,_,_ = net(val_w, val_d, val_r, supports,val_adj_r)
            l = loss_function(output, torch.reshape(val_t,[shape[0], shape[1]]))
            tmp.append(l.item())
    
        validation_loss = sum(tmp) / len(tmp)
    
        print('epoch: %s, validation loss: %.2f' % (epoch, validation_loss))
        return validation_loss


def predict(net, test_loader, supports, device):
    '''
    predict

    Parameters
    ----------
    net: model

    test_loader: gluon.data.DataLoader

    Returns
    ----------
    prediction: np.ndarray,
                shape is (num_of_samples, num_of_vertices, num_for_predict)

    '''
    net.eval()
    with torch.no_grad():
        prediction = []
        for index, (test_w, test_d, test_r, test_t, test_adj_r) in enumerate(test_loader):
            test_w=test_w.to(device)
            test_d=test_d.to(device)
            test_r=test_r.to(device)
            test_t=test_t.to(device)
            test_adj_r = test_adj_r.to(device)
            output,_,_=net(test_w, test_d, test_r, supports,test_adj_r)
            prediction.append(output.cpu().detach().numpy())




        start_adj = time()
        #get first batch's spatial attention matrix    
        for index, (test_w, test_d, test_r, test_t, test_adj_r) in enumerate(test_loader):
            test_w=test_w.to(device)
            test_d=test_d.to(device)
            test_r=test_r.to(device)
            test_t=test_t.to(device)
            test_adj_r = test_adj_r.to(device)
            _,spatial_at,temporal_at=net(test_w, test_d, test_r, supports,test_adj_r)
            spatial_at=spatial_at.cpu().detach().numpy()
            temporal_at=temporal_at.cpu().detach().numpy()
            break
        print(spatial_at.shape)
        calculate_laplacian(spatial_at[0,:,:])

        end_adj= time()
        test_time = end_adj - start_adj
        print("create laplacian time: %.5f" % test_time)

        prediction = np.concatenate(prediction, 0)
        return prediction,spatial_at,temporal_at


def evaluate(net, test_loader, true_value, supports, device, epoch):
    '''
    compute MAE, RMSE, MAPE scores of the prediction
    for 3, 6, 12 points on testing set

    Parameters
    ----------
    net: model

    test_loader: gluon.data.DataLoader

    true_value: np.ndarray, all ground truth of testing set
                shape is (num_of_samples, num_for_predict, num_of_vertices)

    num_of_vertices: int, number of vertices

    epoch: int, current epoch

    '''
    net.eval()
    with torch.no_grad():
        prediction, _, _ = predict(net, test_loader, supports, device)

        # print(prediction.shape)
        # prediction = (prediction.transpose((0, 2, 1))
        #        .reshape(prediction.shape[0], -1))
        for i in [1]:
            print('current epoch: %s, predict %s points' % (epoch, i))

            mae = mean_absolute_error(true_value[:,:, 0],
                                  prediction[:, :])
            rmse = mean_squared_error(true_value[:,:, 0],
                                  prediction[:, :]) ** 0.5
            mape = masked_mape_np(true_value[:,:, 0],
                              prediction[:, :], 0)

            print('MAE: %.2f' % (mae))
            print('RMSE: %.2f' % (rmse))

    return rmse,mae,mape


        
