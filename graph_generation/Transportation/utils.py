# -*- coding: utf-8 -*-
"""
Created on Mon May  5 15:14:25 2023

@author: Gorgen
@Fuction：
    （1）“Dynamic Causal Explanation Based Diffusion-Variational Graph Neural Network for Spatio-temporal Forecasting”；
"""

import tensorflow.compat.v1 as tf
import numpy as np




tf.disable_v2_behavior()

def UniformRandomWeights(shape, name = None):
    randoms = tf.random_uniform(shape, minval = -np.sqrt(6.0 / (shape[0] + shape[1])), maxval = np.sqrt(6.0 / (shape[0] + shape[1])), dtype=tf.float32)
    return tf.Variable(randoms, name = name)#返回Tensorflow变量




def GaussianSampleWithNP(mean1, diag_cov1 ,mean2, diag_cov2 ):
    e1 = mean1 + np.random.normal(size = diag_cov1.shape)
    e2 = mean2 + np.random.normal(size = diag_cov2.shape)
    return e1,e2


def FirstGCNLayerWithActiveFun_NoX(norm_adj_mat, W, b,x):
    return tf.nn.relu(tf.add(tf.sparse_tensor_dense_matmul(norm_adj_mat, tf.matmul(x,W)), b))


def SecondGCNLayerWithoutActiveFun(norm_adj_mat, h, W, b):
    return  tf.nn.sigmoid(tf.add(tf.matmul(tf.sparse_tensor_dense_matmul(norm_adj_mat, h), W), b))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def preprocess_data(data, time_len, rate):
    train_size = int(time_len * rate)
    train_data = data[:, 0:train_size]
    test_data = data[:, train_size:time_len]


    return train_data, test_data