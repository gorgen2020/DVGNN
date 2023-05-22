# -*- coding: utf-8 -*-
""" Created on Mon May  5 15:14:25 2023
 @author: Gorgen
 @Fuction：     （1）“Dynamic Causal Explanation Based Diffusion-Variational Graph Neural Network for Spatio-temporal Forecasting”；
"""


import os
import numpy as np
import scipy.sparse as sp
import tensorflow.compat.v1 as tf
import pandas as pd
from model import DVGAE

import argparse
import configparser

# prepare dataset
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/PEMS08.conf', type=str,
                    help="configuration file path")

args = parser.parse_args()
config1 = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
config1.read(args.config)
data_config = config1['Data']
dataset = data_config['dataset']

def main():
    tf.disable_v2_behavior()
    print('gpu is running or not? ' + str(tf.test.is_gpu_available()))

    if dataset == 'PEMS08':
        data_seq = np.load(data_config['graph_signals'])
        features = data_seq['data']
    else:
        features = pd.read_csv(data_config['graph_signals'])
    features = np.array(features)
    n_nodes = features.shape[1]
    features_num = data_config['features_num']
    features_num = int(features_num)
    print("nodes number {0}".format(n_nodes))
    print("features number {0}".format(features_num))



    adj_predefine = pd.read_csv(data_config['adj'], header=None)
    adj_predefine = np.array(adj_predefine)

    adj_predefine = np.array(adj_predefine > 0.5, dtype=float)

    coo_adjacency = sp.coo_matrix(adj_predefine)

    features = features.swapaxes(0, 1)
    max_value = 1.0
    features = features / max_value


    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as tf_sess:
        model = DVGAE(tf_sess, n_nodes, adj_predefine, features_num, config1)
        model.Train(coo_adjacency, features, max_value)


if __name__ == '__main__':
    main()