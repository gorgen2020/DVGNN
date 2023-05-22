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
parser.add_argument("--config", default='configurations/FMRI-13.conf', type=str,
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

    features = pd.read_csv(data_config['graph_signals'])
    features = np.array(features)
    # features = features[0:features.shape[0], :]
    n_nodes = features.shape[1]
    features_num = data_config['features_num']
    features_num = int(features_num)
    print("nodes number {0}".format(n_nodes))
    print("features number {0}".format(features_num))


    adj_predefine = np.eye(n_nodes, dtype=float)
    groundTruth_adj = np.zeros([n_nodes, n_nodes])
    with open(data_config['adj'], encoding='utf-8-sig') as FMRI:
        node_num = len(FMRI.readlines())  # print(node_num)
    with open(data_config['adj'], encoding='utf-8-sig') as FMRI:
        for j in range(node_num):
            line = FMRI.readline()
            line = line.strip()
            line = line.split(',')
            groundTruth_adj[int(line[0]), int(line[1])] = float(line[2])

    coo_adjacency = sp.coo_matrix(adj_predefine)

    features = features.swapaxes(0, 1)
    max_value = 1.0
    features = features / max_value


    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as tf_sess:
        model = DVGAE(tf_sess, n_nodes, groundTruth_adj, features_num, config1)
        model.Train(coo_adjacency, features, groundTruth_adj, max_value)


if __name__ == '__main__':
    main()