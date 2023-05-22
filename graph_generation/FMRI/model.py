# -*- coding: utf-8 -*-
"""
Created on Mon May  5 15:14:25 2023

@author: Gorgen
@Fuction：
    （1）“Dynamic Causal Explanation Based Diffusion-Variational Graph Neural Network for Spatio-temporal Forecasting”；
"""
import math
import os
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve

import time
import GraphReader
import utils
import argparse
import configparser
import scipy.sparse as sp


class DVGAE(object):
    def __init__(self, tf_sess, n_nodes,adj,features_num,config):
        self.tf_sess = tf_sess
        self.config = config
        data_config = self.config['Data']
        training_config = self.config['Training']
        self.data_name = data_config['dataset']

        if training_config['mask'] == 'True':
            self.set_mask = True
        else:
            self.set_mask = False

        if training_config['save_result'] == 'True':
            self.save_result = True
        else:
            self.save_result = False

        self.features_num = features_num

        self.adj = adj

        self.n_nodes = n_nodes
        self.epochs = int(training_config['epochs'])

        self.pre_len = int(training_config['pre_len'])
        self.seq_len = int(training_config['seq_len'])
        self.train_rate = float(training_config['train_rate'])


        self.method = training_config['method']

        self.n_hiddens = int(training_config['n_hiddens'])

        self.n_embeddings = int(training_config['n_embeddings'])
        self.test_ratio = float(training_config['edges_test_ratio'])

        self.valid_ratio = float(training_config['valid_ratio'])

        if training_config['dropout'] == 'True':
            self.dropout = True
        else:
            self.dropout = False

        self.learning_rate1 = float(training_config['learning_rate1'])
        self.keep_prob = float(training_config['keep_prob'])
        self.shape = np.array([self.n_nodes, self.n_nodes])

        self.lamada = float(training_config['lamada'])

        self.tf_sparse_adjacency = tf.sparse_placeholder(tf.float32, shape=self.shape, name='tf_sparse_adjacency')

        self.tf_norm_sparse_adjacency = tf.sparse_placeholder(tf.float32, shape=self.shape, name='tf_norm_sparse_adjacency')
        self.sigmoid = np.vectorize(utils.sigmoid)


        print('node number is {0}. sequence length is {1}'.format(self.n_nodes, self.seq_len) )

        self.inputs1 = tf.placeholder(tf.float32,shape=[self.n_nodes, self.seq_len * self.features_num])

        self.inputs2 = tf.placeholder(tf.float32,shape=[self.n_nodes, self.seq_len * self.features_num])


        self.adjacence = tf.placeholder(tf.float32,shape=[self.n_nodes, self.n_nodes])

        print('method is ' + self.method)

        self.__BuildVGAE()
        
    def __BuildVGAE(self):
        self.TFNode_VEncoder()

        self.tfnode_raw_adjacency_pred = self.TFNode_VDecoder()



        self.tfnode_latent_loss1 = -(0.5 / self.n_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * tf.log(self.tfnode_sigma1) - tf.square(self.tfnode_mu1) - tf.square(self.tfnode_sigma1), 1))
        self.tfnode_latent_loss2 = -(0.5 / self.n_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * tf.log(self.tfnode_sigma2) - tf.square(self.tfnode_mu2) - tf.square(self.tfnode_sigma2), 1))
        self.tfnode_latent_loss = self.tfnode_latent_loss1 + self.tfnode_latent_loss2


        tf_dense_adjacency = tf.reshape(tf.sparse_tensor_to_dense(self.tf_sparse_adjacency, validate_indices=False),
                                        self.shape)


        tfnode_w1 = (self.n_nodes * self.n_nodes - tf.reduce_sum(tf_dense_adjacency)) / tf.reduce_sum(
            tf_dense_adjacency)

        tfnode_w2 = self.n_nodes * self.n_nodes / (self.n_nodes * self.n_nodes - tf.reduce_sum(tf_dense_adjacency))


        self.tfnode_reconst_loss = tfnode_w2 * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(targets=tf_dense_adjacency, logits=self.tfnode_raw_adjacency_pred,
                                                     pos_weight=tfnode_w1))

        self.tfnode_all_loss = self.tfnode_reconst_loss + self.tfnode_latent_loss

        self.tf_optimizer1 = tf.train.GradientDescentOptimizer(self.learning_rate1)
        self.tf_optimizer_minimize1 = self.tf_optimizer1.minimize(self.tfnode_all_loss)

        tf_init = tf.global_variables_initializer()
        self.tf_sess.run(tf_init)

    def TFNode_VEncoder(self):
        self.W0 = utils.UniformRandomWeights(shape=[self.seq_len * self.features_num, self.n_hiddens])


        self.mu_B0 = tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[self.n_hiddens]))
        self.mu_W1 = utils.UniformRandomWeights(shape=[self.n_hiddens, self.n_embeddings])
        self.mu_B1 = tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[self.n_embeddings]))


        self.sigma_B0 = tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[self.n_hiddens]))
        self.sigma_W1 = utils.UniformRandomWeights(shape=[self.n_hiddens, self.n_embeddings])
        self.sigma_B1 = tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[self.n_embeddings]))


        tfnode_mu_hidden01 = utils.FirstGCNLayerWithActiveFun_NoX(self.tf_norm_sparse_adjacency, self.W0, self.mu_B0,
                                                                  self.inputs1)
        if self.dropout:
            tfnode_mu_hidden0_dropout1 = tf.nn.dropout(tfnode_mu_hidden01, self.keep_prob)
        else:
            tfnode_mu_hidden0_dropout1 = tfnode_mu_hidden01


        self.tfnode_mu1 = utils.SecondGCNLayerWithoutActiveFun(self.tf_norm_sparse_adjacency,
                                                               tfnode_mu_hidden0_dropout1,
                                                               self.mu_W1, self.mu_B1)


        tfnode_sigma_hidden01 = utils.FirstGCNLayerWithActiveFun_NoX(self.tf_norm_sparse_adjacency, self.W0,
                                                                     self.sigma_B0,
                                                                     self.inputs1)
        if self.dropout:
            tfnode_sigma_hidden0_dropout1 = tf.nn.dropout(tfnode_sigma_hidden01, self.keep_prob)
        else:
            tfnode_sigma_hidden0_dropout1 = tfnode_sigma_hidden01


        tfnode_log_sigma1 = utils.SecondGCNLayerWithoutActiveFun(self.tf_norm_sparse_adjacency,
                                                                 tfnode_sigma_hidden0_dropout1, self.sigma_W1,
                                                                 self.sigma_B1)
        self.tfnode_sigma1 = tf.exp(tfnode_log_sigma1)


        tfnode_mu_hidden02 = utils.FirstGCNLayerWithActiveFun_NoX(self.tf_norm_sparse_adjacency, self.W0, self.mu_B0,
                                                                  self.inputs2)
        if self.dropout:
            tfnode_mu_hidden0_dropout2 = tf.nn.dropout(tfnode_mu_hidden02, self.keep_prob)
        else:
            tfnode_mu_hidden0_dropout2 = tfnode_mu_hidden02


        self.tfnode_mu2 = utils.SecondGCNLayerWithoutActiveFun(self.tf_norm_sparse_adjacency,
                                                               tfnode_mu_hidden0_dropout2,
                                                               self.mu_W1, self.mu_B1)


        tfnode_sigma_hidden02 = utils.FirstGCNLayerWithActiveFun_NoX(self.tf_norm_sparse_adjacency, self.W0,
                                                                     self.sigma_B0,
                                                                     self.inputs2)
        if self.dropout:
            tfnode_sigma_hidden0_dropout2 = tf.nn.dropout(tfnode_sigma_hidden02, self.keep_prob)
        else:
            tfnode_sigma_hidden0_dropout2 = tfnode_sigma_hidden02


        tfnode_log_sigma2 = utils.SecondGCNLayerWithoutActiveFun(self.tf_norm_sparse_adjacency,
                                                                 tfnode_sigma_hidden0_dropout2, self.sigma_W1,
                                                                 self.sigma_B1)
        self.tfnode_sigma2 = tf.exp(tfnode_log_sigma2)

    def TFNode_VDecoder(self):
        # 隐变量采样（均值与标准方差）
        self.Weight_fi = utils.UniformRandomWeights(shape=[self.n_nodes, self.n_nodes])

        self.Weight_fi_transpose = tf.transpose(self.Weight_fi, [1, 0])
        self.Sigma_Weight_fi = 2 * tf.matmul(self.Weight_fi, self.Weight_fi_transpose)

        self.tfnode_sigma2_transpose = tf.transpose(self.tfnode_sigma2, [1, 0])
        self.Sigma_Embedding = tf.matmul(self.tfnode_sigma1, self.tfnode_sigma2_transpose)


        self.index = -1 * tf.abs(self.lamada * tf.log(self.Sigma_Embedding + 0.0001) - tf.log(2 * math.pi * tf.abs(self.Sigma_Embedding**2 - self.Sigma_Weight_fi) + 0.0001) + tf.divide(self.Sigma_Embedding**2, (self.Sigma_Embedding**2 - self.Sigma_Weight_fi) + 0.0001))
        # self.index = tf.clip_by_value(self.index, -100, 100)

        self.index = tf.exp(self.index)


        self.adjacency_pred = self.index

        return self.adjacency_pred


    def Train(self, coo_adjacency, features, groundTruth_adj, max_value):
        saver = tf.train.Saver(tf.global_variables())


        gt_adj = sp.coo_matrix(groundTruth_adj)
        _, test_edges, test_edges_neg, valid_edges, valid_edges_neg = GraphReader.SplitTrainTestDataset(
            gt_adj, self.test_ratio, self.valid_ratio)

        edges, values = GraphReader.GetAdjacencyInfo(coo_adjacency)
        norm_edges, norm_values = GraphReader.GetNormAdjacencyInfo(coo_adjacency)


        self.time_len = features.shape[1]
        trainX, testX = utils.preprocess_data(features, self.time_len, 1.0)
        print('total dataset length : ' + str(self.time_len))
        print('train dataset length : ' + str(int(self.train_rate * trainX.shape[1])))
        print('test dataset length : ' + str(testX.shape[1]))

        fig_train_loss1, VGAE_train_latent_loss, VGAE_train_reconst_loss = [], [], []

        fig_train_loss2, GCN_train_loss, GCN_train_error = [], [], []


        total_probability = []
        total_precision = []
        total_AUC = []

        temp_probability = []

        mmu_output1 =[]
        ssigma_output1=[]
        mmu_output2 =[]
        ssigma_output2=[]
        SSigma_Weight_fi_out=[]

        WWeight_fi=[]

        for i in range(self.epochs):
            time_start = time.time()
            for m in range(trainX.shape[1]-1):
                if self.method == 'dynamic':
                    feed_dict = {self.tf_sparse_adjacency: (edges, values),
                                     self.tf_norm_sparse_adjacency: (norm_edges, norm_values), self.inputs1: np.reshape(trainX[:, m:m+self.seq_len], [self.n_nodes, self.seq_len * self.features_num]),
                                     self.inputs2: np.reshape(trainX[:, m + 1:m + 1 + self.seq_len],[self.n_nodes, self.seq_len * self.features_num])}

                    minimizer1, latent_loss, reconst_loss, self.mu_output1, self.sigma_output1, self.mu_output2, self.sigma_output2, self.Sigma_Weight_fi_out, self.index11, self.Weigh_out = self.tf_sess.run(
                        [self.tf_optimizer_minimize1, self.tfnode_all_loss, self.tfnode_reconst_loss, self.tfnode_mu1,
                         self.tfnode_sigma1, self.tfnode_mu2, self.tfnode_sigma2, self.Sigma_Weight_fi, self.tfnode_raw_adjacency_pred, self.Weight_fi], feed_dict=feed_dict)

                    self.adj_generated = self.index11


                    temp_probability.append(self.adj_generated)


                    VGAE_train_latent_loss.append(latent_loss)
                    VGAE_train_reconst_loss.append(reconst_loss)

            if self.method == 'dynamic':
                print("At step {0} VGAE_reconst_loss: {1}  VGAE_train_latent_loss : {2} .".format(i,np.sum(VGAE_train_reconst_loss), np.sum(VGAE_train_latent_loss)))

            if self.method == 'dynamic':
                aa = np.array(temp_probability)
                auc, ap, precision, recall = self.CalcAUC_AP(test_edges, test_edges_neg, aa)


                print("At step {0}  auc Loss: {1} ROC Average Accuracy: {2}. Precision:{3} recall: {4} F1-score: {5} ".format(i, auc, ap, precision, recall, 2 * precision * recall / (precision + recall) ))
                total_precision.append(precision)
                total_AUC.append(auc)


            time_end = time.time()
            print(time_end - time_start, 's')

            fig_train_loss1.append(np.sum(VGAE_train_latent_loss))
            fig_train_loss2.append(np.sum(GCN_train_loss))

            VGAE_train_latent_loss, VGAE_train_reconst_loss = [], []
            # 初始化第二阶段的损失函数和输出结果
            GCN_train_loss, GCN_train_error = [], []

        for m in range(trainX.shape[1] -1):
            if self.method == 'dynamic':
                feed_dict = {self.tf_sparse_adjacency: (edges, values),
                             self.tf_norm_sparse_adjacency: (norm_edges, norm_values), self.inputs1: np.reshape(trainX[:, m:m+self.seq_len], [self.n_nodes, self.seq_len * self.features_num]),self.inputs2: np.reshape(trainX[:, m + 1:m + 1 + self.seq_len],[self.n_nodes, self.seq_len * self.features_num])}
                latent_loss, reconst_loss, self.mu_output1, self.sigma_output1, self.mu_output2, self.sigma_output2, self.Sigma_Weight_fi_out, self.index11, self.Weigh_out = self.tf_sess.run( [ self.tfnode_all_loss, self.tfnode_reconst_loss, self.tfnode_mu1, self.tfnode_sigma1, self.tfnode_mu2, self.tfnode_sigma2, self.Sigma_Weight_fi, self.tfnode_raw_adjacency_pred, self.Weight_fi], feed_dict=feed_dict)
                VGAE_train_latent_loss.append(latent_loss)
                VGAE_train_reconst_loss.append(reconst_loss)



                mmu_output1.append(self.mu_output1)
                ssigma_output1.append(self.sigma_output1)
                mmu_output2.append(self.mu_output2)
                ssigma_output2.append(self.sigma_output2)
                SSigma_Weight_fi_out.append(self.Sigma_Weight_fi_out)
                WWeight_fi.append(self.Weigh_out)

                total_probability.append(self.index11)


        if self.save_result:
            mmu_output1 = np.array(mmu_output1)
            ssigma_output1 = np.array(ssigma_output1)
            mmu_output2=  np.array(mmu_output2)
            ssigma_output2 = np.array(ssigma_output2)
            SSigma_Weight_fi_out=  np.array(SSigma_Weight_fi_out)
            WWeight_fi = np.array(WWeight_fi)
            np.savez_compressed(self.data_name+'_normalization_parameter.npz', mu_output1=mmu_output1,sigma_output1=ssigma_output1,
                                mu_output2=mmu_output2,sigma_output2=ssigma_output2,Sigma_Weight_fi_out=SSigma_Weight_fi_out,Weight_fi=WWeight_fi)

        total_precision = np.array(total_precision)
        fig1 = plt.figure()
        plt.plot(total_precision)
        plt.xlabel('iteration')
        plt.ylabel('precision')
        plt.title('total_precision ')
        plt.show()

        total_AUC = np.array(total_AUC)
        fig2 = plt.figure()
        plt.plot(total_AUC)
        plt.xlabel('iteration')
        plt.ylabel('total_AUC')
        plt.title('total_AUC ')
        plt.show()


        fig3 = plt.figure()
        plt.plot(fig_train_loss1)
        plt.xlabel('iteration')
        plt.ylabel('VGAE_train_latent_loss')
        plt.title('VGAE_train_latent_loss ')
        plt.show()


    def CalcAUC_AP(self, pos_edges, neg_edges, adjacent):
        adjacency_pred = adjacent
        y_scores = []
        for i in range(adjacent.shape[0]):
            for edge in pos_edges:
                y_scores.append(adjacency_pred[i, edge[0], edge[1]])
        for j in range(adjacent.shape[0]):
            for edge in neg_edges:
              y_scores.append(adjacency_pred[j, edge[0], edge[1]])

        y_trues = np.hstack([np.ones(adjacent.shape[0] * len(pos_edges)), np.zeros(adjacent.shape[0] * len(neg_edges))])
        yy = np.array(y_scores)
        auc_score = roc_auc_score(y_trues,  yy)
        ap_score = average_precision_score(y_trues, yy)
        precision, recall, _ = precision_recall_curve(y_trues, yy)
        size = precision.shape[0]
        precision = np.sum(precision) / size
        recall = np.sum(recall) / size
        return auc_score, ap_score, precision, recall
