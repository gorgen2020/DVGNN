# -*- coding: utf-8 -*-
"""
Created on Mon May  5 15:14:25 2023

@author: Gorgen
@Fuction：
    （1）“Dynamic Causal Explanation Based Diffusion-Variational Graph Neural Network for Spatio-temporal Forecasting”；
"""

import numpy as np
import scipy.sparse as sp
import igraph as ig
import os


def ReadGraphFile(filename):
    with open(filename, 'r') as file:
        indexes = list(map(int, file.read().split()))
        n_nodes = max(indexes) + 1
    
    G = ig.Graph()
    G.add_vertices(n_nodes)
    
    for i_edge in range(len(indexes)//2):
        G.add_edge(indexes[2 * i_edge], indexes[2 * i_edge + 1])#添加边

    return G

def GetGraphCluster(filename, idx = -2):
    G = ReadGraphFile(filename)

    G.vs["name"] = list(range(G.vcount()))

    if idx == -2:
        G = G.components().giant()
    if idx == -1:
        G = G.components().giant()
        G = G.community_multilevel().giant()       
    else:       
        com = G.community_multilevel()
        for i in range(com.__len__()) :
            if idx in com.subgraph(i).vs["name"]:
                G = com.subgraph(i)
                break

    ig.plot(G)
    ig.plot(G, target = './data/strong_components_' +  '.pdf')

    edges = G.get_edgelist()

    n_nodes = G.vcount()
    row = []
    col = []
    data = []
    for edge in edges:
        row.extend([edge[0], edge[1]])
        col.extend([edge[1], edge[0]])
        data.extend([1, 1])

    coo_adjacency = sp.coo_matrix((data, (row, col)), shape = (n_nodes, n_nodes))

    return coo_adjacency, edges

def SaveCooAdjacency(filename, coo_adjacency):
    np.savez(filename, data = coo_adjacency.data, row = coo_adjacency.row, col = coo_adjacency.col, shape = coo_adjacency.shape)

def LoadCooAdjacency(filename):
    loader = np.load(filename)
    return sp.coo_matrix((loader['data'], (loader['row'],loader['col'])), shape=loader['shape'])


def GetNormAdjacencyInfo(coo_adjacency):
    coo_adjacency = coo_adjacency + sp.eye(coo_adjacency.shape[0])
    degree = np.array(coo_adjacency.sum(1))
    d_inv = sp.diags(np.power(degree, -0.5).flatten())
    normalized = coo_adjacency.dot(d_inv).transpose().dot(d_inv)
    
    return GetAdjacencyInfo(sp.coo_matrix(normalized))

def GetAdjacencyInfo(coo_adjacency):
    edges = np.vstack((coo_adjacency.row, coo_adjacency.col)).transpose()

    values = coo_adjacency.data

    return edges, values

def SplitTrainTestDataset(coo_adjacency, test_ratio, valid_ratio):
    n_nodes = coo_adjacency.shape[0]


    edges, values = GetAdjacencyInfo(coo_adjacency)

    n_tests = int(np.floor(edges.shape[0] * test_ratio))

    n_valids = int(np.floor(edges.shape[0] * valid_ratio))


    idx_all = list(range(edges.shape[0]))

    np.random.shuffle(idx_all)

    idx_tests = idx_all[:n_tests]

    idx_valids = idx_all[n_tests : (n_tests + n_valids)]

    test_edges = edges[idx_tests]
    valid_edges = edges[idx_valids]

    train_edges = np.delete(edges, idx_tests + idx_valids, axis = 0)
    

    test_edges_neg = []
    valid_edges_neg = []
    
    while (len(test_edges_neg) < len(test_edges)):
        n1 = np.random.randint(0, n_nodes)
        n2 = np.random.randint(0, n_nodes)
        if n1 == n2:
            continue
        if n1 < n2:
            edge_to_add = [n1, n2]
        else:
            edge_to_add = [n2, n1]

        if any((edges[:] == edge_to_add).all(1)):
            continue
        test_edges_neg.append(edge_to_add)
        
    while (len(valid_edges_neg) < len(valid_edges)):
        n1 = np.random.randint(0, n_nodes)
        n2 = np.random.randint(0, n_nodes)
        if n1 == n2:
            continue        
        if n1 < n2:
            edge_to_add = [n1, n2]
        else:
            edge_to_add = [n2, n1]
            
        if any((edges[:] == edge_to_add).all(1)):
            continue
        valid_edges_neg.append(edge_to_add)

    row = []
    col = []
    data = []
    for edge in train_edges:
        row.extend([edge[0], edge[1]])
        col.extend([edge[1], edge[0]])
        data.extend([1, 1])
    train_coo_adjacency = sp.coo_matrix((data, (row,col)), shape=(n_nodes, n_nodes))

    return train_coo_adjacency, test_edges, test_edges_neg, valid_edges, valid_edges_neg