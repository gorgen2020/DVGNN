import numpy as np
import time


dataset_name = 'FMRI-3'

weigh_fi = np.load(dataset_name + '_normalization_parameter.npz') ['Weight_fi']

time_len = weigh_fi.shape[0]
nodes = weigh_fi.shape[1]
dynamic_matrix = np.zeros([time_len, nodes, nodes])


sigma_cov = np.load(dataset_name + '_normalization_parameter.npz') ['Sigma_Weight_fi_out']
mu1 =np.load(dataset_name + '_normalization_parameter.npz') ['mu_output1']

mu1 = mu1[:,:,0]
sigma1 = np.load(dataset_name + '_normalization_parameter.npz') ['sigma_output1']
sigma1 = np.sum(sigma1,axis=2)
mu2 = np.load(dataset_name + '_normalization_parameter.npz') ['mu_output2']


mu2 = mu2[:,:,0]
sigma2 = np.load(dataset_name + '_normalization_parameter.npz') ['sigma_output2']
sigma2 = np.sum(sigma2,axis=2)

bb =[]


for i in range(time_len):
    z1 = mu1[i,:]
    z2 = mu2[i,:]
    mean = np.matmul(weigh_fi[i,:],z1)
    sigma_cov = np.abs(np.array(sigma_cov))
    for j in range(nodes):
        for k in range(nodes):
            if sigma_cov[i,j,k] != 0:
                pp1 =  1.0 / np.power(2 * np.pi, 1.0  / 2) / np.sqrt(np.abs(sigma_cov[i,j,k])) * np.exp(
                    -1.0 * (z2[k] - mean[j]) ** 2 / (2 * sigma_cov[i, j, k] ** 2))  # 概率密度
                dynamic_matrix[i,j,k] = round(pp1,2)

    print(i)
    if i==1:
        a = time.time()
    if i ==2:
        b = time.time()
        print(b-a)

    amax = np.sum(dynamic_matrix[i,:, :], axis=1)
    for m in range(dynamic_matrix.shape[1]):
        if amax[m] != 0:
            dynamic_matrix[i,m, :] = dynamic_matrix[i,m, :] / amax[m]

    calc_number = np.array(dynamic_matrix[i,:, :] !=0,dtype=int)
    num = np.sum(calc_number)
    print("unprocess element :" + str(num))


np.save( "dynamic_"+dataset_name+"_adj.npy", dynamic_matrix)

