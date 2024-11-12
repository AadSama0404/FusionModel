'''
coding:utf-8
@Software:PyCharm
@Time:2024/11/11 14:43
@Author:tianyi.zhu
'''

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# Adjustment of parameters according to the training set
subgroup_num = 4
max_clone = 5
max_datanum = 74

def Matrics_Calculation(train_bag_list):
    X_dict = {str(i + 1): [] for i in range(subgroup_num)}

    for row in train_bag_list:
        data = torch.tensor(row[5], dtype=torch.float).squeeze(0)
        if row[0] in X_dict:
            X_dict[row[0]].append(data)
        else:
            print("error!")
    datasets = list(X_dict.values())
    S = np.zeros((subgroup_num, subgroup_num))
    L = np.zeros((subgroup_num, subgroup_num))

    for i in range(subgroup_num):
        for j in range(subgroup_num):
            avg_similarity = Average_Cosine_Similarity_Calculation(datasets[i], datasets[j])
            S[i][j] = avg_similarity
            if i != j:
                L[i][j] = avg_similarity

    for i in range(subgroup_num):
        L[i][i] = np.sum(L[i])

    return S, L

def Average_Cosine_Similarity_Calculation(data1, data2):
    padded1 = Pad_to_Fixed_Length(data1)
    padded2 = Pad_to_Fixed_Length(data2)

    # cosine similarity
    similarity = cosine_similarity(padded1.reshape(1,-1), padded2.reshape(1,-1))
    return similarity

def Pad_to_Fixed_Length(data, target_length = max_clone * 3, target_count = max_datanum):
    padded_data = []
    for sample in data:
        flattened = sample.flatten()
        padded_sample = np.zeros(target_length)
        padded_sample[:min(target_length, flattened.shape[0])] = flattened[:target_length]
        padded_data.append(padded_sample)
    while len(padded_data) < target_count:
        padded_data.append(np.zeros(target_length))
    return np.array(padded_data).flatten()
