'''
coding:utf-8
@Software:PyCharm
@Time:2024/8/14 11:57
@Author:tianyi.zhu
'''

from torch.utils.data import Dataset
import torch
import numpy as np

class TmbDataset(Dataset):
    def __init__(self, bag_list):
        self.data = bag_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        study_id = int(sample[0])
        PFS = float(sample[2])
        Status = int(sample[3])
        lable = float(sample[4])
        bag_data = sample[5]
        data = []
        response = []
        response.append(lable)
        response.append(study_id)
        response.append(PFS)
        response.append(Status)

        for instance in bag_data:
            row = [instance[0],
                   instance[1],
                   instance[2]]
            data.append(row)
        data = torch.tensor(data, dtype=torch.float)

        # Disrupting the order of the instances to reduce dependence on sequence
        k = data.size(0)
        random_indices = torch.randperm(k)
        data_shuffled = data[random_indices]

        return data_shuffled, response

class TmbDataset_Cox(Dataset):
    def __init__(self, bag_list):
        self.data = bag_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        study_id = int(sample[0])
        PFS = float(sample[2])
        Status = int(sample[3])
        lable = float(sample[4])
        bag_data = sample[5]
        data = []
        response = []
        response.append(lable)
        response.append(study_id)
        response.append(PFS)
        response.append(Status)
        for instance in bag_data:
            row = [instance[0],
                   instance[1],
                   instance[2]]
            data.append(row)
        data = torch.tensor(data, dtype=torch.float)
        return data, response