'''
coding:utf-8
@Software:PyCharm
@Time:2024/11/11 14:25
@Author:tianyi.zhu
'''

import torch
import torch.optim
import torch.utils.data
import matplotlib.pyplot
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import pandas as pd
from model import Self_Attention
from tmb_dataset import TmbDataset
from cox import Matrics_Calculation

# hyper-parameters
f_epochs = 150
f_wd = 10e-5

lr_1 = 0.001
lr_2 = 0.0005
lr_3 = 0.001
lr_4 = 0.001

subgroup_num = 4
max_clone = 5
pos_weights = [1.2, 3, 2, 3]
param_gamma = 0.1

train_bag_list = torch.load("train_set.pt")
test_bag_list = torch.load("test_set.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader_kwargs = {'num_workers': 6, 'pin_memory': True} if torch.cuda.is_available() else {}
train_loader = torch.utils.data.DataLoader(TmbDataset(train_bag_list),
                                               batch_size = 1,
                                               shuffle = True,
                                               **loader_kwargs)
test_loader = torch.utils.data.DataLoader(TmbDataset(test_bag_list),
                                               batch_size = 1,
                                               shuffle = False,
                                               **loader_kwargs)

S, L = Matrics_Calculation(train_bag_list)

def Test(data_set_name):
    for i in range(subgroup_num):
        models[i].eval()
    test_loss = torch.zeros(subgroup_num)
    test_loss_all = 0.
    test_error = torch.zeros(subgroup_num)
    test_error_all = 0.
    batch_count = torch.zeros(subgroup_num)
    label_prediction_list = []
    data_set = test_loader
    if data_set_name == "train":
        data_set = train_loader
    A_matrix = np.zeros((len(data_set), 2 * max_clone + 2), dtype=float)
    study_list = []
    PFS_list = []
    Status_list = []
    with torch.no_grad():
        for batch_idx, (data, response) in enumerate(data_set):
            study_id = response[1].item()
            study_list.append(study_id)
            study_index = study_id - 1
            label = response[0]
            PFS = response[2].item()
            Status = response[3].item()
            PFS_list.append(PFS)
            Status_list.append(Status)
            data = torch.tensor(data, dtype=torch.float)
            label = torch.tensor(label, dtype=torch.float)
            data, label = data.to(device), label.to(device)
            loss, predicted_prob, error, predicted_label, A = models[study_index].calculate(data, label)
            test_loss[study_index] = test_loss[study_index] + loss.data[0]
            test_loss_all = test_loss_all + loss.data[0]
            test_error[study_index] = test_error[study_index] + error
            test_error_all = test_error_all + error
            batch_count[study_index] = batch_count[study_index] + 1
            label_prediction_list.append([label.item(), predicted_prob.item(), predicted_label.item(), study_id])
            ccf_column = data[0][:, 1]
            _, ccf_index = torch.sort(ccf_column, descending=True)
            for i in range(A.shape[1]):
                A_matrix[batch_idx][i] = np.round(A[0][i].detach().numpy(), 4)
                A_matrix[batch_idx][max_clone + i] = ccf_index[i] + 1
            A_matrix[batch_idx][2 * max_clone] = label
            A_matrix[batch_idx][2 * max_clone + 1] = study_id

    true_labels = [true_label for true_label, _, _, _ in label_prediction_list]
    predicted_probabilities = [predicted_prob for _, predicted_prob, _, _ in label_prediction_list]
    auc = roc_auc_score(true_labels, predicted_probabilities)
    if epoch == f_epochs-1 and data_set_name == "test":
        test_loss_all = test_loss_all / len(data_set)
        test_error_all = test_error_all / len(data_set)
        print(f'Test Loss: {test_loss_all.item():.4f}, Test error: {test_error_all:.4f}')
        correct_predictions = sum(1 for true_label, _, predicted_label, _ in label_prediction_list if true_label == predicted_label)
        total_samples = len(label_prediction_list)
        accuracy = correct_predictions / total_samples
        predicted_labels = [predicted_label for _, _, predicted_label, _ in label_prediction_list]
        precision = precision_score(true_labels, predicted_labels, zero_division=0.0)
        recall = recall_score(true_labels, predicted_labels, zero_division=0.0)
        f1 = f1_score(true_labels, predicted_labels, zero_division=0.0)
        print('Accuracy: {:.4f}'.format(accuracy))
        print('Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}'.format(precision, recall, f1))
        print('AUC: {:.4f}'.format(auc))
        np.savetxt("A_matrix.txt", A_matrix, fmt='%.4f')
        fpr, tpr, thread = roc_curve(true_labels, predicted_probabilities)
        roc_data = pd.DataFrame({
            'Y': true_labels,
            'Sample score': predicted_probabilities,
            'Group': predicted_labels,
            'PFS': PFS_list,
            'Status': Status_list,
            'Subgroup ID': study_list
        })
        roc_data.to_csv('Output.csv', index=False)
    return [test_loss.cpu().numpy(), test_error, auc]

def Train():
    for i in range(subgroup_num):
        models[i].train()
    for batch_idx, (data, response) in enumerate(train_loader):
        study_id = response[1].item()
        study_index = study_id - 1
        label = response[0]
        data = torch.tensor(data, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)
        loss = []
        error = []
        theta_X = []
        for i in range(subgroup_num):
            optimizers[i].zero_grad()
            loss_i, predicted_prob_i, error_i, _, _ = models[i].calculate(data, label, pos_weights[i])
            loss.append(loss_i)
            error.append(error_i)
            theta_X.append(predicted_prob_i)

        loss_all = (loss[0] * S[study_index][0] + \
                    loss[1] * S[study_index][1] + \
                    loss[2] * S[study_index][2] + \
                    loss[3] * S[study_index][3] + \
                    (param_gamma) * theta_X[0]**2 * L[0][0] +\
                                    theta_X[1]**2 * L[1][1] +\
                                    theta_X[2]**2 * L[2][2] +\
                                    theta_X[3]**2 * L[3][3])
        loss_all.backward()
        for i in range(subgroup_num):
            optimizers[i].step()
    return 0

if __name__ == "__main__":
    models = {}
    optimizers = {}
    learning_rates = [lr_1, lr_2, lr_3, lr_4]
    for i, lr in enumerate(learning_rates):
        torch.manual_seed(42)
        model = Self_Attention().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=f_wd)
        models[i] = model
        optimizers[i] = optimizer

    train_auc = []
    test_auc = []
    epoch_num = 0
    for epoch in range(f_epochs):
        epoch_num += 1
        print(f'Epoch: {epoch_num}--------------------------------------------------------------------------------------')
        Train()
        train_row = Test("train")
        train_auc.append(train_row[2])
        test_row = Test("test")
        test_auc.append(test_row[2])
        if train_row[2] > 0.80 and epoch < f_epochs - 1:
            epoch = f_epochs - 1
            Test("train")
            Test("test")
            break
    matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(range(1, epoch_num + 1), train_auc, label='Train AUC')
    matplotlib.pyplot.plot(range(1, epoch_num + 1), test_auc, label='Test AUC')
    matplotlib.pyplot.xlabel('Epoch')
    matplotlib.pyplot.ylabel('AUC')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()
