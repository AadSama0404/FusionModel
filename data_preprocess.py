'''
coding:utf-8
@Software:PyCharm
@Time:2024/9/9 11:02
@Author:tianyi.zhu
'''

import pandas as pd
import torch
import numpy as np
from torch.utils.data import random_split
from imblearn.over_sampling import RandomOverSampler

torch.manual_seed(42)
train_ratio = 0.6
subgroup_num = 4

def Oversampling(train_set):
    id_list = []
    lable_list = []
    for row in train_set:
        id_list.append([row[0], row[1], row[2], row[3]])
        lable_list.append(row[4])
    ros = RandomOverSampler(random_state=42)
    id_list_resampled, lable_list_resampled = ros.fit_resample(id_list, lable_list)
    set_resampled = [(id_list_resampled[i][0], \
                      id_list_resampled[i][1], \
                      id_list_resampled[i][2], \
                      id_list_resampled[i][3], \
                      lable_list_resampled[i]) for i in range(len(lable_list_resampled))]
    count_1 = lable_list_resampled.count(1)
    count_0 = lable_list_resampled.count(0)
    return set_resampled

def Dataset_Separate(patient_list):
    train_size = int(train_ratio * len(patient_list))
    test_size = len(patient_list) - train_size
    train_indices, test_indices = random_split(patient_list, [train_size, test_size])
    train_set = [patient_list[idx] for idx in train_indices.indices]
    test_set = [patient_list[idx] for idx in test_indices.indices]

    train_set_resampled = Oversampling(train_set)

    return train_set_resampled, test_set

def Clinical_Data_Preprocess(clinical_data):
    clinical_data_groups = {i: [] for i in range(1, subgroup_num + 1)}

    for row in clinical_data:
        # Process response column
        if row[4] in {'0', '1'}:
            row[4] = int(row[4])
        else:
            print("Error in response column!")
            continue

        # Process and assign rows by PATIENT_ID
        try:
            patient_id = int(row[0])
            if 1 <= patient_id <= subgroup_num:
                clinical_data_groups[patient_id].append(row)
            else:
                print("Error in PATIENT_ID!")
        except ValueError:
            print("Invalid PATIENT_ID format!")

    # Separate into train and test sets for each group and combine results
    train_set_resampled = []
    test_set = []

    for group_data in clinical_data_groups.values():
        train_set, test_set_group = Dataset_Separate(group_data)
        train_set_resampled.extend(train_set)
        test_set.extend(test_set_group)

    return train_set_resampled, test_set

def Genomic_Data_Preprocess(genomic_data):
    instances = {}
    for row in genomic_data:
        # [Study ID, PATIENT_ID, Clone ID]
        key = (row[0], row[2], row[5])
        if key not in instances:
            instances[key] = []
        instances[key].append(row[0:5])
    instance_list = []
    for key, values in instances.items():
        item = values[0][4]
        if all(row[4] == item for row in values):
            instance_list.append(list(key) + [values])
        else:
            print("error")
    return instance_list

def Match(patient_list, instance_list):
    raw_data = []
    for row in patient_list:
        row = list(row)
        patient = row[1]
        bag_data = []
        for instance in instance_list:
            if instance[1] == patient:
                bag_data.append([instance[3]])
        if len(bag_data) >= 1:
            row.append(bag_data)
            raw_data.append(row)
        else:
            print(f"No instance found in this bag. PatientID: {patient}")
    return raw_data

def Scaling_Calculations(raw_data_train):
    lists = [[] for _ in range(subgroup_num)]
    for row in raw_data_train:
        study = int(row[0])  # Convert to integer for direct indexing
        if 1 <= study <= subgroup_num:
            for instance in row[5]:
                mutation_num = len(instance[0])
                lists[study - 1].append(mutation_num)
    divisors = []
    targets_above_20 = [50, 100, 150, 200, 300, 500]

    for i, lst in enumerate(lists, start=1):
        if len(lst) > 0:
            q1 = np.percentile(lst, 25)
            q3 = np.percentile(lst, 75)
            iqr = q3 - q1
            threshold = q3 + 1.5 * iqr

            if threshold < 20:
                rounded_threshold = round(threshold / 5) * 5
            else:
                rounded_threshold = min(targets_above_20, key=lambda x: abs(x - threshold))

            divisors.append(rounded_threshold)
        else:
            print("error!")
            divisors.append(np.nan)
    return divisors

def Mutation_to_Feature(raw_data, divisors):
    for row in raw_data:
        study = int(row[0])  # Convert to integer for indexing
        patient = row[1]
        instance_list = row[5]

        if 1 <= study <= len(divisors):
            divisor = divisors[study - 1]
            for instance in instance_list:
                mutation_num = len(instance[0])
                instance.insert(0, mutation_num / divisor)  # Feature scaling

                mutation_list = instance[1]
                allele_frequency_sum = 0
                mutation_count = 0
                CCF = None

                for mutation in mutation_list:
                    CCF = mutation[4]  # Latest CCF will be retained after loop
                    allele_frequency = mutation[3]

                    if isinstance(allele_frequency, (int, float)) and allele_frequency != "":
                        allele_frequency_sum += allele_frequency
                        mutation_count += 1

                del instance[1]  # Remove original mutation list
                instance.append(CCF)

                if mutation_count > 0:
                    average_allele_frequency = allele_frequency_sum / mutation_count
                    instance.append(average_allele_frequency)
                else:
                    print(f"No valid allele frequency data found in the mutation list. PatientID: {patient}")
        else:
            print(f"Error: Study number {study} out of range for patient {patient}")

    return raw_data

if __name__ == '__main__':
    data_path = pd.ExcelFile("data/SYUCC&GENE+.xlsx")
    clinical_data = pd.read_excel(data_path, 0, dtype={'Study ID': str, 'PATIENT_ID': str, 'ORR': str})
    clinical_data = clinical_data[['Study ID', 'PATIENT_ID', 'PFS', 'Status', 'ORR']].values.tolist()
    genomic_data = pd.read_excel(data_path, 1, dtype={'Study ID': str, "PATIENT_ID": str}, na_values=[''])
    genomic_data.fillna(0, inplace=True)
    genomic_data = genomic_data.values.tolist()

    patient_list_train, patient_list_test = Clinical_Data_Preprocess(clinical_data)
    instance_list = Genomic_Data_Preprocess(genomic_data)

    raw_data_train = Match(patient_list_train, instance_list)
    raw_data_test = Match(patient_list_test, instance_list)

    divisors = Scaling_Calculations(raw_data_train)

    train_set = Mutation_to_Feature(raw_data_train, divisors)
    test_set = Mutation_to_Feature(raw_data_test, divisors)

    torch.save(raw_data_train, "train_set.pt")
    torch.save(raw_data_test, "test_set.pt")

    print("Data preprocessing completed!")
