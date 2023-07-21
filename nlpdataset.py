import torch.utils.data as data
import numpy as np
from torch.utils.data import DataLoader, Dataset
from data_partition import partition_data
import torch
import os

class SentDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __getitem__(self, i):
        return torch.LongTensor(self.data[i]), self.labels[i]

    def __len__(self) -> int:
        return self.labels.shape[0]
    

def nlpdataset_read(dataset, base_path, batch_size, n_parties, partition, beta, skew_class):
    if dataset == "yahoo_answers":
        traindata = torch.load(os.path.join(base_path, 'yahoo_answers_csv/sents/TRAIN_data.pth.tar'))
        testdata = torch.load(os.path.join(base_path, 'yahoo_answers_csv/sents/TEST_data.pth.tar'))
        train_data = np.array(traindata['sents'])
        train_label = np.array(traindata['labels'])
        test_data = np.array(testdata['sents'])
        test_label = np.array(testdata['labels'])
        n_train = train_label.shape[0]
        net_dataidx_map, traindata_cls_counts, data_distributions = partition_data(partition, n_train, n_parties, train_label, beta, skew_class)
        train_dataloaders = []
        val_dataloaders = []
        for i in range(n_parties):
            train_idxs = net_dataidx_map[i][:int(0.8*len(net_dataidx_map[i]))]
            val_idxs = net_dataidx_map[i][int(0.8*len(net_dataidx_map[i])):]
            train_dataset = SentDataset(data=train_data[train_idxs], labels=train_label[train_idxs])
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            val_dataset = SentDataset(data=train_data[val_idxs], labels=train_label[val_idxs])
            val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
            train_dataloaders.append(train_loader)
            val_dataloaders.append(val_loader)
    test_dataset = SentDataset(data=test_data, labels=test_label)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloaders, val_dataloaders, test_loader, net_dataidx_map, traindata_cls_counts, data_distributions