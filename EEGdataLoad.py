import os.path
import torch
import scipy.io as io
from torch.utils.data import Dataset, ConcatDataset
import numpy as np
from einops import rearrange


class EEGLoader2a(Dataset):
    def __init__(self, num_subject, t):

        # datamat = io.loadmat('F:/dataset/BCI2008/2a/data/A01E.mat')
        path2a='D:/chenwan/data/BCI2008/2a/data/'
        # path2a = 'F:/dataset/BCI2008/2a/data/'
        data_path = os.path.join(path2a, num_subject + '.mat')
        datamat = io.loadmat(data_path)
        datamat = datamat['data']
        runs = datamat.shape[1]
        data = []
        label = []
        fs = 250
        if runs == 9:
            for run in range(3, 9):
                data_all = datamat[0][run]
                data1 = data_all[0][0]['X']
                label_run = data_all[0][0]['y'] - 1
                label_run = (torch.from_numpy(label_run)).type('torch.LongTensor')
                label.append(label_run)
                trail = data_all[0][0]['trial']
                trail = np.squeeze(trail)
                for i in range(len(label_run)):
                    data2 = data1[trail[i] + round(fs*t[0]):trail[i] + round(fs*t[1]), 0:22]
                    data.append(data2)
        else:
            for run in range(1, 7):
                data_all = datamat[0][run]
                data1 = data_all[0][0]['X']
                label_run = data_all[0][0]['y'] - 1
                label_run = (torch.from_numpy(label_run)).type('torch.LongTensor')
                label.append(label_run)
                trail = data_all[0][0]['trial']
                trail = np.squeeze(trail)
                for i in range(len(label_run)):
                    data2 = data1[trail[i] + round(fs*t[0]):trail[i] + round(fs*t[1]), 0:22]
                    data.append(data2)

        data = torch.tensor(np.array(data)).type('torch.FloatTensor')
        data = rearrange(data, 'b t c -> b c t')
        data = data.unsqueeze(dim=1)
        label = torch.cat(label, dim=0)

        self.data=data
        self.len=len(label)
        self.label=label.squeeze()

    def __getitem__(self, index):
        data=self.data[index]
        labels=self.label[index]
        return data, labels

    def __len__(self):
        return self.len

class EEGLoader2b(Dataset):
    def __init__(self, num_subject, t):

        # datamat = io.loadmat('F:/dataset/BCI2008/2a/data/A01E.mat')
        path2a='D:/chenwan/data/BCI2008/2b/data/'
        # path2a = 'F:/dataset/BCI2008/2b/data/'
        data_path = os.path.join(path2a, num_subject + '.mat')
        datamat = io.loadmat(data_path)
        datamat = datamat['data']
        runs = datamat.shape[1]
        data = []
        label = []
        fs = 250

        for run in range(runs):
            data_all = datamat[0][run]
            data1 = data_all[0][0]['X']
            label_run = data_all[0][0]['y'] - 1
            label_run = (torch.from_numpy(label_run)).type('torch.LongTensor')
            label.append(label_run)
            trail = data_all[0][0]['trial']
            trail = np.squeeze(trail)
            for i in range(len(label_run)):
                data2 = data1[trail[i] + round(fs*t[0]):trail[i] + round(fs*t[1]), 0:3]
                data.append(data2)

        data = torch.tensor(np.array(data)).type('torch.FloatTensor')
        data = rearrange(data, 'b t c -> b c t')
        data = data.unsqueeze(dim=1)
        label = torch.cat(label, dim=0)


        self.data=data
        self.len=len(label)
        self.label=label.squeeze()

    def __getitem__(self, index):
        data=self.data[index]
        labels=self.label[index]
        return data, labels

    def __len__(self):
        return self.len

class EEGLoader2a2(Dataset):
    def __init__(self, num_subject, t):

        # datamat = io.loadmat('F:/dataset/BCI2008/2a/data/A01E.mat')
        path2a='D:/chenwan/data/BCI2008/2a/data/'
        # path2a = 'F:/dataset/BCI2008/2a/data/'
        data_path = os.path.join(path2a, num_subject + '.mat')
        datamat = io.loadmat(data_path)
        datamat = datamat['data']
        runs = datamat.shape[1]
        data = []
        label = []
        fs = 250
        if runs == 9:
            for run in range(3, 9):
                data_all = datamat[0][run]
                data1 = data_all[0][0]['X']
                label_run = data_all[0][0]['y'] - 1
                label_run = (torch.from_numpy(label_run)).type('torch.LongTensor')
                label.append(label_run)
                trail = data_all[0][0]['trial']
                trail = np.squeeze(trail)
                for i in range(len(label_run)):
                    data2 = data1[trail[i] + round(fs*t[0]):trail[i] + round(fs*t[1]), 0:22]
                    data.append(data2)
        else:
            for run in range(1, 7):
                data_all = datamat[0][run]
                data1 = data_all[0][0]['X']
                label_run = data_all[0][0]['y'] - 1
                label_run = (torch.from_numpy(label_run)).type('torch.LongTensor')
                label.append(label_run)
                trail = data_all[0][0]['trial']
                trail = np.squeeze(trail)
                for i in range(len(label_run)):
                    data2 = data1[trail[i] + round(fs*t[0]):trail[i] + round(fs*t[1]), 0:22]
                    data.append(data2)

        data = torch.tensor(np.array(data)).type('torch.FloatTensor')
        data = rearrange(data, 'b t c -> b c t')
        data = data.unsqueeze(dim=1)
        label = torch.cat(label, dim=0)
        label = label.squeeze()

        mask = (label==0) | (label==1)
        data = data[mask]
        label = label[mask]

        self.data=data
        self.len=len(label)
        self.label=label

    def __getitem__(self, index):
        data=self.data[index]
        labels=self.label[index]
        return data, labels

    def __len__(self):
        return self.len

class EEGLoader2a3(Dataset):
    def __init__(self, num_subject, t):

        # datamat = io.loadmat('F:/dataset/BCI2008/2a/data/A01E.mat')
        path2a='D:/chenwan/data/BCI2008/2a/data/'
        # path2a = 'F:/dataset/BCI2008/2a/data/'
        data_path = os.path.join(path2a, num_subject + '.mat')
        datamat = io.loadmat(data_path)
        datamat = datamat['data']
        runs = datamat.shape[1]
        data = []
        label = []
        fs = 250
        if runs == 9:
            for run in range(3, 9):
                data_all = datamat[0][run]
                data1 = data_all[0][0]['X']
                label_run = data_all[0][0]['y'] - 1
                label_run = (torch.from_numpy(label_run)).type('torch.LongTensor')
                label.append(label_run)
                trail = data_all[0][0]['trial']
                trail = np.squeeze(trail)
                for i in range(len(label_run)):
                    data2 = data1[trail[i] + round(fs*t[0]):trail[i] + round(fs*t[1]), 0:22]
                    data.append(data2)
        else:
            for run in range(1, 7):
                data_all = datamat[0][run]
                data1 = data_all[0][0]['X']
                label_run = data_all[0][0]['y'] - 1
                label_run = (torch.from_numpy(label_run)).type('torch.LongTensor')
                label.append(label_run)
                trail = data_all[0][0]['trial']
                trail = np.squeeze(trail)
                for i in range(len(label_run)):
                    data2 = data1[trail[i] + round(fs*t[0]):trail[i] + round(fs*t[1]), 0:22]
                    data.append(data2)

        data = torch.tensor(np.array(data)).type('torch.FloatTensor')
        data = rearrange(data, 'b t c -> b c t')
        data = data.unsqueeze(dim=1)
        label = torch.cat(label, dim=0)
        label = label.squeeze()

        mask = (label==0) | (label==1)
        data = data[mask]
        data = data[:,:,[7,9,11],:]
        label = label[mask]

        self.data=data
        self.len=len(label)
        self.label=label

    def __getitem__(self, index):
        data=self.data[index]
        labels=self.label[index]
        return data, labels

    def __len__(self):
        return self.len


def loader_2a2(t):
    nameE = ['A01E', 'A02E', 'A03E', 'A04E', 'A05E', 'A06E', 'A07E', 'A08E', 'A09E']  # 受试者编码
    nameT = ['A01T', 'A02T', 'A03T', 'A04T', 'A05T', 'A06T', 'A07T', 'A08T', 'A09T']  # 受试者编码

    for sub_num in range(9):
        nameE_sub = nameE[sub_num]
        nameT_sub = nameT[sub_num]

        dataloaderE = EEGLoader2a3(nameE_sub, t)
        dataloaderT = EEGLoader2a3(nameT_sub, t)
        if sub_num == 0:
            dataloader = ConcatDataset([dataloaderE, dataloaderT])  # 合并数据集
        else:
            dataloader = ConcatDataset([dataloader, dataloaderE, dataloaderT])

    return dataloader


def loader_2b(t):
    nameE = ['B01E', 'B02E', 'B03E', 'B04E', 'B05E', 'B06E', 'B07E', 'B08E', 'B09E']  # 受试者编码
    nameT = ['B01T', 'B02T', 'B03T', 'B04T', 'B05T', 'B06T', 'B07T', 'B08T', 'B09T']  # 受试者编码

    for sub_num in range(9):
        nameE_sub = nameE[sub_num]
        nameT_sub = nameT[sub_num]

        dataloaderE = EEGLoader2b(nameE_sub, t)
        dataloaderT = EEGLoader2b(nameT_sub, t)
        if sub_num == 0:
            dataloader = ConcatDataset([dataloaderE, dataloaderT])  # 合并数据集
        else:
            dataloader = ConcatDataset([dataloader, dataloaderE, dataloaderT])

    return dataloader



