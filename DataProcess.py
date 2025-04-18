import numpy as np
import pandas as pd
import torch
import os
import joblib
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

class DataProcess():
    def __init__(self, mode, data_path, use_gpu, gpu_num):
        self.mode = mode
        self.x_path = os.path.join(data_path, f'x_{self.mode}.txt')
        self.y_path = os.path.join(data_path, f'y_{self.mode}.txt')
        self.use_gpu = use_gpu
        self.gpu_num = gpu_num

    def preprocess_data(self):
        if self.use_gpu:
            device = torch.device(f'cuda:{self.gpu_num}')
        else: device = torch.device('cpu')

        x_data = np.loadtxt(self.x_path)
        y_data = np.loadtxt(self.y_path)-1
        if self.mode == 'train':
            # 训练集：计算均值μ和方差σ并转换成均值为0，方差为1的高斯分布，并保存在scaler.pkl文件中
            scaler = StandardScaler()
            x_data = scaler.fit_transform(x_data)
            joblib.dump(scaler, 'train_scaler.pkl')
        else:
            # 测试集需要使用训练集的均值和方差进行转换
            scaler = joblib.load('train_scaler.pkl')
            x_data = scaler.transform(x_data)

        data = TensorDataset(torch.FloatTensor(x_data), torch.LongTensor(y_data))
        return data

    def get_loader(self, batch_size, num_workers):
        print('Starting {} dataset creation process...'.format(self.mode))
        data = self.preprocess_data()
        data_loader = None
        if self.mode == 'train':
            data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        else:
            data_loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=num_workers)
        print('{} dataset creation process Done.\n'.format(self.mode))

        return data_loader

