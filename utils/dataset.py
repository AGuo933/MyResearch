import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from utils.timefeatures import time_features
from sklearn.preprocessing import StandardScaler


class TimeSeriesDataset(Dataset):
    def __init__(self, filepath, train=True, train_ratio=0.8):
        """
        Args:
            filepath: CSV文件路径
            train: 是否为训练集
            train_ratio: 训练集比例
        """
        # 读取CSV
        df = pd.read_csv(filepath, parse_dates=[0])

        # 提取时间特征
        time_features = self._extract_time_features(df.iloc[:, 0])

        # 提取输入特征 (16列) 并归一化
        features = df.iloc[:, 1:17].values
        self.feature_scaler = StandardScaler()
        normalized_features = self.feature_scaler.fit_transform(features)

        # 归一化目标值 (power)
        self.target_scaler = StandardScaler()
        normalized_labels = self.target_scaler.fit_transform(
            df.iloc[:, -1].values.reshape(-1, 1)
        )

        # 划分训练集/验证集
        train_size = int(len(df) * train_ratio)
        if train:
            self.features = torch.FloatTensor(normalized_features[:train_size])
            self.labels = torch.FloatTensor(normalized_labels[:train_size])
            self.time_features = torch.FloatTensor(time_features[:train_size])
        else:
            self.features = torch.FloatTensor(normalized_features[train_size:])
            self.labels = torch.FloatTensor(normalized_labels[train_size:])
            self.time_features = torch.FloatTensor(time_features[train_size:])

    def _extract_time_features(self, time_series):
        # 提取时间特征：小时、星期几、月份等
        hour = time_series.dt.hour.values / 24.0
        weekday = time_series.dt.weekday.values / 7.0
        month = time_series.dt.month.values / 12.0
        return np.stack([hour, weekday, month], axis=1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            self.features[idx].unsqueeze(0),  # 添加时间维度
            self.labels[idx],
            self.time_features[idx],
        )

    def inverse_transform_y(self, y):
        """将标准化的y值转换回原始值"""
        return y * self.label_std + self.label_mean
