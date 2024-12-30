import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from utils.timefeatures import time_features


class TimeSeriesDataset(Dataset):
    def __init__(self, filepath, train=True, train_ratio=0.8):
        """
        Args:
            filepath: CSV文件路径
            train: 是否为训练集
            train_ratio: 训练集比例
        """
        # 读取CSV文件
        df = pd.read_csv(filepath)

        # 将时间字符串转换为datetime对象
        df["datetime"] = pd.to_datetime(df.iloc[:, 0])

        # 使用 timefeatures 提取时间特征
        dates_df = pd.DataFrame({"date": df["datetime"]})
        time_features_data = time_features(
            dates_df, timeenc=1, freq="h"
        )  # 返回 4 个特征

        # 提取特征和标签
        features = df.iloc[:, 1:-1].values  # 排除第一列(时间)和最后一列(标签)
        labels = df.iloc[:, -1].values  # 最后一列(power)

        # 数据标准化 (只标准化特征，不标准化时间)
        self.feature_means = features.mean(axis=0)
        self.feature_stds = features.std(axis=0)
        self.label_mean = labels.mean()
        self.label_std = labels.std()

        normalized_features = (features - self.feature_means) / self.feature_stds
        normalized_labels = (labels - self.label_mean) / self.label_std

        # 划分训练集和测试集
        total_len = len(df)
        train_size = int(total_len * train_ratio)

        if train:
            self.x_data = normalized_features[:train_size]
            self.y_data = normalized_labels[:train_size]
            self.time_data = time_features_data[:train_size]
        else:
            self.x_data = normalized_features[train_size:]
            self.y_data = normalized_labels[train_size:]
            self.time_data = time_features_data[train_size:]

        # 转换为torch tensor
        self.features = torch.FloatTensor(self.x_data)
        self.labels = torch.FloatTensor(self.y_data)
        self.time_features = torch.FloatTensor(self.time_data)
        self.len = len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.time_features[index], self.labels[index]

    def __len__(self):
        return self.len

    def inverse_transform_y(self, y):
        """将标准化的y值转换回原始值"""
        return y * self.label_std + self.label_mean
