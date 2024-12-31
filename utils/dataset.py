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
        df = pd.read_csv(
            filepath,
            parse_dates=[0],  # 第一列解析为日期
            dtype={col: float for col in range(1, 18)},  # 其他列为float
        )

        # 提取时间特征
        dates_df = pd.DataFrame({"date": df.iloc[:, 0]})
        time_features_data = time_features(dates_df, timeenc=1, freq="h")

        # 提取特征和标签
        features = df.iloc[:, 1:-1].values  # 16列输入特征
        labels = df.iloc[:, -1].values  # 最后一列作为标签

        # 数据标准化
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
            self.features = torch.FloatTensor(normalized_features[:train_size])
            self.labels = torch.FloatTensor(normalized_labels[:train_size])
            self.time_features = torch.FloatTensor(time_features_data[:train_size])
        else:
            self.features = torch.FloatTensor(normalized_features[train_size:])
            self.labels = torch.FloatTensor(normalized_labels[train_size:])
            self.time_features = torch.FloatTensor(time_features_data[train_size:])

        # 重塑特征维度
        self.features = self.features.unsqueeze(
            1
        )  # [N, feature_dim] -> [N, 1, feature_dim]

    def __getitem__(self, index):
        """返回单个样本的特征和标签"""
        return (
            self.features[index],  # [1, feature_dim]
            self.labels[index],  # [1]
            self.time_features[index],  # [time_feature_dim]
        )

    def __len__(self):
        """返回数据集大小"""
        return len(self.features)

    def inverse_transform_y(self, y):
        """将标准化的y值转换回原始值"""
        return y * self.label_std + self.label_mean
