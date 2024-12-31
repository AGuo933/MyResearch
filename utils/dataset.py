import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from utils.timefeatures import time_features
from sklearn.preprocessing import StandardScaler
import logging


class TimeSeriesDataset(Dataset):
    def __init__(self, filepath, train=True, train_ratio=0.8):
        """
        Args:
            filepath: CSV文件路径
            train: 是否为训练集
            train_ratio: 训练集比例
        """
        try:
            # 读取CSV
            df = pd.read_csv(filepath, parse_dates=[0])

            # 计算特征数量（总列数减去时间列和标签列）
            self.feature_dim = len(df.columns) - 2
            logging.info(f"Dataset initialization - File: {filepath}")
            logging.info(
                f"Total samples: {len(df)}, Feature dimension: {self.feature_dim}"
            )

            # 1. 提取时间特征
            time_features = self._extract_time_features(df.iloc[:, 0])

            # 2. 提取输入特征并归一化
            features = df.iloc[:, 1:-1].values
            self.feature_scaler = StandardScaler()
            normalized_features = self.feature_scaler.fit_transform(features)
            self.feature_means = features.mean(axis=0)

            # 3. 归一化目标值
            self.target_scaler = StandardScaler()
            normalized_labels = self.target_scaler.fit_transform(
                df.iloc[:, -1].values.reshape(-1, 1)
            )
            self.label_mean = df.iloc[:, -1].mean()
            self.label_std = self.target_scaler.scale_

            # 4. 划分训练集/验证集
            train_size = int(len(df) * train_ratio)
            if train:
                self.features = torch.FloatTensor(normalized_features[:train_size])
                self.labels = torch.FloatTensor(normalized_labels[:train_size])
                self.time_features = torch.FloatTensor(time_features[:train_size])
                logging.info(f"Training set created with {train_size} samples")
            else:
                self.features = torch.FloatTensor(normalized_features[train_size:])
                self.labels = torch.FloatTensor(normalized_labels[train_size:])
                self.time_features = torch.FloatTensor(time_features[train_size:])
                logging.info(
                    f"Validation set created with {len(df) - train_size} samples"
                )

        except Exception as e:
            logging.error(f"Failed to initialize dataset from {filepath}")
            logging.error(f"Error details: {str(e)}")
            raise

    def _extract_time_features(self, time_series):
        # 提取时间特征：小时、星期几、月份等
        hour = time_series.dt.hour.values / 24.0
        weekday = time_series.dt.weekday.values / 7.0
        month = (time_series.dt.month.values - 1) / 11.0

        # 组合时间特征
        time_features = np.stack([hour, weekday, month], axis=1)
        return time_features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.time_features[idx]
