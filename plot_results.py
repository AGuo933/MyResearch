# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from pathlib import Path
from models.model import Informer
from utils.dataset import TimeSeriesDataset
from utils.visualization import plot_all

# 设置设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
TRAIN_RATIO = 0.8

def load_model_and_data(model_path):
    """加载模型和数据"""
    # 加载所有风机的数据
    turbines = ["T01", "T06", "T07", "T11"]
    val_datasets = []

    for turbine in turbines:
        data_path = f"data/cleaned_data/turbine_{turbine}_cleaned.csv"
        val_dataset = TimeSeriesDataset(
            data_path, train=False, train_ratio=TRAIN_RATIO
        )
        val_datasets.append(val_dataset)

    # 合并数据集
    combined_val_dataset = ConcatDataset(val_datasets)
    val_loader = DataLoader(
        combined_val_dataset, batch_size=BATCH_SIZE, num_workers=4
    )

    # 获取特征维度并初始化模型
    first_dataset = val_datasets[0]
    feature_dim = first_dataset.feature_dim
    
    model = Informer(
        enc_in=feature_dim,
        d_model=128,
        n_heads=8,
        e_layers=3,
        d_ff=512,
        dropout=0.1,
    ).to(DEVICE)

    # 加载模型权重
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model, val_loader

def get_predictions(model, val_loader):
    """获取模型预测结果"""
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark in val_loader:
            batch_x = batch_x.to(DEVICE)
            batch_x_mark = batch_x_mark.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            
            output = model(batch_x, batch_x_mark)
            output = output.squeeze()
            batch_y = batch_y.squeeze()
            
            # 直接使用原始值
            output_np = output.cpu().numpy()
            batch_y_np = batch_y.cpu().numpy()
            
            predictions.extend(output_np)
            actuals.extend(batch_y_np)
    
    return np.array(predictions), np.array(actuals)

def main():
    # 设置保存目录
    save_dir = Path("visualization_results")
    save_dir.mkdir(exist_ok=True)
    
    # 加载最新的模型
    model_dir = Path("saved_models")
    model_files = list(model_dir.glob("*/best_model.pth"))
    if not model_files:
        raise ValueError("No model files found!")
    
    # 选择最新的模型文件
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    print(f"Using model: {latest_model}")
    
    # 加载模型和数据
    model, val_loader = load_model_and_data(latest_model)
    
    # 获取预测结果
    predictions, actuals = get_predictions(model, val_loader)
    
    # 绘制所有图表
    print("Generating plots...")
    save_dir = plot_all(predictions=predictions, actuals=actuals, save_dir=save_dir)
    print(f"All plots have been saved to {save_dir}")

if __name__ == "__main__":
    main() 