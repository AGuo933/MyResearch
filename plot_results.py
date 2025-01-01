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

def load_model_and_data():
    """加载模型和数据"""
    # 查找最新的模型目录
    model_root = Path("saved_models")
    if not model_root.exists():
        raise ValueError("No saved_models directory found")
    
    model_dirs = [d for d in model_root.iterdir() if d.is_dir()]
    if not model_dirs:
        raise ValueError("No model directories found")
    
    # 按时间戳排序，获取最新的目录
    latest_dir = max(model_dirs, key=lambda x: x.name)
    model_path = latest_dir / "best_model.pth"
    
    if not model_path.exists():
        raise ValueError(f"Model file not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    
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

    # 加载模型权重和训练历史
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 获取训练历史
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])

    return model, val_loader, train_losses, val_losses

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
            
            predictions.extend(output.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())
    
    return np.array(predictions), np.array(actuals)

def main():
    # 设置保存目录
    save_dir = Path("visualization_results")
    save_dir.mkdir(exist_ok=True)
    
    try:
        # 加载模型和数据
        print("Loading model and data...")
        model, val_loader, _, _ = load_model_and_data()  # 忽略训练历史
        
        # 获取预测结果
        print("Generating predictions...")
        predictions, actuals = get_predictions(model, val_loader)
        
        # 绘制所有图表
        print("Generating plots...")
        save_dir = plot_all(
            predictions=predictions, 
            actuals=actuals,
            save_dir=save_dir
        )
        print(f"All plots have been saved to {save_dir}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main() 