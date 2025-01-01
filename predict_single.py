# -*- coding: utf-8 -*-

import torch
from pathlib import Path
from models.model import Informer
from utils.dataset import TimeSeriesDataset
import logging
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_RATIO = 0.8
BATCH_SIZE = 32  # 添加批次大小参数

def predict_single_sample(model_timestamp, turbine_id, sample_index):
    """
    预测指定样本的值
    Args:
        model_timestamp: 模型时间戳文件夹名称，如"20231201_1430"
        turbine_id: 风机ID，如"T01"
        sample_index: 要预测的样本索引
    """
    try:
        # 加载模型
        model_dir = Path(f"saved_models/{model_timestamp}")
        model_path = model_dir / "best_model.pth"
        
        if not model_path.exists():
            raise ValueError(f"Model not found at {model_path}")
            
        # 加载数据
        data_path = f"data/cleaned_data/turbine_{turbine_id}_cleaned.csv"
        dataset = TimeSeriesDataset(data_path, train=False, train_ratio=TRAIN_RATIO)
        
        if sample_index >= len(dataset):
            raise ValueError(f"Sample index {sample_index} exceeds dataset length {len(dataset)}")
        
        # 获取单个样本
        x, y, x_mark = dataset[sample_index]
        
        # 调整输入维度以匹配模型期望的输入格式，並確保序列長度至少為2
        x = x.unsqueeze(0)  # [batch_size=1, feature_dim]
        x = x.unsqueeze(1)  # [batch_size=1, seq_len=1, feature_dim]
        x = x.repeat(1, 2, 1)  # 重複一次得到序列長度為2
        x_mark = x_mark.unsqueeze(0)  # [batch_size=1, time_feature_dim]
        x_mark = x_mark.unsqueeze(1)  # [batch_size=1, seq_len=1, time_feature_dim]
        x_mark = x_mark.repeat(1, 2, 1)  # 重複一次得到序列長度為2
        
        # 打印输入形状以进行调试
        print(f"Input shapes - x: {x.shape}, x_mark: {x_mark.shape}")
        
        # 初始化模型
        feature_dim = dataset.feature_dim
        model = Informer(
            enc_in=feature_dim,
            d_model=128,
            n_heads=8,
            e_layers=3,
            d_ff=512,
            dropout=0.1,
        ).to(DEVICE)
        
        # 加载模型权重
        checkpoint = torch.load(model_path)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 直接加载状态字典
            model.load_state_dict(checkpoint)
        model.eval()
        
        # 预测
        with torch.no_grad():
            prediction = model(x.to(DEVICE), x_mark.to(DEVICE))
            if prediction is not None:
                # 只取第一个预测结果
                prediction = prediction[0].squeeze().cpu().numpy()
                
                # 反归一化预测值和实际值
                prediction = prediction * dataset.target_scaler.scale_[0] + dataset.target_scaler.mean_[0]
                actual = y.item() * dataset.target_scaler.scale_[0] + dataset.target_scaler.mean_[0]
                
                print("\n预测结果:")
                print(f"样本索引: {sample_index}")
                print(f"预测值: {prediction:.2f}")
                print(f"实际值: {actual:.2f}")
                print(f"误差: {abs(prediction - actual):.2f}")
                print(f"相对误差: {(abs(prediction - actual) / actual * 100):.2f}%")
            else:
                print("模型预测失败，返回了None")
        
    except Exception as e:
        print(f"预测过程出错: {str(e)}")
        # 打印更详细的错误信息
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    # 示例使用
    MODEL_TIMESTAMP = "20250101_1154"  # 替换为实际的模型时间戳
    TURBINE_ID = "T01"                 # 选择风机
    SAMPLE_INDEX = -1                # 最后一个样本
    
    predict_single_sample(MODEL_TIMESTAMP, TURBINE_ID, SAMPLE_INDEX) 