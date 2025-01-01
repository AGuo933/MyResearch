# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import gaussian_kde

def set_plot_style():
    """设置绘图样式"""
    plt.style.use('seaborn')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False

def plot_loss_curve(train_losses, val_losses, save_dir):
    """绘制训练过程中的损失变化曲线"""
    set_plot_style()
    
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和验证损失变化曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_comparison(predictions, actuals, save_dir):
    """绘制预测值vs实际值的对比图"""
    set_plot_style()
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(predictions))
    
    # 绘制实际值和预测值
    plt.plot(x, actuals, label='实际值', color='blue', linewidth=2)
    plt.plot(x, predictions, label='预测值', color='red', linewidth=2)
    
    # 添加误差区域
    error = predictions - actuals
    plt.fill_between(x, np.zeros_like(x), error, 
                    where=(error >= 0), 
                    color='red', alpha=0.3, 
                    label='正误差')
    plt.fill_between(x, np.zeros_like(x), error, 
                    where=(error <= 0), 
                    color='blue', alpha=0.3, 
                    label='负误差')
    
    plt.xlabel('样本')
    plt.ylabel('RUL')
    plt.title('RUL预测结果对比')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'rul_prediction.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_distribution(predictions, actuals, save_dir):
    """绘制每个样本的误差大小"""
    set_plot_style()
    
    errors = predictions - actuals
    plt.figure(figsize=(12, 6))
    x = np.arange(len(errors))
    
    # 绘制误差柱状图
    plt.bar(x, errors, alpha=0.6, color=['red' if e >= 0 else 'blue' for e in errors],
           label='预测误差')
    
    # 添加零误差线
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='零误差线')
    
    # 计算一些统计量
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    # 添加均值线
    plt.axhline(y=mean_error, color='r', linestyle='--', alpha=0.5, 
               label=f'平均误差: {mean_error:.2f}')
    
    # 添加标准差范围
    plt.fill_between(x, mean_error - std_error, mean_error + std_error,
                    color='gray', alpha=0.2, 
                    label=f'标准差范围: ±{std_error:.2f}')
    
    plt.xlabel('样本')
    plt.ylabel('误差值')
    plt.title('每个样本的预测误差')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_score_function(save_dir):
    """绘制评分函数图像"""
    set_plot_style()
    
    plt.figure(figsize=(10, 5))
    x = np.linspace(-100, 100, 1000)
    y = np.exp(np.log(0.5) * np.abs(x) / 10)
    
    plt.plot(x, y, 'g-', linewidth=2)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    
    # 添加说明文字
    plt.text(50, 0.8, '对称惩罚: exp(ln(0.5)|error|/10)', 
             fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # 添加特殊点的标注
    plt.plot(0, 1, 'ro', label='无误差点 (0, 1)')
    plt.plot(10, 0.5, 'bo', label='半分点 (10, 0.5)')
    
    plt.xlabel('误差百分比')
    plt.ylabel('评分')
    plt.title('评分函数曲线（对称惩罚）')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'score_function.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_all(predictions=None, actuals=None, train_losses=None, val_losses=None, save_dir=None):
    """绘制所有图表的便捷函数"""
    if save_dir is None:
        save_dir = Path("visualization_results")
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # 如果提供了训练历史，则绘制损失曲线
    if train_losses is not None and val_losses is not None:
        plot_loss_curve(train_losses, val_losses, save_dir)
        
    # 如果提供了预测结果，则绘制预测相关的图表
    if predictions is not None and actuals is not None:
        plot_prediction_comparison(predictions, actuals, save_dir)
        plot_error_distribution(predictions, actuals, save_dir)
    
    # 始终绘制评分函数
    plot_score_function(save_dir)
    
    return save_dir 