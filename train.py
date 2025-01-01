# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from utils.dataset import TimeSeriesDataset
from models.model import Informer
from utils.visualization import plot_all

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)

# 超参数
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
EPOCHS = 100
TRAIN_RATIO = 0.8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def score_function(prediction_list, true_y):
    """
    根据误差百分比计算评分函数，对正负误差采用相同的惩罚力度。
    
    Args:
        prediction_list: RUL预测值
        true_y: RUL真实值
    
    Returns:
        float: 对应的评分
    """
    score = 0
    for predicted, actual in zip(prediction_list, true_y):
        percent_error = (actual-predicted)/(actual+0.0001)
        # 统一使用相同的惩罚系数10
        score += np.exp(np.log(0.5) * abs(percent_error) / 10)
            
    return score/len(prediction_list)


def train_one_epoch(model, train_loader, criterion, optimizer):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    valid_batches = 0
    total_batches = len(train_loader)

    for batch_idx, (batch_x, batch_y, batch_x_mark) in enumerate(train_loader):
        try:
            optimizer.zero_grad()

            # 将数据移到GPU并确保数据类型
            batch_x = batch_x.float().to(DEVICE)
            batch_x_mark = batch_x_mark.float().to(DEVICE)
            batch_y = batch_y.float().to(DEVICE)

            # 前向传播
            output = model(batch_x, batch_x_mark)

            if output is None:
                logging.error(
                    f"Batch {batch_idx}/{total_batches} failed: Model returned None output"
                )
                continue

            output = output.squeeze()
            batch_y = batch_y.squeeze()

            # 计算损失
            loss = criterion(output, batch_y)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            valid_batches += 1

            # 每10个batch打印一次进度
            if (batch_idx + 1) % 10 == 0:
                logging.info(
                    f"Training progress: {batch_idx + 1}/{total_batches} batches, Current loss: {loss.item():.4f}"
                )

        except RuntimeError as e:
            logging.error(f"Error in batch {batch_idx}/{total_batches}: {str(e)}")
            logging.error(
                f"Input shapes - batch_x: {batch_x.shape}, batch_y: {batch_y.shape}, batch_x_mark: {batch_x_mark.shape}"
            )
            continue

    return (
        total_loss / (valid_batches * BATCH_SIZE) if valid_batches > 0 else float("inf")
    )


def validate(model, val_loader, criterion):
    """验证模型性能"""
    model.eval()
    total_loss = 0
    total_mae_loss = 0
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
            
            # 获取当前批次的数据集
            dataset = val_loader.dataset.datasets[0] if isinstance(val_loader.dataset, ConcatDataset) else val_loader.dataset
            
            # 计算归一化状态下的损失（用于训练）
            loss = criterion(output, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            mae_loss = nn.L1Loss()(output, batch_y)
            total_mae_loss += mae_loss.item() * batch_x.size(0)
            
            # 反归一化预测值和实际值
            output_np = output.cpu().numpy()
            batch_y_np = batch_y.cpu().numpy()
            predictions_denorm = output_np * dataset.target_scaler.scale_[0] + dataset.target_scaler.mean_[0]
            actuals_denorm = batch_y_np * dataset.target_scaler.scale_[0] + dataset.target_scaler.mean_[0]
            
            predictions.extend(predictions_denorm)
            actuals.extend(actuals_denorm)
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # 计算各种评估指标（使用归一化后的损失）
    val_loss = total_loss / len(val_loader.dataset)  # MSE
    mae = total_mae_loss / len(val_loader.dataset)   # MAE
    rmse = np.sqrt(val_loss)                         # RMSE
    
    # 使用反归一化后的值计算score
    score = score_function(predictions, actuals)      # Score
    
    return val_loss, rmse, mae, score, predictions, actuals


def combined_loss(pred, target):
    criterion = nn.MSELoss()
    mae_loss = nn.L1Loss()
    return criterion(pred, target) + 0.1 * mae_loss(pred, target)


def main():
    try:
        # 创建带有时间戳的保存模型目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        save_dir = Path(f"saved_models/{timestamp}")
        save_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Model outputs will be saved to: {save_dir}")
        logging.info("Starting model training with configuration:")
        logging.info(f"Batch Size: {BATCH_SIZE}, Learning Rate: {LEARNING_RATE}")
        logging.info(f"Weight Decay: {WEIGHT_DECAY}, Epochs: {EPOCHS}")
        logging.info(f"Device: {DEVICE}")

        # 加载所有风机的数据
        turbines = ["T01", "T06", "T07", "T11"]
        train_datasets = []
        val_datasets = []

        for turbine in turbines:
            data_path = f"data/cleaned_data/turbine_{turbine}_cleaned.csv"
            try:
                train_dataset = TimeSeriesDataset(
                    data_path, train=True, train_ratio=TRAIN_RATIO
                )
                val_dataset = TimeSeriesDataset(
                    data_path, train=False, train_ratio=TRAIN_RATIO
                )

                train_datasets.append(train_dataset)
                val_datasets.append(val_dataset)
                logging.info(f"Successfully loaded data for turbine {turbine}")
            except Exception as e:
                logging.error(f"Failed to load data for turbine {turbine}: {str(e)}")
                raise

        # 合并数据集
        combined_train_dataset = ConcatDataset(train_datasets)
        combined_val_dataset = ConcatDataset(val_datasets)

        train_loader = DataLoader(
            combined_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            combined_val_dataset, batch_size=BATCH_SIZE, num_workers=4
        )

        # 获取特征维度
        first_dataset = train_datasets[0]
        feature_dim = first_dataset.feature_dim

        # 初始化模型
        model = Informer(
            enc_in=feature_dim,  # 使用数据集的特征维度
            d_model=128,
            n_heads=8,
            e_layers=3,
            d_ff=512,
            dropout=0.1,
        ).to(DEVICE)

        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            amsgrad=True,
        )

        # 添加学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # 训练循环
        train_loss_list = []
        test_loss_list = []
        best_val_loss = float("inf")
        best_predictions = None
        best_actuals = None
        no_improvement_count = 0

        for epoch in range(EPOCHS):
            logging.info(f"\nStarting Epoch {epoch+1}/{EPOCHS}")

            # 训练
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer)

            # 验证
            val_loss, rmse, mae, score, predictions, actuals = validate(model, val_loader, criterion)

            # 更新学习率
            old_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]["lr"]

            if new_lr != old_lr:
                logging.info(
                    f"Learning rate adjusted from {old_lr:.6f} to {new_lr:.6f}"
                )

            # 记录损失
            train_loss_list.append(train_loss)
            test_loss_list.append(val_loss)

            # 保存最佳模型和对应的预测结果
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_predictions = predictions
                best_actuals = actuals
                model_path = save_dir / 'best_model.pth'
                torch.save(model.state_dict(), model_path)
                logging.info(
                    f"New best model saved! Previous best: {best_val_loss:.4f}, New best: {val_loss:.4f}"
                )
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
            # 打印训练信息
            logging.info(f"Epoch {epoch+1} Summary:")
            logging.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            logging.info(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
            logging.info(f"Score: {score:.4f}")

        # 训练结束后绘制所有图表
        logging.info("Training completed. Generating visualization plots...")
        save_dir = plot_all(
            predictions=best_predictions,
            actuals=best_actuals,
            train_losses=train_loss_list,
            val_losses=test_loss_list,
            save_dir=save_dir
        )
        logging.info(f"All plots have been saved to {save_dir}")

    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
