import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from utils.dataset import TimeSeriesDataset
from models.model import Informer

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


def train_one_epoch(model, train_loader, criterion, optimizer):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    valid_batches = 0

    for batch_x, batch_y, batch_x_mark in train_loader:
        optimizer.zero_grad()

        # 检查输入数据是否包含NaN
        if (
            torch.isnan(batch_x).any()
            or torch.isnan(batch_y).any()
            or torch.isnan(batch_x_mark).any()
        ):
            print("Skipping batch with NaN inputs")
            continue

        # 将数据移到GPU并确保数据类型和requires_grad
        batch_x = batch_x.float().requires_grad_(True).to(DEVICE)
        batch_x_mark = batch_x_mark.float().requires_grad_(True).to(DEVICE)
        batch_y = batch_y.float().to(DEVICE)

        # 前向传播
        output = model(batch_x, batch_x_mark)

        if output is None:
            print("Model returned None output")
            continue

        output = output.squeeze()
        batch_y = batch_y.squeeze()

        # 检查输出是否有效
        if torch.isnan(output).any():
            print("NaN detected in model output")
            continue

        # 计算损失
        try:
            loss = criterion(output, batch_y)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            valid_batches += 1

        except RuntimeError as e:
            print(f"Error in training step: {e}")
            continue

    if valid_batches == 0:
        return float("inf")

    return total_loss / (valid_batches * BATCH_SIZE)


def validate(model, val_loader, criterion):
    """验证模型性能"""
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark in val_loader:  # 修改这里，只接收3个值
            batch_x = batch_x.to(DEVICE)
            batch_x_mark = batch_x_mark.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            output = model(batch_x, batch_x_mark)
            output = output.squeeze()
            batch_y = batch_y.squeeze()

            loss = criterion(output, batch_y)
            total_loss += loss.item() * batch_x.size(0)

            predictions.extend(output.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())

    val_loss = total_loss / len(val_loader.dataset)
    rmse = np.sqrt(val_loss)
    mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))

    return val_loss, rmse, mae


def combined_loss(pred, target):
    criterion = nn.MSELoss()
    mae_loss = nn.L1Loss()
    return criterion(pred, target) + 0.1 * mae_loss(pred, target)


def main():
    # 创建保存模型的目录
    save_dir = Path("saved_models")
    save_dir.mkdir(exist_ok=True)

    # 加载所有风机的数据
    turbines = ["T01", "T06", "T07", "T11"]
    train_datasets = []
    val_datasets = []

    for turbine in turbines:
        data_path = f"data/cleaned_data/turbine_{turbine}_cleaned.csv"
        train_dataset = TimeSeriesDataset(
            data_path, train=True, train_ratio=TRAIN_RATIO
        )
        val_dataset = TimeSeriesDataset(data_path, train=False, train_ratio=TRAIN_RATIO)

        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    # 合并数据集
    combined_train_dataset = ConcatDataset(train_datasets)
    combined_val_dataset = ConcatDataset(val_datasets)

    train_loader = DataLoader(
        combined_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(combined_val_dataset, batch_size=BATCH_SIZE, num_workers=4)

    # 初始化模型
    model = Informer(
        enc_in=16,
        d_model=128,
        n_heads=8,
        e_layers=3,
        d_ff=512,
        dropout=0.1,
    ).to(DEVICE)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, amsgrad=True
    )

    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 参考CNN-Informer的训练循环
    train_loss_list = []
    test_loss_list = []
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        # 训练
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)

        # 验证
        val_loss, rmse, mae = validate(model, val_loader, criterion)

        # 记录损失
        train_loss_list.append(train_loss)
        test_loss_list.append(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                save_dir / f'best_model_{datetime.now().strftime("%Y%m%d_%H%M")}.pth',
            )

        # 打印训练信息
        print(f"Epoch: {epoch}, TrainLoss: {train_loss}, TestLoss: {val_loss}")
        print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # 保存训练记录用于绘图
    np.savetxt("./训练记录.csv", [train_loss_list], delimiter=",")


if __name__ == "__main__":
    main()
