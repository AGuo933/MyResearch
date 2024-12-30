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
BATCH_SIZE = 32
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0001
EPOCHS = 100
TRAIN_RATIO = 0.8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, train_loader, criterion, optimizer):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    for batch_x, batch_time, batch_y in train_loader:
        optimizer.zero_grad()

        # 将数据移到GPU（如果使用的话）
        batch_x = batch_x.to(DEVICE)
        batch_time = batch_time.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        # 前向传播
        output = model(batch_x, batch_time)
        loss = criterion(output, batch_y)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)

    return total_loss / len(train_loader.dataset)


def validate(model, val_loader, criterion, val_dataset):
    """验证函数"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for batch_x, batch_time, batch_y in val_loader:
            # 将数据移到GPU
            batch_x = batch_x.to(DEVICE)
            batch_time = batch_time.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            # 前向传播
            outputs = model(batch_x, batch_time)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item() * batch_x.size(0)

            # 收集预测结果
            predictions.extend(outputs.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())

    # 计算评估指标
    predictions = np.array(predictions)
    targets = np.array(targets)

    # 将标准化的值转换回原始值
    predictions = val_dataset.inverse_transform_y(predictions)
    targets = val_dataset.inverse_transform_y(targets)

    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mae = np.mean(np.abs(predictions - targets))

    return total_loss / len(val_loader.dataset), rmse, mae


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
        enc_in=17,  # 17个输入特征(16个原始特征 + 1个时间特征)
        d_model=128,  # 嵌入维度
        n_heads=8,  # 注意力头数
        e_layers=3,  # encoder层数
        d_ff=512,  # 前馈网络维度
        dropout=0.2,
    ).to(DEVICE)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # 训练循环
    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        # 训练
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)

        # 验证
        val_loss, rmse, mae = validate(model, val_loader, criterion, val_datasets[0])

        # 记录日志
        logging.info(f"Epoch {epoch+1}/{EPOCHS}:")
        logging.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logging.info(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                save_dir / f'best_model_{datetime.now().strftime("%Y%m%d_%H%M")}.pth',
            )


if __name__ == "__main__":
    main()
