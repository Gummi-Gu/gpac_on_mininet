import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import os
from datetime import datetime
from tqdm import tqdm

# 假设模型和数据集类已导入
from Client.model.model import build_model
from Client.model.dataset import VRDataset



save_target=[]
def save(quaternions=None, csv_file=None):

    # 打开 CSV 文件并写入数据
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # 写入表头
        writer.writerow(['UnitQuaternion.x', 'UnitQuaternion.y', 'UnitQuaternion.z', 'UnitQuaternion.w'])

        # 写入四元数数据
        for i in quaternions:
            writer.writerows(i)

    print(f"数据已保存到 {csv_file}")

def train():
    # 训练参数
    config = {
        'batch_size': 16,
        'num_epochs': 1,
        'lr': 1e-5,
        'lr_step_size': 10,
        'lr_gamma': 0.1,
        'weight_decay': 1e-5,
        'save_dir': 'Client/model/checkpoints',
        'log_interval': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'val': True
    }

    # 初始化数据集和数据加载器

    train_dataset = VRDataset(
        video_root='Client/model/data/train/video',
        motion_csv_root='Client/model/data/train/csv',
        temporal_length=32
    )
    if config['val']:
        val_dataset = VRDataset(
            video_root='Client/model/data/val/video',
            motion_csv_root='Client/model/data/val/csv',
            temporal_length=32
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    if config['val']:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            num_workers=4
        )

    print(f"训练集样本量: {len(train_dataset)}")
    print(f"Batch数目: {len(train_loader)}")

    # 初始化模型
    model = build_model(pretrained=True).to(config['device'])

    def quaternion_angle_loss(q_pred, q_true):
        # 归一化四元数
        q_pred = q_pred / torch.norm(q_pred, dim=-1, keepdim=True)
        q_true = q_true / torch.norm(q_true, dim=-1, keepdim=True)

        # 计算点积
        dot_product = torch.sum(q_pred * q_true, dim=-1)
        dot_product_abs = torch.clamp(torch.abs(dot_product), 1e-6, 1.0)  # 防止梯度爆炸

        # 计算角度差（弧度）
        angle_diff = 2 * torch.acos(dot_product_abs)

        return torch.mean(angle_diff ** 2)  # 平方误差
    criterion = quaternion_angle_loss
    optimizer = Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = StepLR(
        optimizer,
        step_size=config['lr_step_size'],
        gamma=config['lr_gamma']
    )

    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_val_loss = float('inf')

    # 训练循环
    for epoch in range(config['num_epochs']):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1} Training')

        for batch_idx, batch in enumerate(progress_bar):
            save_target.append(batch['target'])
            # 数据转移到设备
            video = batch['video'].to(config['device'], dtype=torch.float32)
            motion = batch['motion'].to(config['device'], dtype=torch.float32)
            target = batch['target'].to(config['device'], dtype=torch.float32)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(video, motion)[0]

            # 计算损失（仅取最后一个时间步的预测与目标）
            loss = criterion(outputs, target)  # 假设目标取最后一个时间步的平均

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if batch_idx % config['log_interval'] == 0:
                progress_bar.set_postfix({'loss': loss.item()})
                print("")

        avg_train_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch + 1} Train Loss: {avg_train_loss:.4f}')

        if config['val']:
            # 验证阶段
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                val_progress = tqdm(val_loader, desc=f'Epoch {epoch + 1} Validation')
                for batch in val_progress:
                    video = batch['video'].to(config['device'], dtype=torch.float32)
                    motion = batch['motion'].to(config['device'], dtype=torch.float32)
                    target = batch['target'].to(config['device'], dtype=torch.float32)

                    outputs = model(video, motion)[0]
                    loss = criterion(outputs, target)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f'Epoch {epoch + 1} Val Loss: {avg_val_loss:.4f}')

            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': best_val_loss
                }
                torch.save(checkpoint, os.path.join(config['save_dir'], f'best_model_{timestamp}.pt'))
        # 更新学习率
        scheduler.step()


if __name__ == '__main__':
    train()
    save(save_target,"Client/model/data/target.csv")