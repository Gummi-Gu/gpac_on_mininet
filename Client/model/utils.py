import torch
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import defaultdict


# 保存模型
def save_model(model, optimizer, epoch, loss, model_path):
    """
    保存训练的模型及优化器状态
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(state, model_path)
    print(f'Model saved to {model_path}')


# 加载模型
def load_model(model, optimizer, model_path):
    """
    加载预训练模型和优化器状态
    """
    if os.path.isfile(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, optimizer, epoch, loss
    else:
        print(f"No checkpoint found at {model_path}")
        return model, optimizer, 0, None


# 计算平均值和标准差
def compute_mean_std(dataset, batch_size=16, num_workers=4):
    """
    计算数据集的均值和标准差
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    mean = 0.0
    std = 0.0
    total_images_count = 0

    for batch in dataloader:
        images = batch['video']
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)  # [B, C, H, W] -> [B, C, H*W]

        mean += images.mean(dim=2).sum(dim=0)
        std += images.std(dim=2).sum(dim=0)
        total_images_count += batch_size

    mean /= total_images_count
    std /= total_images_count

    return mean, std


# 可视化函数：可视化视频帧、运动数据、预测结果
def visualize_video(video_frames, pred_saliency=None, target_saliency=None, title=""):
    """
    可视化视频帧和显著图
    """
    fig, ax = plt.subplots(1, len(video_frames) + 2, figsize=(12, 6))

    # 显示视频帧
    for i, frame in enumerate(video_frames):
        ax[i].imshow(frame)
        ax[i].axis('off')
        ax[i].set_title(f"Frame {i + 1}")

    if pred_saliency is not None:
        ax[-2].imshow(pred_saliency, cmap='hot', interpolation='nearest')
        ax[-2].set_title("Predicted Saliency")
        ax[-2].axis('off')

    if target_saliency is not None:
        ax[-1].imshow(target_saliency, cmap='hot', interpolation='nearest')
        ax[-1].set_title("Target Saliency")
        ax[-1].axis('off')

    plt.suptitle(title)
    plt.show()


# 计算模型准确率（示例：根据预测的头部位置与目标位置计算）
def compute_accuracy(pred_pos, target_pos):
    """
    计算预测的头部位置与目标位置的均方误差（MSE）
    """
    mse = torch.mean((pred_pos - target_pos) ** 2)
    return mse.item()


# 打印训练和验证过程的日志
def log_training_progress(epoch, train_loss, val_loss=None, val_acc=None, logger=None):
    """
    打印训练过程的日志，显示当前训练损失、验证损失等信息
    """
    log_msg = f"Epoch {epoch}: Train Loss = {train_loss:.4f}"

    if val_loss is not None:
        log_msg += f", Val Loss = {val_loss:.4f}"

    if val_acc is not None:
        log_msg += f", Val Accuracy = {val_acc:.4f}"

    print(log_msg)

    if logger is not None:
        logger.append(log_msg)


# 用于生成对比结果图的工具
def plot_comparison(predictions, ground_truths, output_dir="comparison"):
    """
    绘制对比图并保存到指定的输出文件夹
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(pred)
        axes[0].set_title(f"Predicted {idx + 1}")
        axes[0].axis('off')

        axes[1].imshow(gt)
        axes[1].set_title(f"Ground Truth {idx + 1}")
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"comparison_{idx + 1}.png"))
        plt.close()


# 可视化数据样本（例如：视频和运动数据）
def visualize_sample(sample, show_video=True, show_motion=True):
    """
    可视化一个数据样本，包括视频帧和运动数据
    """
    if show_video:
        video = sample['video']
        fig, ax = plt.subplots(1, len(video), figsize=(15, 6))
        for i, frame in enumerate(video):
            ax[i].imshow(frame.permute(1, 2, 0).numpy())  # 转换为 HWC 格式
            ax[i].axis('off')
            ax[i].set_title(f"Frame {i + 1}")
        plt.show()

    if show_motion:
        motion = sample['motion'].numpy()
        plt.plot(motion[:, 0], label="Quat X")
        plt.plot(motion[:, 1], label="Quat Y")
        plt.plot(motion[:, 2], label="Quat Z")
        plt.plot(motion[:, 3], label="Quat W")
        plt.plot(motion[:, 4], label="Pos X")
        plt.plot(motion[:, 5], label="Pos Y")
        plt.plot(motion[:, 6], label="Pos Z")
        plt.legend()
        plt.title("Motion Data")
        plt.show()


# 验证模型
def validate(model, dataloader, device, criterion):
    """
    在验证集上评估模型性能
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['video'].to(device)
            targets = batch['motion'].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    val_loss /= len(dataloader)
    accuracy = 100 * correct / total
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return val_loss, accuracy


# 推理函数
def inference(model, dataloader, device):
    """
    用训练好的模型进行推理
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['video'].to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu())

    predictions = torch.cat(predictions, dim=0)
    return predictions


# 结果可视化
def visualize_results(predictions, targets, output_dir="output_results"):
    """
    可视化预测结果和目标值
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, (pred, target) in enumerate(zip(predictions, targets)):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(pred.permute(1, 2, 0).numpy())  # 假设为图像数据
        axes[0].set_title(f"Predicted {idx + 1}")
        axes[0].axis('off')

        axes[1].imshow(target.permute(1, 2, 0).numpy())
        axes[1].set_title(f"Target {idx + 1}")
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"result_{idx + 1}.png"))
        plt.close()
