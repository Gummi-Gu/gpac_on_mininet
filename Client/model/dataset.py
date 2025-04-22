# dataset.py
import os
import sys

import cv2
import numpy as np
import psutil
import torch
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset
from typing import Tuple, List, Any
import gc

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
caps=dict()
process = psutil.Process()


class VRDataset(Dataset):
    def __init__(self,
                 video_root: str,
                 motion_csv_root: str,
                 temporal_length: int = 32,
                 target_size: Tuple[int, int] = (224, 384)):  # H x W
        """
        参数说明:
        video_root: 视频文件目录，包含.mp4文件
        motion_csv_root: 运动数据目录，与视频同名的.csv文件
        temporal_length: 时间序列长度 (默认32帧)
        target_size: 目标尺寸 (高度, 宽度) = (224, 384)
        """
        self.video_root = video_root
        self.motion_csv_root = motion_csv_root
        self.temporal_len = temporal_length
        self.target_height, self.target_width = target_size
        self.max=5000
        # 构建样本列表并验证文件存在性
        self.samples = self._validate_samples()

        # 统计信息
        print(f"成功加载 {len(self.samples)} 个样本")

    def _validate_samples(self) -> list[tuple[list[Tensor], list[Any], int]]:
        """验证视频与运动数据文件对应关系（一个视频对应多个人物，每个人物有多个视频数据）"""
        valid_samples = []

        # 遍历人物编号文件夹
        for person_id in os.listdir(self.motion_csv_root):
            person_csv_dir = os.path.join(self.motion_csv_root, person_id)

            if not os.path.isdir(person_csv_dir):
                continue  # 如果不是文件夹则跳过

            # 遍历该人物对应的每个视频文件夹
            for video_fname in os.listdir(self.video_root):
                if video_fname.endswith('.mp4'):
                    video_path = os.path.join(self.video_root, video_fname)
                    # 获取该人物对应的视频文件夹
                    person_video_csv_path = os.path.join(person_csv_dir, video_fname.replace('.mp4', '.csv'))

                    if os.path.exists(person_video_csv_path):
                        valid_samples.append((video_path, person_video_csv_path))
                    else:
                        print(
                            f"警告: 找不到视频 {video_path} 对应的人物 {person_id} 的 CSV 文件 {person_video_csv_path}")
        aligned_video_frames = []
        aligned_motion_data = []
        valid_samples=sorted(valid_samples)
        for video_path, person_video_csv_path in valid_samples:
            print(f'processing + {video_path} + {person_video_csv_path}')
            video_frames = self._load_and_process_video(video_path)  # [T, H, W, C]
            motion_data = self._load_and_process_motion(person_video_csv_path)
            video_frames_count = len(video_frames)
            previous_frame = -1  # 用于检测重复帧
            for row in motion_data:
                motion_frame = int(row[0] * 29.97)
                # 如果运动数据的帧超过了视频帧数，跳出
                if motion_frame >= video_frames_count:
                    break

                # 对应帧的视频帧
                video_frame = video_frames[motion_frame]

                # 如果是重复帧，复制上一帧的视频数据
                if motion_frame == previous_frame:
                    continue
                else:
                    aligned_video_frames.append(video_frame)

                # 将对应的运动数据添加到对齐数据中
                aligned_motion_data.append(row[1:])
                #print(row[1:])
                # 更新上一帧
                previous_frame = motion_frame

        total=len(aligned_video_frames)
        #aligned_video_frames = torch.stack(aligned_video_frames, dim=0)
        #aligned_motion_data= torch.stack(aligned_motion_data, dim=0)
        frame_32_sample=[]
        for idx,_ in enumerate(aligned_video_frames):
            if idx<total-32 and idx>320:
                frame_32=(aligned_video_frames, aligned_motion_data,idx)
                frame_32_sample.append(frame_32)

        return frame_32_sample

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """返回一个样本字典，包含：
        - video: uint8 tensor [T, C, H, W] (BGR顺序)
        - motion: float32 tensor [T, 4]
        - target: float32 tensor [T, 4] (标准化后的四元数)
        """
        #aligned_video, aligned_motion = self.samples
        # 截取时间窗口
        #temp=self._sample_temporal_window(aligned_video, aligned_motion,idx)
        aligned_video_frames, aligned_motion_data, idx=self.samples[idx]
        return self._sample_temporal_window(aligned_video_frames, aligned_motion_data, idx)


    def _load_and_process_video(self, path: str) -> torch.Tensor:
        """加载视频并预处理为BGR顺序的numpy数组"""
        idx=0
        if path in caps.keys():
            frames = caps[path]
        else:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise IOError(f"无法打开视频文件 {path}")
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # 调整尺寸并保持BGR顺序
                resized = cv2.resize(frame, (self.target_width, self.target_height))
                frames.append(resized)
                if idx> self.max:
                    break
                idx += 1
            cap.release()
            frames=torch.from_numpy(np.stack(frames))
            caps[path] = frames


        return frames  # [T, H, W, C]

    def _load_and_process_motion(self, path: str) -> torch.Tensor:
        """加载并标准化运动数据"""
        df = pd.read_csv(path)
        # 保留需要的列
        required_cols = ['PlaybackTime', 'UnitQuaternion.x', 'UnitQuaternion.y', 'UnitQuaternion.z', 'UnitQuaternion.w']
        df = df[required_cols]

        # 按 PlaybackTime 升序排序
        df = df.sort_values(by='PlaybackTime').reset_index(drop=True)

        # 数据校验
        if not set(required_cols).issubset(df.columns):
            missing = set(required_cols) - set(df.columns)
            raise ValueError(f"CSV文件缺少必要列: {missing}")

        data = df[required_cols].values.astype(np.float32)

        return torch.from_numpy(data)

    def _sample_temporal_window(self, video, motion, idx) -> dict:
        # 确保视频和运动数据是张量类型

        # 计算时间窗口的结束位置
        end = idx + self.temporal_len

        # 取视频和运动数据
        sampled_video = video[idx:end]  # [temporal_length, C, H, W]
        sampled_motion = motion[idx:end]  # [temporal_length, 4]
        sampled_video =torch.stack(sampled_video, dim=0)
        sampled_motion = torch.stack(sampled_motion, dim=0)
        # 计算目标四元数，取后32位的前4位
        target_motion = motion[(idx + self.temporal_len) : (idx + self.temporal_len + self.temporal_len)]
        target_motion=torch.stack(target_motion, dim=0)
        target_motion=target_motion[:,:4]
        target = target_motion.mean(dim=0)  # 平均值作为目标四元数

        return {
            'video': sampled_video.permute(0,3,1,2),  # 形状: [temporal_length, C, H, W]
            'motion': sampled_motion,  # 形状: [temporal_length, 4]
            'target': target  # 形状: [4]，四元数
        }


if __name__ == '__main__':
    # 单元测试
    test_set = VRDataset(
        video_root='./data/video',
        motion_csv_root='./data/csv',
        #temporal_length=32
    )

    sample = test_set[0]
    print("\n样本验证:")
    print(f"视频张量形状: {sample['video'].shape} | 类型: {sample['video'].dtype}")
    print(f"运动张量形状: {sample['motion'].shape} | 类型: {sample['motion'].dtype}")
    print(f"目标张量形状: {sample['target'].shape}")

    # 可视化检查
    import matplotlib.pyplot as plt

    frame = sample['video'][0].permute(1,2,0).numpy()  # 转为HWC
    plt.title(f"BGR通道验证 (B:{frame[0, 0, 0]} G:{frame[0, 0, 1]} R:{frame[0, 0, 2]})")
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.show()