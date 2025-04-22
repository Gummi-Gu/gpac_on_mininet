import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from Client.model.TASEDmodel import TASED_v2  # 假设TASED_v2来自原项目
from scipy.ndimage.filters import gaussian_filter
import torch.nn.functional as F

class TASEDFeatureExtractor(nn.Module):
    """修改原TASED模型作为特征提取器"""

    def __init__(self, original_model):
        super().__init__()
        self.model=original_model
        self.image_dir="data/output"
        self.image_counter=0

    def forward(self, x):
        # 输入形状: (T,C,H, W)
        x=x.permute(0,2,1,3,4)
        # clip.shape = (B, C, T, H, W)
        outputs = []
        for i in range(x.size(0)):
            clip = self.transform(x[i])
            single_clip = clip  # shape: (1, C, T, H, W)
            processed = self.process(single_clip)  # 假设输出形状为 (D,)
            processed = torch.from_numpy(processed)
            outputs.append(processed)

        # 合并成一个新的 batch，outputs 的 shape 就是 (B, D)
        x = torch.stack(outputs, dim=0).to('cuda')
        return x

    def process(self,clip):
        ''' process one clip and save the predicted saliency map '''
        with torch.no_grad():
            smap = self.model(clip.cuda()).cpu().data[0]
        smap = (smap.numpy() * 255.).astype(np.int32) / 255.
        smap = gaussian_filter(smap, sigma=7)
        img= (smap / np.max(smap) * 255.).astype(np.uint8) #(H,W)

        image_filename = os.path.join(self.image_dir, f"frame_{self.image_counter:04d}.jpg")
        self.image_counter += 1

        # 保存当前帧为 JPG 图像
        cv2.imwrite(image_filename, img)
        return img

    def transform(self,snippet):
        ''' stack & noralization '''
        snippet = snippet.mul_(2.).sub_(255).div(255)
        snippet=snippet.view(1,snippet.shape[0],snippet.shape[1],snippet.shape[2],snippet.shape[3]).to('cuda')

        return snippet

class HeadMotionPredictor(nn.Module):
    """完整预测模型"""

    def __init__(self, tased_model, lstm_hidden=256,lstm_hidden2=64):
        super().__init__()
        self.tased = tased_model
        self.compressor = nn.Sequential(
            nn.Linear(16 * 24, 128),  # 假设池化后特征维度16x24=384
            nn.ReLU(),
            nn.Linear(128, 24)
        )
        self.lstm = nn.LSTM(
            input_size=24 + 4,  # 视频特征+运动数据
            hidden_size=lstm_hidden,
            batch_first=True
        )
        self.fc = nn.Linear(lstm_hidden, lstm_hidden2)
        self.fc2 = nn.Linear(lstm_hidden2, 4)

    def forward(self, video, motion):
        # 视频输入: (B, T, 3, H, W)(16,32,3,224,384)
        # 运动数据: (B, T, 4)
        batch_size, _, _ = video.shape[:3]
        T=32

        # 提取视频特征
        vid_feat = self.tased(video)#(16,224,384)
        vid_feat = vid_feat.float()
        # 维度修正与池化
        vid_feat = vid_feat.unsqueeze(1)  # 添加通道维度 [B, 1, 224, 348]
        pooled = F.adaptive_avg_pool2d(vid_feat, (16, 24))  # [B, 1, 16, 24]
        pooled_flat = pooled.view(batch_size, -1)  # [B, 384]

        # 可学习压缩
        compressed = self.compressor(pooled_flat)  # [B, 24]

        # 时间维度扩展
        compressed_seq = compressed.unsqueeze(1).expand(-1, T, -1)  # [B, T, 24]

        # 特征拼接
        combined = torch.cat((compressed_seq, motion), dim=2)  # [B, T, 31]
        # LSTM处理
        lstm_out, _ = self.lstm(combined)  # (B, T, H)

        # 取最后一个时间步并预测
        output = self.fc(lstm_out[:, -1, :])
        output = self.fc2(output)# (B, 4)

        return output,vid_feat


# 初始化流程
def build_model(pretrained=True , weight_dict='Client/model/TASED_updated.pt'):
    # 加载原始TASED模型
    base_model = TASED_v2().to('cuda')
    if pretrained:
        if os.path.isfile(weight_dict):
            print('loading weight file')
            weight_dict = torch.load(weight_dict)
            model_dict = base_model.state_dict()
            for name, param in weight_dict.items():
                if 'module' in name:
                    name = '.'.join(name.split('.')[1:])
                if name in model_dict:
                    if param.size() == model_dict[name].size():
                        model_dict[name].copy_(param)
                    else:
                        print(' size? ' + name, param.size(), model_dict[name].size())
                else:
                    print(' name? ' + name)

            print(' loaded')
        else:
            print('weight file?')

    # 构建特征提取器
    tased_feat = TASEDFeatureExtractor(base_model)

    # 完整模型
    model = HeadMotionPredictor(tased_feat).to('cuda')
    return model