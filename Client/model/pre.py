import math
import threading
import time
from time import sleep

import numpy as np
import torch
from sympy.polys.polyconfig import query

import Client.model.draw_pre_image as pred
import Client.Factory as Factory

from Client.model.model import build_model

config = {
    'batch_size': 16,
    'num_epochs': 2,
    'lr': 1e-5,
    'lr_step_size': 10,
    'lr_gamma': 0.1,
    'weight_decay': 1e-5,
    'save_dir': './checkpoints',
    'log_interval': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'val': True
}


last_rebuff_time=0
last_rebuff_count=0
last_Qoa=None

model = build_model(pretrained=True).to(config['device'])
model.load_state_dict(torch.load('Client/model/checkpoints/best_model_20250414_143127.pt')['state_dict'])
model.eval()  # 设置为评估模式

def evenly_expand_list(original_list, target_length):
    """
    将原始列表均匀复制填满到目标长度。

    参数:
        original_list (list): 原始列表
        target_length (int): 目标长度，必须 >= len(original_list)

    返回:
        list: 均匀扩展后的新列表
    """
    n = len(original_list)
    if target_length < n:
        return original_list
    base_repeat = target_length // n
    extra = target_length % n

    extended = []
    for i, item in enumerate(original_list):
        repeat = base_repeat + (1 if i < extra else 0)
        extended.extend([item] * repeat)

    return extended

def start():
    #print()
    thread = threading.Thread(target=pre)
    thread.daemon = True  # 设置为守护线程，确保程序退出时线程也会退出
    thread.start()

def pre():
    time.sleep(1)
    while True:
        time.sleep(1)  # 模拟一秒一个输入数据，可以去掉或改为接收信号的时间间隔
        # 模拟接收一帧实时数据（10维向量）
        video_data=Factory.videoSegmentStatus.get_video()
        try:
            video_data = evenly_expand_list(video_data, 32)
        except Exception as e:
            continue
        tensor_list = [torch.tensor(ndarray) for ndarray in video_data]
        tensor_stack = torch.stack(tensor_list)
        video_data = tensor_stack.unsqueeze(0).permute(0,1,4,2,3)

        motion_data=Factory.videoSegmentStatus.get_motion()
        try:
            motion_data=evenly_expand_list(motion_data, 32)
        except Exception as e:
            continue
        tensor_list = [torch.tensor(ndarray) for ndarray in motion_data]
        tensor_stack = torch.stack(tensor_list)
        motion_data = tensor_stack.unsqueeze(0)
        # 模型推理
        with torch.no_grad():
            output = model(video_data.to('cuda',dtype=torch.float32),motion_data.to('cuda',dtype=torch.float32))
            rgb=output[1][0].permute(1,2,0).cpu().numpy()
            prediction = output[0][0].tolist()
            yaw,pitch,_,_,_,_=Factory.viewpoint.quaternion_to_yaw_pitch(prediction[0], prediction[1], prediction[2], prediction[3])
            try:
                map_x, map_y, u, v = Factory.viewpoint.load_maps(yaw, pitch, Factory.fov)
            except Exception as e:
                map_x, map_y, u, v = Factory.viewpoint.focal_cal(yaw, pitch, Factory.fov)
            Factory.preu = u
            Factory.prev = v
            merged_level,level1,level2,level3=pred.get_qualitys(rgb,Factory.u,Factory.v,Factory.preu,Factory.prev)
            Factory.videoSegmentStatus.Qoe=QoEpre(level2)
            merged_level=np.insert(merged_level.flatten(), 0, 0).tolist()
            #print(f"[PRE] {merged_level}")
            Factory.videoSegmentStatus.set_all_quality_tiled(merged_level)




def QoEpre(level):
    global last_rebuff_time, last_rebuff_count, last_Qoa
    qua=Factory.videoSegmentStatus.get_quality()[1:]
    level=level.flatten()
    B=0
    for j,i in enumerate(qua):
        if i==0:
            B+=1*level[j]
        if i==1:
            B+=2*level[j]
        if i==2:
            B+=3*level[j]
        if i == 3:
            B += 4 * level[j]
    B=B/Factory.tile_num*10

    matrix1,matrix2=qua,last_Qoa
    if last_Qoa is None:
        matrix2=matrix1
    bitrate_map = Factory.videoSegmentStatus.bitrate_map

    # 将矩阵元素转换为实际比特率
    matrix1_bitrate = np.vectorize(bitrate_map.get)(matrix1)
    matrix2_bitrate = np.vectorize(bitrate_map.get)(matrix2)
    # 计算两个矩阵之间的元素差异（以比特率为单位）
    diff_matrix = np.abs(matrix1_bitrate - matrix2_bitrate)
    S = (diff_matrix - np.min(diff_matrix)) / max(0.1,(np.max(diff_matrix) - np.min(diff_matrix)))
    S = np.sum(S)

    D1=Factory.videoSegmentStatus.get_rebuff_time()-last_rebuff_time
    D2=Factory.videoSegmentStatus.get_rebuff_count()-last_rebuff_count
    #print(f"[PRE] {D1},{D2}")

    matrix=qua
    matrix=np.array(matrix).reshape((Factory.tile_size, Factory.tile_size))
    # 初始化差异矩阵
    diff_matrix = np.zeros(matrix.shape)
    # 遍历矩阵中的每个元素
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            diffs = []
            # 上邻居
            if i > 0:
                diffs.append(abs(matrix[i, j] - matrix[i - 1, j]))
            # 下邻居
            if i < matrix.shape[0] - 1:
                diffs.append(abs(matrix[i, j] - matrix[i + 1, j]))
            # 左邻居
            if j > 0:
                diffs.append(abs(matrix[i, j] - matrix[i, j - 1]))
            # 右邻居
            if j < matrix.shape[1] - 1:
                diffs.append(abs(matrix[i, j] - matrix[i, j + 1]))
            # 计算当前元素的相邻差异
            diff_matrix[i, j] = np.mean(diffs)
    # 对差异矩阵进行 Min-Max 归一化
    diff_matrix_normalized = (diff_matrix - np.min(diff_matrix)) / max(0.1,(np.max(diff_matrix) - np.min(diff_matrix)))
    U=np.sum(diff_matrix_normalized/4)

    Qoe = -1 * D1 - 1 * D2 + 1 * B - 2 * S - 1 * U

    last_Qoa=qua
    last_rebuff_time=Factory.videoSegmentStatus.get_rebuff_time()
    last_rebuff_count=Factory.videoSegmentStatus.get_rebuff_count()
    print(f"[PRE]{Qoe}")
    return Qoe


