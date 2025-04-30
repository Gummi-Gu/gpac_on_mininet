# 计算焦距
import atexit
import math
import os
import queue
import threading
import time

import pandas as pd

import Client.Factory as Factory

from pynput import keyboard

import cv2
import numpy as np


class Viewpoint:
    def __init__(self,choice=0):
        # 设置输出参数
        self.output_width = 800
        self.output_height = 600
        self.angle_thread = threading.Thread(target=self.listen_for_keys)
        self.angle_thread.daemon = True  # 设置为守护线程，确保程序退出时线程也会退出
        self.angle_thread.start()

        self.choice = choice



    def focal_cal(self, yaw,pitch,fov,equi_width=None, equi_height=None,):
        if equi_width is None:
            equi_width=Factory.width
        if equi_height is None:
            equi_height = Factory.height
        start_time=time.time()
        focal = self.output_width / (2 * np.tan(np.radians(fov / 2)))

        # 生成像素网格
        x, y = np.meshgrid(np.arange(self.output_width), np.arange(self.output_height))
        x_centered = (x - self.output_width / 2) / focal
        y_centered = (y - self.output_height / 2) / focal
        z_cam = np.ones_like(x_centered)

        # 归一化方向向量
        norm = np.sqrt(x_centered ** 2 + y_centered ** 2 + z_cam ** 2)
        x_cam = x_centered / norm
        y_cam = y_centered / norm
        z_cam = z_cam / norm

        # 转换为弧度
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)

        # 构造旋转矩阵（先偏航后俯仰）
        R_y = np.array([
            [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
            [0, 1, 0],
            [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
        ])
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad), np.cos(pitch_rad)]
        ])
        R = np.dot(R_x, R_y)  # 组合旋转矩阵

        # 将相机坐标系转换到世界坐标系
        cam_vectors = np.stack([x_cam, y_cam, z_cam], axis=-1)
        world_vectors = np.dot(cam_vectors, R.T)

        # 转换为球面坐标
        x_world, y_world, z_world = world_vectors[..., 0], world_vectors[..., 1], world_vectors[..., 2]
        theta = np.arctan2(y_world, x_world)  # 经度
        theta = np.where(theta < 0, theta + 2 * np.pi, theta)  # 转换到[0, 2π)
        phi = np.arcsin(z_world)  # 纬度

        # 映射到等距柱状图的UV坐标
        u = theta / (2 * np.pi) * equi_width
        v = (phi + np.pi / 2) / np.pi * equi_height

        # 处理边界
        u = np.clip(u, 0, equi_width - 1)
        v = np.clip(v, 0, equi_height - 1)

        # 生成映射矩阵
        map_x = u.astype(np.float32)
        map_y = v.astype(np.float32)

        x_cam = 0.0
        y_cam = 0.0
        z_cam = 1.0
        cam_vectors = np.stack([x_cam, y_cam, z_cam], axis=-1)
        world_vectors = np.dot(cam_vectors, R.T)
        x_world, y_world, z_world = world_vectors[..., 0], world_vectors[..., 1], world_vectors[..., 2]
        theta = np.arctan2(y_world, x_world)  # 经度
        theta = theta if theta >= 0 else theta + 2 * np.pi  # [0, 2π)
        phi = np.arcsin(z_world)
        u = theta / (2 * np.pi) * equi_width
        v = (phi + np.pi / 2) / np.pi * equi_height
        #print("[Rendering]",time.time()-start_time)

        filename = f"yaw_{int(yaw)}_pitch_{int(pitch)}_fov_{fov}.npz"
        path = os.path.join("Client\precomputed_maps", filename)
        np.savez_compressed(
            path,
            map_x=map_x,
            map_y=map_y,
            u=u,
            v=v
        )
        print(f"[Rendering]Generated {filename}")



        return map_x, map_y,u,v
    def load_maps(self,yaw,pitch,fov):
        """Load precomputed maps from file"""
        filename = f"yaw_{int(yaw)}_pitch_{int(pitch)}_fov_{fov}.npz"
        path = os.path.join("Client\precomputed_maps", filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No precomputed map for {yaw}/{pitch}")
        try:
            data = np.load(path)
        except Exception as e:
            raise e
        return data["map_x"], data["map_y"],data["u"], data["v"]

    def listen_for_keys(self):
        # 启动监听器
        #with keyboard.Listener(on_press=self.on_press) as listener:
        #    listener.join()
        #data = pd.read_csv('./Client/video_0_droped.csv')  # 替换为你的实际路径
        Factory.yaw = -19
        Factory.pitch = 94
        Factory.videoSegmentStatus.set_xyzw(0.078, -0.716, 0.026, 0.693)
        return

        df = pd.read_csv('Client/model/data/train/csv/4/video_4.csv')

        # 保留1位小数并去重
        df['PlaybackTimeRounded'] = df['PlaybackTime'].round(1)
        df_unique = df.drop_duplicates(subset='PlaybackTimeRounded')

        # 删除辅助列并输出
        df_unique = df_unique.drop(columns=['PlaybackTimeRounded'])
        data=df_unique
        # 转换函数
        # 计算 yaw 和 pitch
        for i, row in data.iterrows():
            yaw, pitch,x,y,z,w = self.quaternion_to_yaw_pitch(row['UnitQuaternion.x'],
                                                 row['UnitQuaternion.y'],
                                                 row['UnitQuaternion.z'],
                                                 row['UnitQuaternion.w'])
            Factory.yaw=yaw
            Factory.pitch=pitch
            Factory.videoSegmentStatus.set_xyzw(x,y,z,w)
            print(yaw,pitch,x,y,z,w)
            time.sleep(0.1)
            #print(f"[Rendering] {time.time()}")
            #print(f"[Rendering] yaw:{self.yaw}_pitch:{self.pitch}")

    def quaternion_to_yaw_pitch(self,x, y, z, w):
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        # 防止除以零
        if norm == 0:
            raise ValueError("The quaternion has zero magnitude, cannot normalize.")
        # 归一化四元数
        w /= norm
        x /= norm
        y /= norm
        z /= norm
        # pitch（绕 x 轴）
        pitch = math.asin(2.0 * (w * y - z * x))
        # yaw（绕 y 轴）
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return math.degrees(yaw) + 90, math.degrees(pitch) + 180, x, y, z, w



class Renderer:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=10)  # 可以限制最大缓存帧数
        self.running = False
        self.start()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._render_loop)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def push_frame(self, rgb, title, dur):
        if not self.frame_queue.full():
            self.frame_queue.put((rgb, title, dur))  # 将帧和标题一起放进去

    def _render_loop(self):
        while self.running:
            try:
                rgb, title, dur = self.frame_queue.get(timeout=0.5)
                self.render(rgb, title, dur)
                time.sleep(dur)
            except queue.Empty:
                continue  # 队列空了就继续循环，不会卡死

    def render(self, rgb, title,dur):
        try:
            map_x, map_y, u, v = Factory.viewpoint.load_maps(Factory.yaw, Factory.pitch, Factory.fov)
        except Exception as e:
            map_x, map_y, u, v = Factory.viewpoint.focal_cal(Factory.yaw, Factory.pitch, Factory.fov)
        Factory.u = u
        Factory.v = v
        if Factory.clientname == 'client0':
            output_img = cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR)
            cv2.imshow(Factory.clientname, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
            cv2.setWindowTitle(Factory.clientname, title)
            cv2.waitKey(1)  # 必须有waitKey，不然窗口不刷新


import concurrent.futures

def precompute_all_mappings_threaded(yaw_list, pitch_list, fov_list):
    """
    多线程快速预计算所有(yaw, pitch, fov)组合，返回dict
    """
    mapping_cache = {}

    def compute_mapping(yaw, pitch, fov):
        key = (yaw, pitch, fov)
        try:
            map_x, map_y, u, v = Factory.viewpoint.load_maps(yaw, pitch, fov)
        except FileNotFoundError:
            map_x, map_y, u, v = Factory.viewpoint.focal_cal(yaw, pitch, fov)
        return key, (map_x, map_y, u, v)

    # 把所有组合列出来
    tasks = [(yaw, pitch, fov) for yaw in yaw_list for pitch in pitch_list for fov in fov_list]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_task = {executor.submit(compute_mapping, *task): task for task in tasks}

        for future in concurrent.futures.as_completed(future_to_task):
            key, mapping = future.result()
            mapping_cache[key] = mapping
            print(f"完成: yaw={key[0]}, pitch={key[1]}, fov={key[2]}")

    return mapping_cache

if __name__ == "__main__":
    yaw_list = [x for x in range(-89,179)]
    pitch_list = [x for x in range(0,179)]
    fov_list = [120]
    Factory.Factory_init()
    mapping_cache = precompute_all_mappings_threaded(yaw_list, pitch_list, fov_list)