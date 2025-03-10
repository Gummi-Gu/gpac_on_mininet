# 计算焦距
import atexit
import os
import threading
import time
import Factory

from pynput import keyboard

import cv2
import numpy as np


class Viewpoint:
    def __init__(self,choice=0):
        # 设置输出参数
        self.v = None
        self.u = None
        self.output_width = 800
        self.output_height = 600
        self.fov = 150  # 视场角（单位：度）
        # 初始视角参数
        self.yaw = 0.0  # 偏航角
        self.pitch = 90  # 俯仰角

        self.angle_thread = threading.Thread(target=self.listen_for_keys)
        self.angle_thread.daemon = True  # 设置为守护线程，确保程序退出时线程也会退出
        self.angle_thread.start()

        self.choice = choice



    def focal_cal(self, equi_width=Factory, equi_height=2048):
        start_time=time.time()
        focal = self.output_width / (2 * np.tan(np.radians(self.fov / 2)))

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
        yaw_rad = np.radians(self.yaw)
        pitch_rad = np.radians(self.pitch)

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

        filename = f"yaw_{int(self.yaw)}_pitch_{int(self.pitch)}_fov_{self.fov}.npz"
        path = os.path.join("Client\precomputed_maps", filename)
        np.savez_compressed(
            path,
            map_x=map_x,
            map_y=map_y,
            u=u,
            v=v
        )
        print(f"Generated {filename}")



        return map_x, map_y,u,v
    def load_maps(self):
        """Load precomputed maps from file"""
        filename = f"yaw_{int(self.yaw)}_pitch_{int(self.pitch)}_fov_{self.fov}.npz"
        path = os.path.join("Client\precomputed_maps", filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No precomputed map for {self.yaw}/{self.pitch}")

        data = np.load(path)
        return data["map_x"], data["map_y"],data["u"], data["v"]

    def get_view_position(self):
        return self.u, self.v

    def set_view_position(self, x, y):
        self.u, self.v = x,y

    def on_press(self, key):
        print(self.choice)
        if self.choice == 0:
            try:
                if key.char == 'q':
                    print("Exiting...")
                    return False  # 停止监听
                elif key.char == 'a':  # 左转
                    self.yaw -= 15
                elif key.char == 'd':  # 右转
                    self.yaw += 15
                elif key.char == 'w':  # 上仰
                    self.pitch += 15
                elif key.char == 's':  # 下俯
                    self.pitch -= 15
                #print(f"[Viewpoint]Yaw: {self.yaw}, Pitch: {self.pitch}")
                self.yaw= self.yaw% 360
                self.pitch= self.pitch % 360
            except AttributeError:
                # 处理其他特殊键（例如 shift、alt 等）
                pass
        elif self.choice == 1:
            try:
                if key.char == 'q':
                    print("Exiting...")
                    return False  # 停止监听
                elif key.char == 'j':  # 左转
                    self.yaw -= 15
                elif key.char == 'l':  # 右转
                    self.yaw += 15
                elif key.char == 'i':  # 上仰
                    self.pitch += 15
                elif key.char == 'k':  # 下俯
                    self.pitch -= 15
                #print(f"[Viewpoint]Yaw: {self.yaw}, Pitch: {self.pitch}")
                self.yaw= self.yaw% 360
                self.pitch= self.pitch % 360
            except AttributeError:
                # 处理其他特殊键（例如 shift、alt 等）
                pass
    def listen_for_keys(self):
        # 启动监听器
        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()

class Render:

    def render(self,rgb,title):
        #equi_height, equi_width = rgb.shape[:2]
        try:
            map_x,map_y,u,v= Factory.viewpoint.load_maps()
        except FileNotFoundError:
            map_x,map_y,u,v= Factory.viewpoint.focal_cal()
        Factory.viewpoint.set_view_position(u,v)
        # 重映射图像
        output_img = cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR)
        #resize_factor = 0.625
        #small_rgb = cv2.resize(rgb, (int(equi_width * resize_factor), int(equi_height * resize_factor)))
        cv2.imshow('360 View', cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
        cv2.setWindowTitle('360 View', title)