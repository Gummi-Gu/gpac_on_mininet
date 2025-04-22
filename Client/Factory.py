import atexit
import csv
from datetime import datetime
import math
import os
import threading
import time

import cv2
import numpy as np

import BufferFilter
import DASH
import Rendering
import Interfaces
import util


width=4096#2560#3840
height=2048#1440#1920
tile_num=10
tile_size=3
level_num=4
bitrate_map = {0:200, 1: 785, 2: 3150, 3: 12600}
Winname=None
press_start=True
dash=None
viewpoint=None
fs=None
bufferFilter=None
dash_interface=None
render=None
videoSegmentStatus=None
fov,yaw,pitch,u,v,preu,prev=120,0,0,0,0,0,0
Qoe=0
pre_qua=[]
streamingMonitorClient=util.StreamingMonitorClient('http://192.168.3.22:5000')
class VideoSegmentStatus:
    def __init__(self, tile_num, log_dir="logs"):
        self.x=0
        self.y=0
        self.z=0
        self.w=0
        self.rgb = None
        self.rgb_resized = None
        self.rebuff_time = 0
        self.rebuff_count = 0
        self.quality_tiled = [0 for _ in range(tile_num)]
        self.tile_num = tile_num
        self.videos = []
        self.motions = []


        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp_prefix = timestamp
        self.quer_filename = os.path.join(log_dir, f"quer_{timestamp}.csv")
        self.rebuff_filename = os.path.join(log_dir, f"rebuff_quality_{timestamp}.csv")
        self.image_dir = os.path.join(log_dir, f"images_{timestamp}")
        os.makedirs(self.image_dir, exist_ok=True)
        self.image_counter = 0
        '''
        with open(self.quer_filename, mode="w", newline='') as f:
            writer = csv.writer(f)
            headers = ["timestamp", "x", "y", "z", "w"]
            writer.writerow(headers)

        with open(self.rebuff_filename, mode="w", newline='') as f:
            writer = csv.writer(f)
            headers = ["timestamp", "rebuff_time", "rebuff_count"] + [f"tile_{i}" for i in range(tile_num)]
            writer.writerow(headers)
        # 启动线程
        self.quer_thread = threading.Thread(target=self.log_yaw_pitch_periodically)
        self.rebuff_thread = threading.Thread(target=self.log_rebuff_quality_periodically)
        self.quer_thread.daemon = True
        self.rebuff_thread.daemon = True
        self.quer_thread.start()
        self.rebuff_thread.start()
        '''

    # -------------------- 状态设置函数 --------------------

    def set_xyzw(self, x,y,z,w):
        """
        设置 yaw 角度（单位：度），并转换为弧度
        """
        self.x=x
        self.y=y
        self.z=z
        self.w=w
        self.motions.append([self.x, self.y, self.z, self.w])
        if len(self.motions) > 10:
            self.motions.pop(0)



    def set_rebuff_time(self, value): self.rebuff_time = value
    def set_rebuff_count(self, value): self.rebuff_count = value

    def set_quality_tiled(self, index, quality):
        if 0 <= index < self.tile_num:
            self.quality_tiled[index] = quality
        else:
            raise IndexError("Invalid tile index")

    def set_all_quality_tiled(self, quality_list):
        if len(quality_list) != self.tile_num:
            raise ValueError("Length mismatch")
        self.quality_tiled = quality_list
        streamingMonitorClient.submit_chunk_qualities(self.quality_tiled)


    def set_rgb(self, rgb_image):
        """
        输入应为 numpy.ndarray 格式的 RGB 图像 (H, W, 3)
        """
        if not isinstance(rgb_image, np.ndarray):
            raise ValueError("rgb_image must be numpy.ndarray")
        self.rgb = rgb_image
        self.rgb_resized = cv2.resize(rgb_image, (348, 224))  # (W, H)

        # 构造文件路径
        image_filename = os.path.join(self.image_dir, f"frame_{self.image_counter:04d}.jpg")
        self.image_counter += 1
        self.videos.append(self.rgb_resized)
        if len(self.videos) > 32:
            self.videos.pop(0)
        # 保存当前帧为 JPG 图像
        cv2.imwrite(image_filename, cv2.cvtColor(self.rgb_resized, cv2.COLOR_RGB2BGR))
    # -------------------- 日志记录 --------------------

    def log_yaw_pitch(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        row = [now, self.x,self.y,self.z,self.w]
        with open(self.quer_filename, mode="a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def log_rebuff_quality(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        row = [now, self.rebuff_time, self.rebuff_count] + self.quality_tiled

        with open(self.rebuff_filename, mode="a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    def get_video(self):
        return self.videos
    def get_motion(self):
        return self.motions
    def get_quality(self):
        return self.quality_tiled
    def get_quality_idx(self,idx):
        return self.quality_tiled[idx]
    def get_rebuff_time(self):
        return self.rebuff_time
    def get_rebuff_count(self):
        return self.rebuff_count


    # -------------------- 定时记录 --------------------

    def log_yaw_pitch_periodically(self):
        last_log_time = time.time()
        while True:
            # 每 0.1 秒记录一次 yaw 和 pitch
            if time.time() - last_log_time >= 0.1:
                self.log_yaw_pitch()
                last_log_time = time.time()
            time.sleep(0.01)  # 稍微等待，避免占用过多 CPU

    def log_rebuff_quality_periodically(self):
        last_log_time = time.time()
        while True:
            # 每 1 秒记录一次 rebuff_time, rebuff_count 和 quality_tiled
            if time.time() - last_log_time >= 0.5:
                self.log_rebuff_quality()
                last_log_time = time.time()
            time.sleep(0.1)  # 稍微等待，避免占用过多 CPU



videoSegmentStatus=VideoSegmentStatus(tile_num,'./Client/logs')
dash= DASH.MyCustomDASHAlgo()
viewpoint= Rendering.Viewpoint(press_start)
fs = BufferFilter.MyFilterSession()
bufferFilter= BufferFilter.MyFilter(fs)
render= Rendering.Render()
dash_interface=Interfaces.DashInterface()