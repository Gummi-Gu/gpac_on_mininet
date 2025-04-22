import math
import time

import util
import Client.Factory  as Factory

class DashInterface:
    def __init__(self):
        self.srd_quantity = None
    def set_quality(self, slice_idx, srd_position=None):
        """
                根据鼠标位置设置切片的质量
                :param slice_idx: 给定的切片序号（从0开始）
                :param view_x: 鼠标的x坐标
                :param view_y: 鼠标的y坐标
                :return: 切片的质量
        """
        return Factory.videoSegmentStatus.get_quality_idx(slice_idx)
    def set_quality_old(self, slice_idx, srd_position=None):
        """
        根据鼠标位置设置切片的质量
        :param slice_idx: 给定的切片序号（从0开始）
        :param view_x: 鼠标的x坐标
        :param view_y: 鼠标的y坐标
        :return: 切片的质量
        """
        start_time=time.time()
        #view_x, view_y = pyautogui.position()
        #view_x=view_x*1.6
        #view_y=view_y*1.6
        view_x, view_y = Factory.u,Factory.v
        #print(f"[Dash_Interface]视点在原图中的位置：({view_x}, {view_y})")
        if view_x is None or view_y is None:
            return 3
        if slice_idx != 0:
            if self.srd_quantity[slice_idx] == 0:
                return 3
            else:
                return self.srd_quantity[slice_idx]
        n = len(srd_position)  # 切片总数
        self.srd_quantity = [1 for _ in range(n)]

        rows = cols = int(math.sqrt(n))  # 假设是一个正方形网格

        # 计算每个切片的行列信息
        slice_grid = []
        for idx, target_slice in enumerate(srd_position):
            row, col = divmod(idx-1, cols)
            slice_grid.append((row, col, target_slice))
        # 查找视点所在的切片
        target_row, target_col = None, None
        for row, col, target_slice in slice_grid:
            x, y, w, h = target_slice.x, target_slice.y, target_slice.w, target_slice.h
            if x <= view_x <= x + w and y <= view_y <= y + h:
                target_row, target_col = row, col
                break
        if target_row is None or target_col is None:
            return 0  # 视点未命中任何切片

        target_idx = target_row * cols + target_col +1
        self.srd_quantity[target_idx] = 3  # 视点所在切片设为最高质量

        # 计算周期性索引
        def wrap_index(r, c):
            return ((r % rows) * cols) + (c % cols) + 1

        # 上下左右相邻块（质量2）
        adjacent_positions = [
            (target_row - 1, target_col),  # 上
            (target_row + 1, target_col),  # 下
            (target_row, target_col - 1),  # 左
            (target_row, target_col + 1)  # 右
        ]

        # 斜向相邻块（质量1）
        diagonal_positions = [
            (target_row - 1, target_col - 1),  # 左上
            (target_row - 1, target_col + 1),  # 右上
            (target_row + 1, target_col - 1),  # 左下
            (target_row + 1, target_col + 1)  # 右下
        ]

        for r, c in adjacent_positions:
            self.srd_quantity[wrap_index(r, c)] = 2

        for r, c in diagonal_positions:
            self.srd_quantity[wrap_index(r, c)] = 1

        return 0

