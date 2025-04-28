import math
import time

import Client.util as util
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

