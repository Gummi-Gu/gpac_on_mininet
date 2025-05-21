import math
import time

import Client.util as util
import Client.Factory  as Factory

class DashInterface:
    def __init__(self):
        self.srd_quantity = None
    def set_quality(self, slice_idx, srd_position=None):
        """
                获取预测的质量值
        """
        return Factory.videoSegmentStatus.get_quality_idx(slice_idx)

