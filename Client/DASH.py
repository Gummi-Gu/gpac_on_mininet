import math
import time

import requests

import Factory


class MyCustomDASHAlgo:
    #get notifications when a DASH period starts or ends
    def __init__(self):
        self.srd_position=[]
        self.srd_quantity=[]

    def on_period_reset(self, type):
        print('period reset type ' + str(type))

    #get notification when a new group (i.e., set of adaptable qualities, `AdaptationSet` in DASH) is created. Some groups may be left unexposed by the DASH client
    #the qualities are sorted from min bandwidth/quality to max bandwidth/quality
    def on_new_group(self, group):
        #print('new group ' + str(group.idx) + ' qualities ' + str(len(group.qualities)) + ' codec ' + group.qualities[0].codec);
        srd=group.SRD
        #print(f'{srd.x} {srd.y}')
        self.srd_position.append(srd)

    #perform adaptation logic - return value is the new quality index, or -1 to keep as current, -2 to discard  (debug, segments won't be fetched/decoded)
    def on_rate_adaptation(self, group, base_group, force_low_complexity, stats):
        #print('We are adapting on group ' + str(group.idx) )
        # print('' + str(stats))
        # Perform adaptation, check group.SRD to perform spatial adaptation, ...
        #
        # In this example we simply cycle through qualities
        # Send the newq value via GET request to 127.0.0.1:12567/dash
        x, y = Factory.viewpoint.get_view_position()
        if x is None or y is None:
            return 0
        select_num=self.set_quality(group.idx,x,y)
        #print(f"[DASH]{stats.buffer} {stats.buffer_min} {stats.buffer_max} {stats.download_rate}")
        print(f'[DASH]For {group.idx} We choose {select_num}')
        Factory.quantitycollector.collect_data(timestamp=time.time(),resolution=group.qualities.height,frame_rate_std=group.qualities.fps,
                                               real_time_bandwidth=stats.download_rate,buffer_size=stats.buffer,)
        return select_num
        try:
            response = requests.get(f'http://127.0.0.1:12567/dash', params={'value': newq})
            if response.status_code == 200:
                select_num = response.json().get('value', 0)
            else:
                print(f"Failed to send. Status code: {response.status_code}")
        except requests.RequestException as e:
            print(f"Error sending request: {e}")
        return select_num

    # this callback is optional, use it only if your algo may abort a running transfer (this can be very costly as it will require closing and reopening the HTTP connection for HTTP 1.1  )
    #   -1 to continue download
    #   or -2 to abort download but without retrying to download a segment at lower quality for the same media time
    #   or the index of the new quality to download for the same media time
    def on_download_monitor(self, group, stats):
        print('download monitor group ' + str(group.idx) + ' stats ' + str(stats) );
        return -1

    def set_quality(self, slice_idx, view_x=50, view_y=50):
        """
        根据鼠标位置设置切片的质量
        :param slice_idx: 给定的切片序号（从0开始）
        :param view_x: 鼠标的x坐标
        :param view_y: 鼠标的y坐标
        :return: 切片的质量（4, 3, -1）
        """
        if slice_idx != 0:
            return self.srd_quantity[slice_idx]
        n = len(self.srd_position)  # 切片总数
        self.srd_quantity=[2 for i in range(n)]
        rows = cols = int(math.sqrt(n))  # 行列数（取根号向下取整）

        # 遍历所有切片，寻找鼠标所在的切片
        for idx, target_slice in enumerate(self.srd_position):
            x, y, w, h = target_slice.x, target_slice.y, target_slice.w, target_slice.h

            # 判断鼠标是否在当前切片内
            if x <= view_x <= x + w and y <= view_y <= y + h:
                #print(f'x = {mouse_x} y = {mouse_y} the mouse_tile_idx is {idx}')

                self.srd_quantity[idx] = 3   # 鼠标在目标切片内，返回质量3

                # 计算周围切片的索引范围
                row, col = divmod(idx-1, cols)  # 当前切片的行列位置

                # 上下左右相邻切片的序号
                adjacent_indexes = []
                if row > 0:  # 上方
                    adjacent_indexes.append(idx - cols)
                if row < rows - 1:  # 下方
                    adjacent_indexes.append(idx + cols)
                if col > 0:  # 左方
                    adjacent_indexes.append(idx - 1)
                if col < cols - 1:  # 右方
                    adjacent_indexes.append(idx + 1)

                # 对角线相邻的切片
                if row > 0 and col > 0:  # 左上
                    adjacent_indexes.append(idx - cols - 1)
                if row > 0 and col < cols - 1:  # 右上
                    adjacent_indexes.append(idx - cols + 1)
                if row < rows - 1 and col > 0:  # 左下
                    adjacent_indexes.append(idx + cols - 1)
                if row < rows - 1 and col < cols - 1:  # 右下
                    adjacent_indexes.append(idx + cols + 1)

                # 限制切片索引的范围
                valid_adjacent_indexes = [i for i in adjacent_indexes if 0 <= i < len(self.srd_position)]

                # 判断是否为相邻切片
                for i in valid_adjacent_indexes:
                    self.srd_quantity[i] = 2
        return 0
