import math
import pyautogui
import Factory


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
        #view_x, view_y = pyautogui.position()
        #view_x=view_x*1.6
        #view_y=view_y*1.6
        view_x, view_y = Factory.viewpoint.get_view_position()
        print(f"[Dash_Interface]视点在原图中的位置：({view_x}, {view_y})")
        if view_x is None or view_y is None:
            return 0
        if slice_idx != 0:
            return self.srd_quantity[slice_idx]
        n = len(srd_position)  # 切片总数
        self.srd_quantity=[2 for i in range(n)]
        rows = cols = int(math.sqrt(n))  # 行列数（取根号向下取整）

        # 遍历所有切片，寻找鼠标所在的切片
        for idx, target_slice in enumerate(srd_position):
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
                valid_adjacent_indexes = [i for i in adjacent_indexes if 0 <= i < len(srd_position)]

                # 判断是否为相邻切片
                for i in valid_adjacent_indexes:
                    self.srd_quantity[i] = 2
        return 0
