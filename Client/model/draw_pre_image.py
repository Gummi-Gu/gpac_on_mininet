import numpy as np
import cv2
import Client.Factory as Factory
import matplotlib.pyplot as plt
import Client.util



def draw_gradient_circle(center, radius, shape):
    """返回一个带有渐变圆的 RGBA 图像（黑底）"""
    h, w = shape
    y, x = np.ogrid[:h, :w]
    cy, cx = center
    mask = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    alpha = np.clip(1 - (mask / radius), 0, 1)
    alpha = (alpha ** 2) * 255  # 中心亮、边缘透明

    circle = np.zeros((h, w, 4), dtype=np.uint8)
    circle[..., :3] = 255  # 白色
    circle[..., 3] = alpha.astype(np.uint8)
    return circle

def alpha_blend(base, overlay):
    """将 overlay 的 RGBA 叠加到 base（RGB）上"""
    alpha = overlay[..., 3:] / 255.0  # Normalize alpha to [0, 1]
    base_rgb = base.astype(np.float32)
    overlay_rgb = overlay[..., :3].astype(np.float32)
    blended = base_rgb * (1 - alpha) + overlay_rgb * alpha
    return blended.astype(np.uint8)


def pre_rgb(rgb, u, v, preu, prev):
    width = Factory.width//8
    height = Factory.height//8
    u//=8
    v//=8
    preu//=8
    prev//=8
    radius = height

    # 1. resize 原始图像
    rgb_resized = cv2.resize(rgb, (width, height))

    # 2. 黑底 + u,v 圆
    black_bg = np.zeros((height, width, 3), dtype=np.uint8)
    circle_uv = draw_gradient_circle((int(v), int(u)), radius, (height, width))
    image_u = alpha_blend(black_bg, circle_uv)

    # 3. 黑底 + preu, prev 圆
    circle_prev = draw_gradient_circle((int(prev), int(preu)), radius, (height, width))
    image_pre = alpha_blend(black_bg, circle_prev)

    return rgb_resized, image_u, image_pre

def compute_opacity_heatmap(str,rgb,level_num):
    # 用cv2读取图像（带alpha通道，cv2读取顺序是BGRA）
    img = rgb

    if len(img.shape) == 3:
        if img.shape[2] == 4:
            b, g, r, a = cv2.split(img)
            gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
            alpha = a
        elif img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            alpha = np.ones_like(gray, dtype=np.uint8) * 255
    elif len(img.shape) == 2:
        gray = img
        alpha = np.ones_like(gray, dtype=np.uint8) * 255
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    # 计算“可见灰度”分数
    opacity = gray * (alpha / 255.0)

    h, w = opacity.shape
    part_h = h // Factory.tile_size
    part_w = w // Factory.tile_size

    heatmap = np.zeros((Factory.tile_size, Factory.tile_size))
    for i in range(Factory.tile_size):
        for j in range(Factory.tile_size):
            block = opacity[i*part_h:(i+1)*part_h, j*part_w:(j+1)*part_w]
            heatmap[i, j] = np.sum(block)
    # 扁平化再进行分级
    flat = heatmap.flatten()
    min_val = flat.min()
    max_val = flat.max()

    if max_val == min_val:
        # 所有值都一样，统一分配为中等等级
        levels=np.full_like(heatmap, level_num // 2 )
    else:
        # 标准化到0~1之间
        norm = (flat - min_val) / (max_val - min_val)
        # 按比例映射到等级（1~level_num）
        levels = np.around(norm * (level_num-1),0).astype(np.uint8)
    #levels[levels == 0] = 1  # 确保最小等级为1
    levels=levels.reshape(heatmap.shape)

    plt.figure(figsize=(Factory.tile_size, Factory.tile_size))
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Opacity Score')
    plt.title("Opacity Heatmap "+str)
    plt.xticks(range(Factory.tile_size))
    plt.yticks(range(Factory.tile_size))
    plt.gca().invert_yaxis()
    plt.show()


    return levels

def get_qualitys(rgb,u,v,preu,prev):
    #rgb = cv2.imread("output/video/0001.png")  # 读取原图
    frame1, frame2, frame3 = pre_rgb(rgb, u,v,preu,prev)
    level1=compute_opacity_heatmap("resized_rgb",frame1,Factory.level_num)
    level2=compute_opacity_heatmap("u_circle",frame2,Factory.level_num)
    level3=compute_opacity_heatmap("prev_circle",frame3,Factory.level_num)
    # 逐元素比较并取最大值
    merged_level = np.maximum(level1, np.maximum(level2, level3))
    return merged_level,level1,level2,level3


