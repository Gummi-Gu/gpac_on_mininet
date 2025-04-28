import os
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def render_images(image_directory):
    last_image_time = 0
    fig, ax = plt.subplots()  # 创建一个图形和子图对象

    # 获取 matplotlib 的窗口句柄
    manager = plt.get_current_fig_manager()
    window = manager.window  # 获取窗口对象

    # 设置窗口位置和大小（例如：设置在屏幕的 (100, 100) 坐标，宽高为 800x600）
    window.wm_geometry("800x600+100+100")

    while True:
        # 获取目录中的所有图片文件
        image_files = [f for f in os.listdir(image_directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            # 排序，确保按时间顺序显示
            image_files.sort(key=lambda f: os.path.getmtime(os.path.join(image_directory, f)))
            latest_image = image_files[-1]  # 获取最新的图片
            latest_image_path = os.path.join(image_directory, latest_image)

            # 只在有新的图片时更新
            image_time = os.path.getmtime(latest_image_path)
            if image_time > last_image_time:
                last_image_time = image_time
                print(f"Rendering new image: {latest_image}")

                # 在渲染新图像前关闭旧图形
                plt.close(fig)

                # 创建一个新的图形
                fig, ax = plt.subplots()

                # 读取并显示最新的图片
                img = mpimg.imread(latest_image_path)
                ax.imshow(img)
                ax.axis('off')  # 不显示坐标轴
                plt.draw()  # 更新绘图
                plt.pause(0.5)  # 每0.5秒检查一次图片更新

        time.sleep(1)  # 每秒检查一次图片目录

if __name__ == "__main__":
    img_directory = r"Client/logs/img_client1/prev_circle"
    render_images(img_directory)
