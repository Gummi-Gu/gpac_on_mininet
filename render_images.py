import os
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 使用 TkAgg 后端，保证可以控制窗口
matplotlib.use('TkAgg')

def render_images(image_directory):
    last_image_time = 0
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)  # 设置窗口为800x800像素

    img_plot = None  # 保存 imshow 返回的对象

    while True:
        image_files = [f for f in os.listdir(image_directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            image_files.sort(key=lambda f: os.path.getmtime(os.path.join(image_directory, f)))
            latest_image = image_files[-1]
            latest_image_path = os.path.join(image_directory, latest_image)

            image_time = os.path.getmtime(latest_image_path)
            if image_time > last_image_time:
                last_image_time = image_time
                print(f"Rendering new image: {latest_image}")

                img = mpimg.imread(latest_image_path)

                if img_plot is None:
                    img_plot = ax.imshow(img)
                    ax.axis('off')
                    ax.set_aspect('equal')
                    ax.set_xlim(0, img.shape[1])
                    ax.set_ylim(img.shape[0], 0)
                    fig.tight_layout()
                else:
                    img_plot.set_data(img)
                    ax.set_xlim(0, img.shape[1])
                    ax.set_ylim(img.shape[0], 0)

                plt.draw()
                plt.pause(0.5)

        time.sleep(1)

if __name__ == "__main__":
    img_directory = r"Client/logs/img_client1/prev_circle"
    render_images(img_directory)
