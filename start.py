import subprocess
import time
import os
import signal

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))  # 项目根目录
    venv_python = r"D:\Users\GummiGu\PycharmProjects\gpac_on_mininet\.venv\Scripts\python.exe"

    modules = [
        "Client.main1",
        "Client.main2",
        "Client.main3",
    ]

    processes = []

    # 启动图片渲染程序
    #render_process = subprocess.Popen(
    #    [venv_python, "render_images.py"],  # 使用虚拟环境中的 Python 启动 render_images.py
    #    cwd=project_root,  # 保证在项目根目录启动
    #)
    #processes.append(render_process)

    for module in modules:
        print(f"Starting {module}...")
        p = subprocess.Popen(
            [venv_python, "-m", module],
            cwd=project_root,  # 保证在项目根目录启动
        )
        processes.append(p)
        time.sleep(2)  # 小等一下，避免太挤

    print("All clients started.")

    # 等待所有子进程完成
    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("\nStopping all clients...")
        for p in processes:
            # 确保在退出时终止所有子进程
            p.terminate()
            p.wait()  # 等待子进程正确退出
    except Exception as e:
        print(f"An error occurred: {e}")
        for p in processes:
            p.terminate()
            p.wait()  # 等待子进程正确退出

if __name__ == "__main__":
    main()
