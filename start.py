import subprocess
import time
import os
import signal

def start_clients(venv_python, modules, project_root):
    """启动所有客户端进程"""
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
            cwd=project_root,
        )
        processes.append(p)
        time.sleep(1)  # 稍微间隔一下，避免过挤
    print("All clients started.\n")
    return processes

def stop_clients(processes):
    """停止所有客户端进程"""
    for p in processes:
        if p.poll() is None:  # 还在运行的才需要终止
            print(f"Terminating process {p.pid}...")
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"Process {p.pid} did not terminate, killing...")
                p.kill()

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))  # 项目根目录
    venv_python = r"D:\Users\GummiGu\PycharmProjects\gpac_on_mininet\.venv\Scripts\python.exe"

    modules = [
        "Client.main1",
        "Client.main2",
        "Client.main3",
    ]

    timeout = 60  # 每轮运行时间（秒）

    try:
        for _ in range(120):
            processes = start_clients(venv_python, modules, project_root)
            start_time = time.time()

            while True:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    print(f"\nTime limit {timeout}s reached.")
                    break

                # 检查子进程是否意外退出
                for p in processes:
                    if p.poll() is not None:
                        print(f"Process {p.pid} exited early.")
                        break  # 有子进程提前退出了，跳出重新启动

                time.sleep(1)

            print("\nRestarting clients...\n")
            stop_clients(processes)
            time.sleep(1)  # 给点时间再重启

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Stopping all clients...")
        stop_clients(processes)

    except Exception as e:
        print(f"An error occurred: {e}")
        stop_clients(processes)

if __name__ == "__main__":
    main()
