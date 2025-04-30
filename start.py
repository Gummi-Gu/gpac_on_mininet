import subprocess
import sys
import time
import os

def start_monitor(venv_python, project_root):
    monitor_process = subprocess.Popen(
        [venv_python, "Server/monitors.py"],
        cwd=project_root,
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    time.sleep(3)

def start_single_client(venv_python, module, project_root):
    print(f"Starting {module}...")
    p = subprocess.Popen(
        [venv_python, "-m", module],
        cwd=project_root,
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    return {"module": module, "process": p, "start_time": time.time()}

def restart_client(process_info, venv_python, project_root):
    p = process_info["process"]
    module = process_info["module"]
    if p.poll() is None:
        print(f"Terminating process {p.pid} ({module})...")
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(f"Process {p.pid} did not terminate, killing...")
            p.kill()
    return start_single_client(venv_python, module, project_root)

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))  # 项目根目录
    venv_python = r"D:\Users\GummiGu\PycharmProjects\gpac_on_mininet\.venv\Scripts\python.exe"

    modules = [
        "Client.main1",
        "Client.main2",
        "Client.main3",
    ]

    if len(sys.argv) > 1 and sys.argv[1] == 'monitor':
        start_monitor(venv_python, project_root)

    # 启动所有客户端，每隔10秒一个
    process_infos = []
    for module in modules:
        process_infos.append(start_single_client(venv_python, module, project_root))
        time.sleep(10)

    try:
        while True:
            time.sleep(1)  # 主循环每秒检查
            for i, info in enumerate(process_infos):
                p = info["process"]
                if p.poll() is not None:
                    print(f"Process {p.pid} ({info['module']}) exited early. Restarting...")
                    process_infos[i] = start_single_client(venv_python, info["module"], project_root)
                elif time.time() - info["start_time"] > 60:
                    print(f"Process {p.pid} ({info['module']}) reached 60s. Restarting...")
                    process_infos[i] = restart_client(info, venv_python, project_root)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Stopping all clients...")
        for info in process_infos:
            p = info["process"]
            if p.poll() is None:
                print(f"Terminating process {p.pid}...")
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"Process {p.pid} did not terminate, killing...")
                    p.kill()

if __name__ == "__main__":
    main()
