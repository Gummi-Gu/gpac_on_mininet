import os
import subprocess


def generate_restart_scripts(venv_python, modules, project_root):
    script_folder = project_root  # 你也可以改成 project_root + "/scripts"
    for module in modules:
        script_name = f"run_{module.split('.')[-1]}.bat"
        script_path = os.path.join(script_folder, script_name)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(f'''@echo off
:loop
echo Starting {module}...
"{venv_python}" -m {module}
echo {module} exited. Restarting in 1 second...
timeout /t 1 >nul
goto loop
''')
        print(f"Generated: {script_path}")



def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    venv_python = r"D:\Users\GummiGu\PycharmProjects\gpac_on_mininet\.venv\Scripts\python.exe"

    modules = [
        "Client.main1",
        "Client.main2",
    ]

    # 一次性生成脚本
    generate_restart_scripts(venv_python, modules, project_root)

    # 启动所有窗口
    for module in modules:
        script_name = f"run_{module.split('.')[-1]}.bat"
        script_path = os.path.join(project_root, script_name)
        subprocess.Popen([script_path], creationflags=subprocess.CREATE_NEW_CONSOLE)

    # 不再需要循环监控重启

if __name__ == "__main__":
    main()