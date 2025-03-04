import time
import curses
import requests

SERVER_URL = "http://localhost:5000/get_status"


def fetch_status():
    """从服务器获取当前监控状态"""
    try:
        response = requests.get(SERVER_URL)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None


def monitor_cli(stdscr):
    curses.curs_set(0)  # 隐藏光标
    stdscr.nodelay(1)  # 允许非阻塞输入

    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, "视频传输监控界面", curses.A_BOLD)

        # 获取终端宽度
        height, width = stdscr.getmaxyx()

        # 获取数据
        data = fetch_status()
        if data:
            stdscr.addstr(6, 0, f"Buffer 长度: {(data['buffer_length']+1000)/1000:.2f}s / {data['max_buffer_length']/1000:.2f}s")
            stdscr.addstr(7, 0, f"下载速度: {data['download_speed']:<10}bps")
            stdscr.addstr(8, 0, "视频切片状态:")

            for i, slice_info in enumerate(data["slices"]):
                # **确保左对齐**
                line = (f"ID {slice_info['idx']:<3} | Bitrate: {slice_info['bitrate']//1024:<10}kbps | "
                        f"Speed: {slice_info['download_speed']//1024:<10}kbps | FPS: {slice_info['fps']:<5} | "
                        f"Res: {slice_info['resolution']:<10}")

                # **确保不会超出屏幕宽度**
                stdscr.addstr(9 + i, 0, line.ljust(width - 1))  # **ljust 让字符串保持固定长度，防止错位**
        else:
            stdscr.addstr(6, 0, "无法获取服务器数据", curses.color_pair(1))

        stdscr.refresh()
        time.sleep(1)

def start_monitor():
    curses.wrapper(monitor_cli)
