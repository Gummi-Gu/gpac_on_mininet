# client.py
import time

import requests
from tabulate import tabulate


# 获取服务器的统计数据
def fetch_stats():
    response = requests.get('http://127.0.0.1:10086/get_states')
    if response.status_code == 200:
        return response.json()
    else:
        print(f"请求失败，状态码：{response.status_code}")
        return None


# 打印统计数据
def print_stats(stats):
    if not stats:
        return

    def generate_table(data, title,max_col_width=None):
        if not data:
            return f"\n{title} (No data)\n"

        # 提取表头：去掉 "1.", "2." 等编号前缀，并美化格式
        headers = [key.split('.', 1)[-1].replace('_', ' ').title() for key in data[0].keys()]

        # 提取行数据（保持字典顺序）
        rows = [[value for value in entry.values()] for entry in data]
        tablefmt = "grid"
        tabulate_args = {
            "headers": headers,
            "tablefmt": tablefmt,
            "maxcolwidths": max_col_width  # 允许传入数字或列表
        }

        # 生成表格
        return f"\n{title}\n" + tabulate(rows, **tabulate_args)
    # 生成所有表格
    print(generate_table(stats['1/5s_stats'], "1/5 Second Statistics",max_col_width=15))
    print(generate_table(stats['total_stats'], "Total Statistics per Category",max_col_width=15))
# 主函数
def main():
    while True:
        stats = fetch_stats()
        print_stats(stats)
        print("\n" + "=" * 50)
        time.sleep(1)  # 每秒更新一次


if __name__ == '__main__':
    main()
