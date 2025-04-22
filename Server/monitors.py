from flask import Flask, request, jsonify, send_file
from tabulate import tabulate
from collections import defaultdict
from datetime import datetime
import threading
from io import BytesIO

app = Flask(__name__)
lock = threading.Lock()

# 数据结构定义
# 这里track_stats新增了客户端ID的层级
track_stats = defaultdict(lambda: defaultdict(lambda: {
    'avg_delay': 0.0,
    'avg_rate': 0.0,
    'latest_delay': 0.0,
    'latest_rate': 0.0,
    'resolution': '',
    'last_update': None
}))

link_metrics = defaultdict(lambda: {
    'delay': 0.0,
    'loss_rate': 0.0,
    'mark':{},
    'last_update': None
})


@app.route('/track_stats', methods=['POST'])
def update_track_stats():
    data = request.get_json()
    with lock:
        track_id = data['track_id']
        client_id = data['client_id']  # 客户端ID
        track_stats[track_id][client_id].update({
            'avg_delay': data['avg_delay'],
            'avg_rate': data['avg_rate'],
            'latest_delay': data['latest_delay'],
            'latest_rate': data['latest_rate'],
            'last_update': datetime.now()
        })
    return jsonify({'status': 'success'})


@app.route('/link_metrics', methods=['POST'])
def update_link_metrics():
    data = request.get_json()
    with lock:
        client_id = data['client_id']  # 修改为 client_id
        marks = data['marks'] # 获取 marks 数据，默认为空字典

        # 更新 link_metrics 中的数据
        link_metrics[client_id].update({
            'delay': data['delay'],
            'loss_rate': data['loss_rate'],
            'marks': marks,  # 保存 marks 数据
            'last_update': datetime.now()
        })
    return jsonify({'status': 'success'})

latest_heatmap=None
@app.route('/update_heatmap', methods=['POST'])
def update_heatmap():
    global latest_heatmap
    image_data = request.data  # 获取二进制图像数据
    with lock:
        latest_heatmap = image_data
    return jsonify({'status': 'success'})

@app.route('/heatmap')
def serve_heatmap():
    global latest_heatmap
    with lock:
        if latest_heatmap is None:
            return "Heatmap not available", 404
        return send_file(
            BytesIO(latest_heatmap),
            mimetype='image/png'
        )

@app.route('/chunk_quality', methods=['POST'])
def update_chunk_quality():
    data = request.get_json()  # 接收的是一个字典，例如 {"0": "720p", "1": "1080p"}
    with lock:
        for chunk_id, resolution in data.items():
            # 这里没有特别区分客户端，可以选择加上客户端ID来更新
            if resolution == 0:
                resolution=200
            elif resolution == 1:
                resolution=785
            elif resolution == 2:
                resolution=3150
            elif resolution == 3:
                resolution=12600

            for client_id in track_stats[chunk_id]:
                track_stats[chunk_id][client_id].update({
                    'resolution': resolution
                })
    return jsonify({'status': 'success'})


def format_table(data, headers):
    return tabulate(
        [(k,) + tuple(v.values()) for k, v in data.items()],
        headers=headers,
        tablefmt="html",
        floatfmt=".2f"
    )


@app.route('/dashboard')
def show_dashboard():
    # Track Stats 表格
    track_headers = ['Track ID', 'Client ID', 'Avg Delay(ms)', 'Avg Rate(MB/s)',
                     'Latest Delay(ms)', 'Latest Rate(MB/s)', 'Bit rate', 'Last Update', 'Utilization(%)']

    # 格式化track_stats为适合的表格显示
    track_table_data = []
    total_delay = 0
    total_latest_rate = 0
    total_utilization = 0
    total_clients = 0
    for track_id, clients in track_stats.items():
        for client_id, stats in clients.items():
            if track_id == 'default':
                continue
            utilization = (stats['latest_rate'] / 25.0) * 100  # 计算利用率百分比
            total_delay += stats['latest_delay']
            total_latest_rate += stats['latest_rate']
            total_utilization += utilization
            total_clients += 1
            track_table_data.append((
                track_id, client_id,
                stats['avg_delay'], stats['avg_rate'],
                stats['latest_delay'], stats['latest_rate'],
                stats['resolution'], stats['last_update'],
                f"{utilization:.2f}%"
            ))

        # 计算总带宽利用率
    if total_clients > 0:
        average_utilization = (total_utilization / total_clients)
    else:
        average_utilization = 0.0

        # 添加 `default` 行，记录总时延、总最新带宽和总利用率
    track_table_data.append((
        'default', '0',
        0, 0,  # 平均延迟和平均速率为0
        total_delay, total_latest_rate,
        0,  # Bit rate设为0
        'N/A',  # Last Update设为N/A
        f"{total_utilization:.2f}%"  # 总带宽利用率
    ))

    track_table = tabulate(track_table_data, headers=track_headers, tablefmt="html", floatfmt=".2f")

    # Link Metrics 表格
    link_headers = ['Client ID', 'Delay(ms)', 'Loss Rate(%)', '12600_rate(Mbit)', '3150_rate(Mbit)', '785_rate(Mbit)', '200_rate(Mbit)', 'Last Update']
    link_data = []
    def mark2bw(x):
        if x == 10:
            return 50
        if x == 20:
            return 30
        if x == 30:
            return 20
    for client_id, stats in link_metrics.items():
        bw_12600=stats['marks']['12600']
        bw_12600=mark2bw(bw_12600)
        bw_785 = stats['marks']['785']
        bw_785 = mark2bw(bw_785)
        bw_200 = stats['marks']['200']
        bw_200 = mark2bw(bw_200)
        bw_3150 = stats['marks']['3150']
        bw_3150 = mark2bw(bw_3150)
        link_data.append((client_id, stats['delay'], stats['loss_rate'],bw_12600,bw_3150,bw_785,bw_200,stats['last_update']))
    link_table = tabulate(link_data, headers=link_headers, tablefmt="html", floatfmt=".2f")

    return f"""
    <html>
        <head>
            <title>Streaming Monitor Dashboard</title>
            <meta http-equiv="refresh" content="1">
            <style>
                table {{ border-collapse: collapse; margin: 20px; width: 90%; }}
                th, td {{ padding: 8px; border: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                tr:nth-child(odd) {{ background-color: #ffffff; }}
                tr:hover {{ background-color: #e6f7ff; }}
            </style>
        </head>
        <body>
            <h2>Track Statistics</h2>
            {track_table}

            <h2>Link Metrics</h2>
            {link_table}

        </body>
    </html>
    """


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
