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
                     'Latest Delay(ms)', 'Latest Rate(MB/s)', 'Bit rate', 'Last Update']

    # 格式化track_stats为适合的表格显示
    track_table_data = []
    for track_id, clients in track_stats.items():
        for client_id, stats in clients.items():
            if track_id == 'default':
                continue
            track_table_data.append((track_id, client_id) + tuple(stats.values()))

    track_table = tabulate(track_table_data, headers=track_headers, tablefmt="html", floatfmt=".2f")
    print(len(track_table_data))

    # Link Metrics 表格
    link_headers = ['Client ID', 'Delay(ms)', 'Loss Rate(%)', 'Mark', 'Last Update']
    link_data = []
    for client_id, stats in link_metrics.items():
        link_data.append((client_id, stats['1.delay'], stats['2.loss_rate'], stats['3.marks'], stats['4.last_update']))
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
    app.run(host='0.0.0.0', port=5000, debug=False)
