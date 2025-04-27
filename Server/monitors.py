from flask import Flask, request, jsonify, send_file
from tabulate import tabulate
from collections import defaultdict
from datetime import datetime
import threading
from io import BytesIO

app = Flask(__name__)
lock = threading.Lock()
string_dict = {
            '12600': 0,
            '3150': 0,
            '785': 0,
            '200': 0
        }
buffer = {'re_buffer':0.0,'play_buffer':0.0}
quality_map = {
            0: 0,
            1: 1,
            2: 3,
            3: 3
}
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

client_stats = defaultdict(lambda: {
    'rebuffer_time': 0.0,
    'rebuffer_count': 0.0,
    'qoe':0.0,
    'last_update': None
})

# 新增的数据结构
TRAFFIC_CLASSES_MARK = {
    '10.0.0.2' : {'port': 10086, '12600': 20, '3150':20, '785':30, '200':30},
    '10.0.0.3' : {'port': 10086, '12600': 10, '3150':10, '785':30, '200':30},
}
TRAFFIC_CLASSES_DELAY = {
    '10.0.0.2' : {'client': 'client1','delay': 0, 'loss':0},
    '10.0.0.3' : {'client': 'client2','delay': 0, 'loss':0}
}

# 接口：获取 TRAFFIC_CLASSES_MARK
@app.route('/get/traffic_classes_mark', methods=['GET'])
def get_traffic_classes_mark():
    with lock:
        return jsonify(TRAFFIC_CLASSES_MARK)

# 接口：更新 TRAFFIC_CLASSES_MARK
@app.route('/update/traffic_classes_mark', methods=['POST'])
def update_traffic_classes_mark():
    data = request.get_json()
    if data:
        with lock:
            TRAFFIC_CLASSES_MARK.update(data)
        return jsonify({"status": "success", "message": "TRAFFIC_CLASSES_MARK updated"})
    else:
        return jsonify({"status": "error", "message": "Invalid data"}), 400

# 接口：获取 TRAFFIC_CLASSES_DELAY
@app.route('/get/traffic_classes_delay', methods=['GET'])
def get_traffic_classes_delay():
    with lock:
        return jsonify(TRAFFIC_CLASSES_DELAY)

# 接口：更新 TRAFFIC_CLASSES_DELAY
@app.route('/update/traffic_classes_delay', methods=['POST'])
def update_traffic_classes_delay():
    data = request.get_json()
    if data:
        with lock:
            TRAFFIC_CLASSES_DELAY.update(data)
        return jsonify({"status": "success", "message": "TRAFFIC_CLASSES_DELAY updated"})
    else:
        return jsonify({"status": "error", "message": "Invalid data"}), 400



@app.route('/update/string_dict', methods=['POST'])
def update_string_dict():
    new_dict = request.json
    if new_dict:
        string_dict.update(new_dict)
    return jsonify({"status": "success", "message": "string_dict updated"})

@app.route('/get/string_dict', methods=['GET'])
def get_string_dict():
    return jsonify({"string_dict": string_dict})


@app.route('/update/buffer', methods=['POST'])
def update_re_buffer():
    value1 = request.json.get('re_buffer')
    value2 = request.json.get('play_buffer')
    if value1 is not None and value2 is not None:
        buffer['re_buffer'] = value1
        buffer['play_buffer'] = value2
    return jsonify({"status": "success", "message": "buffer updated"})

@app.route('/get/buffer', methods=['GET'])
def get_re_buffer():
    return jsonify({"re_buffer": buffer['re_buffer'], 'play_buffer': buffer['play_buffer']})

@app.route('/update/quality_map', methods=['POST'])
def update_quality_map():
    new_map = request.json
    if new_map:
        quality_map.update(new_map)
    return jsonify({"status": "success", "message": "quality_map updated"})

@app.route('/get/quality_map', methods=['GET'])
def get_quality_map():
    return jsonify({"quality_map": quality_map})


@app.route('/get_track_stats', methods=['GET'])
def get_track_stats():
    with lock:
        return jsonify(track_stats)

@app.route('/get_link_metrics', methods=['GET'])
def get_link_metrics():
    with lock:
        return jsonify(link_metrics)

@app.route('/get_client_stats', methods=['GET'])
def get_client_stats():
    with lock:
        return jsonify(client_stats)

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

@app.route('/client_stats', methods=['POST'])
def update_client_stats():
    data = request.get_json()
    with lock:
        client_id = data['client_id']  # 修改为 client_id
        # 更新 client_stats 中的数据
        client_stats[client_id].update({
            'rebuffer_time': data['rebuffer_time'],
            'rebuffer_count': data['rebuffer_count'],
            'qoe': data['qoe'],
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
    client_summary = {}

    for track_id, clients in track_stats.items():
        for client_id, stats in clients.items():
            if track_id == 'default':
                continue

            utilization = (stats['latest_rate'] / 100.0) * 100  # 假设最大带宽是100
            if client_id not in client_summary:
                client_summary[client_id] = {
                    'total_delay': 0.0,
                    'total_latest_rate': 0.0,
                    'total_utilization': 0.0,
                    'track_count': 0,
                }

            summary = client_summary[client_id]
            summary['total_delay'] += stats['latest_delay']
            summary['total_latest_rate'] += stats['latest_rate']
            summary['total_utilization'] += utilization
            summary['track_count'] += 1

            # 添加原始track的详细信息
            track_table_data.append((
                track_id, client_id,
                stats['avg_delay'], stats['avg_rate'],
                stats['latest_delay'], stats['latest_rate'],
                stats['resolution'], stats['last_update'],
                f"{utilization:.2f}%"
            ))

    # 添加每个客户端的汇总行
    for client_id, summary in client_summary.items():
        avg_utilization = summary['total_utilization'] / summary['track_count']
        track_table_data.append((
            'client_total', client_id,
            0, 0,  # 平均延迟和速率不统计
            summary['total_delay'], summary['total_latest_rate'],
            0,  # 分辨率为0
            'N/A',
            f"{summary['total_utilization']:.2f}%"
        ))

    track_table = tabulate(track_table_data, headers=track_headers, tablefmt="html", floatfmt=".2f")

    # Link Metrics 表格
    link_headers = ['Client ID', 'Delay(ms)', 'Loss Rate(%)', '12600_rate(Mbit)', '3150_rate(Mbit)', '785_rate(Mbit)', '200_rate(Mbit)', 'Last Update']
    link_data = []
    def mark2bw(x):
        if x == 10:
            return 200
        if x == 20:
            return 60
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

    client_headers = ['Client ID', 'Rebuffer Time(s)', 'Rebuffer Count', 'QoE', 'Last Update']
    client_table_data = []
    for client_id, stats in client_stats.items():
        client_table_data.append((
            client_id,
            stats['rebuffer_time'],
            stats['rebuffer_count'],
            stats['qoe'],
            stats['last_update']
        ))
    client_table = tabulate(client_table_data, headers=client_headers, tablefmt="html", floatfmt=".2f")

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

               <h2>Client Statistics</h2>
               {client_table}

           </body>
       </html>
       """


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
