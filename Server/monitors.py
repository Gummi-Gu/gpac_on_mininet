import json

import pandas as pd
from flask import Flask, request, jsonify, send_file
from numpy.distutils.exec_command import temp_file_name
from sympy import threaded
import random
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
quality_map = {
    'client1':{0:0,1:1,2:2,3:3},
    'client2':{0:0,1:1,2:2,3:3},
    'client3':{0:0,1:1,2:2,3:3},
}
rebuffer_config = {
    'client1':{'re_buffer': 1,'play_buffer': 2},
    'client2':{'re_buffer': 2,'play_buffer': 3},
    'client3':{'re_buffer': 1,'play_buffer': 2},
}
# 数据结构定义
# 这里track_stats新增了客户端ID的层级
bitrate_stats = defaultdict(lambda: defaultdict(lambda: {
    'avg_delay': 0.0,
    'avg_rate': 0.0,
    'latest_delay': 0.0,
    'latest_rate': 0.0,
    'resolution': '',
    'last_update': None,
    'avg_size': 0.0,
}))

track_stats = defaultdict(lambda: defaultdict(lambda: {
    'avg_delay': 0.0,
    'avg_rate': 0.0,
    'latest_delay': 0.0,
    'latest_rate': 0.0,
    'resolution': '',
    'last_update': None
}))

summary_rate_stats = {
    'client1': {
        'size': 0,
        'time': 1
    },
    'client2': {
        'size': 0,
        'time': 1
    },
    'client3': {
        'size': 0,
        'time': 1
    },
}

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
    '10.0.0.2' : {'port': 10086, '12600': 0.0, '3150':0.0, '785':0.00, '200':0.00},
    '10.0.0.3' : {'port': 10086, '12600': 0.0, '3150':0.0, '785':0.00, '200':0.00},
    '10.0.0.4' : {'port': 10086, '12600': 0.0, '3150':0.0, '785':0.00, '200':0.00},
}
#TRAFFIC_CLASSES_MARK = {
#    '10.0.0.2' : {'port': 10086, '12600': 0.65, '3150':0.25, '785':0.05, '200':0.05},
#    '10.0.0.3' : {'port': 10086, '12600': 0.0, '3150':0.0, '785':0.00, '200':0.00},
#    '10.0.0.4' : {'port': 10086, '12600': 0.0, '3150':0.0, '785':0.00, '200':0.00},
#}


ip_maps={
    'client1':'0.0.0.0',
    'client2':'0.0.0.0',
}

orign_quality_tiled={
    'client1':[0,2,2,2,2,2,2,2,2,2],
    'client2':[0,2,2,2,2,2,2,2,2,2],
    'client3':[0,2,2,2,2,2,2,2,2,2],
}

#========== 时延 ==========
TRAFFIC_CLASSES_DELAY = {
    '10.0.0.2' : {'client': 'client1','delay': 0, 'loss':0},
    '10.0.0.3' : {'client': 'client2','delay': 0, 'loss':0},
    '10.0.0.4' : {'client': 'client3','delay': 0, 'loss':0}
}
# 加载链路数据（从 CSV 文件读取）
def load_link_data(csv_file):
    # 读取 CSV 文件并返回 DataFrame
    df = pd.read_csv(csv_file)
    return df


# 按 Source 和 Destination 组合分组链路
def group_by_source_destination(df):
    # 按 Source 和 Destination 分组，返回一个字典，key 是 (Source, Destination) 组合，value 是对应的时延数据列表
    grouped = df.groupby(['Source', 'Destination'])['RTT'].apply(list).to_dict()
    return grouped


# 为每个 IP 随机选择一个链路
def assign_random_links(grouped_data):
    for ip in TRAFFIC_CLASSES_DELAY:
        # 随机选择一个 (Source, Destination) 组合
        random_source_dest = random.choice(list(grouped_data.keys()))
        TRAFFIC_CLASSES_DELAY[ip]['source'], TRAFFIC_CLASSES_DELAY[ip]['destination'] = random_source_dest
        # 随机选择该组合的一个时延
        TRAFFIC_CLASSES_DELAY[ip]['delay'] = random.choice(grouped_data[random_source_dest])
        TRAFFIC_CLASSES_DELAY[ip]['index'] = 0  # 初始时设置为第一个时间片


# 更新链路时延数据，切换到下一个时间片
def next_traffic_classes_delay():
    global time_slot_index,grouped_data

    # 锁定更新操作
    with lock:
        time_slot_index = (time_slot_index + 1) % num_time_slots
        for ip in TRAFFIC_CLASSES_DELAY:
            # 获取当前 IP 所选的 Source 和 Destination
            source = TRAFFIC_CLASSES_DELAY[ip]['source']
            destination = TRAFFIC_CLASSES_DELAY[ip]['destination']

            # 获取当前时间片的链路时延
            current_time = time_slot_index + 1
            # 获取这个 (Source, Destination) 对应的所有时延
            link_data = grouped_data.get((source, destination), [])
            if link_data:
                # 如果有时延数据，更新当前时间片的链路时延
                TRAFFIC_CLASSES_DELAY[ip]['delay'] = link_data[(current_time - 1) % len(link_data)]


# 配置时间片数量和锁
num_time_slots = 688  # 假设有 688 个时间片
time_slot_index = 0  # 当前时间片索引

# 加载链路数据
csv_file = 'valid_links.csv'  # 替换成实际的 CSV 文件路径
link_data = load_link_data(csv_file)

# 按 Source 和 Destination 分组链路数据
grouped_data = group_by_source_destination(link_data)

# 初始化时分配随机链路
assign_random_links(grouped_data)

# ========== ===========


def mark2bw(x):
    if x == 10:
        return 50
    if x == 20:
        return 20
    if x == 30:
        return 10
# 接口：获取 ip_maps

@app.route('/get/orign_quality_tiled', methods=['GET'])
def get_orign_quality_tiled():
    with lock:
        return jsonify(orign_quality_tiled)
@app.route('/update/orign_quality_tiled', methods=['POST'])
def update_orign_quality_tiled():
    data = request.get_json()
    if data:
        with lock:
            orign_quality_tiled.update(data)
        return jsonify({"status": "success", "message": "orign_quality_tiled"})
    else:
        return jsonify({"status": "error", "message": "Invalid data"}), 400

@app.route('/get/ip_maps', methods=['GET'])
def get_ip_maps():
    with lock:
        return jsonify(ip_maps)

# 接口：更新 ip_maps
@app.route('/update/ip_maps', methods=['POST'])
def update_ip_maps():
    data = request.get_json()
    if data:
        with lock:
            ip_maps.update(data)
        print(ip_maps)
        return jsonify({"status": "success", "message": "ip_maps updated"})
    else:
        return jsonify({"status": "error", "message": "Invalid data"}), 400

@app.route('/get/summary_rate_stats', methods=['GET'])
def get_summary_rate_stats():
    with lock:
        return jsonify(summary_rate_stats)
@app.route('/update/summary_rate_stats', methods=['POST'])
def update_summary_rate_stats():
    data = request.get_json()
    if data:
        with lock:
            summary_rate_stats.update(data)
            # 将数据以 JSON 格式追加写入到 bandwidth.txt
            with open("bandwidth.txt", "a") as f:
                f.write(json.dumps(data) + "\n")
        return jsonify({"status": "success", "message": "summary_rate_stats updated"})
    else:
        return jsonify({"status": "error", "message": "Invalid data"}), 400


# 接口：获取 TRAFFIC_CLASSES_MARK
temp_float=1.0
@app.route('/get/traffic_classes_mark', methods=['GET'])
def get_traffic_classes_mark():
    global temp_float
    with lock:
        return jsonify(TRAFFIC_CLASSES_MARK)

# 接口：更新 TRAFFIC_CLASSES_MARK
@app.route('/update/traffic_classes_mark', methods=['POST'])
def update_traffic_classes_mark():
    data = request.get_json()
    if data:
        with lock:
            TRAFFIC_CLASSES_MARK.update(data)
        #print(TRAFFIC_CLASSES_MARK)
        return jsonify({"status": "success", "message": "TRAFFIC_CLASSES_MARK updated"})
    else:
        return jsonify({"status": "error", "message": "Invalid data"}), 400

@app.route('/get/traffic_classes_delay', methods=['GET'])
def get_traffic_classes_delay():
    next_traffic_classes_delay()
    with lock:
        return jsonify(TRAFFIC_CLASSES_DELAY)

# 接口：更新 TRAFFIC_CLASSES_DELAY
@app.route('/update/traffic_classes_delay', methods=['POST'])
def update_traffic_classes_delay():
    data = request.get_json()
    if data:
        with lock:
            TRAFFIC_CLASSES_DELAY.update(data)
        #print(TRAFFIC_CLASSES_MARK)
        return jsonify({"status": "success", "message": "TRAFFIC_CLASSES_DELAY updated"})
    else:
        return jsonify({"status": "error", "message": "Invalid data"}), 400

@app.route('/get/quality_map', methods=['GET'])
def get_quality_map():
    #for k, (min_val, max_val) in {2: (1, 3), 3: (2, 3)}.items():
            #delta = random.choice([-1, 0, 1])
            #new_val = quality_map['client2'][k] + delta
            #quality_map['client2'][k] = max(min_val, min(max_val, new_val))
    with lock:
        return jsonify(quality_map)

# 接口：更新 TRAFFIC_CLASSES_MARK
@app.route('/update/quality_map', methods=['POST'])
def update_quality_map():
    data = request.get_json()
    if data:
        with lock:
            quality_map.update(data)
        #print(quality_map)
        return jsonify({"status": "success", "message": "TRAFFIC_CLASSES_MARK updated"})
    else:
        return jsonify({"status": "error", "message": "Invalid data"}), 400

@app.route('/get/rebuffer_config', methods=['GET'])
def get_rebuffer_config():
    with lock:
        return jsonify(rebuffer_config)

@app.route('/update/rebuffer_config', methods=['POST'])
def update_rebuffer_config():
    data = request.get_json()
    if data:
        with lock:
            rebuffer_config.update(data)
        #print(rebuffer_config)
        return jsonify({"status": "success", "message": "rebuffer_config updated"})
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
    return jsonify(string_dict)


@app.route('/get/bitrate_stats', methods=['GET'])
def get_bitrate_stats():
    with lock:
        return jsonify(bitrate_stats)

@app.route('/get/track_stats', methods=['GET'])
def get_track_stats():
    with lock:
        return jsonify(track_stats)

@app.route('/get/link_metrics', methods=['GET'])
def get_link_metrics():
    with lock:
        return jsonify(link_metrics)

@app.route('/get/client_stats', methods=['GET'])
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
        #print(track_stats)
    return jsonify({'status': 'success'})

@app.route('/bitrate_stats', methods=['POST'])
def update_bitrate_stats():
    data = request.get_json()
    with lock:
        bitrate_id = data['bitrate_id']
        client_id = data['client_id']  # 客户端ID
        bitrate_stats[bitrate_id][client_id].update({
            'avg_delay': data['avg_delay'],
            'avg_rate': data['avg_rate'],
            'latest_delay': data['latest_delay'],
            'latest_rate': data['latest_rate'],
            'last_update': datetime.now(),
            'avg_size': data['avg_size'],
        })
        #print(track_stats)
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
        with open("QoE",'a') as f:
            f.write(str(client_stats))
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
        #print(link_metrics)
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
    data = request.get_json()
    with lock:
        client_id = data['client']
        for chunk_id, resolution in data.items():
            if chunk_id == 'client':
                continue
            # 这里没有特别区分客户端，可以选择加上客户端ID来更新
            if resolution == 0:
                resolution=200
            elif resolution == 1:
                resolution=785
            elif resolution == 2:
                resolution=3150
            elif resolution == 3:
                resolution=12600

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
                     'Latest Delay(ms)', 'Latest Rate(MB/s)', 'Bit rate', 'Utilization(%)']

    # 格式化track_stats为适合的表格显示
    track_table_data = []
    client_summary = {}

    for track_id, clients in track_stats.items():
        for client_id, stats in clients.items():
            if track_id == 'default':
                continue

            utilization = (stats['latest_rate'] / 12.5) * 100  # 假设最大带宽是100
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
                stats['resolution'],
                f"{utilization:.2f}%"
            ))
    for client_id, stats in client_summary.items():
        utilization=(summary_rate_stats[client_id]['size'] / 12.5) * 100
        track_table_data.append((
            'summary', client_id,
            0.0, 0.0,
            summary_rate_stats[client_id]['time'], summary_rate_stats[client_id]['size'],
            0.0, 0.0,
            f"{utilization:.2f}%"
        ))

    track_table = tabulate(track_table_data, headers=track_headers, tablefmt="html", floatfmt=".2f")

    # Bitrate Stats 表格
    bitrate_headers = ['Bitrate', 'Client ID', 'Avg Delay(ms)', 'Avg Rate(MB/s)',
                       'Latest Delay(ms)', 'Latest Rate(MB/s)',  'Utilization(%)']

    # 格式化bitrate_stats为适合的表格显示
    bitrate_table_data = []
    bitrate_summary = {}

    for bitrate, clients in bitrate_stats.items():
        for client_id, stats in clients.items():
            utilization = (stats['latest_rate'] / 100.0) * 100  # 假设最大带宽是100
            if client_id not in bitrate_summary:
                bitrate_summary[client_id] = {
                    'total_delay': 0.0,
                    'total_latest_rate': 0.0,
                    'total_utilization': 0.0,
                    'bitrate_count': 0,
                }

            summary = bitrate_summary[client_id]
            summary['total_delay'] += stats['latest_delay']
            summary['total_latest_rate'] += stats['latest_rate']
            summary['total_utilization'] += utilization
            summary['bitrate_count'] += 1

            # 添加原始bitrate的详细信息
            bitrate_table_data.append((
                bitrate, client_id,
                stats['avg_delay'], stats['avg_rate'],
                stats['latest_delay'], stats['latest_rate'],
                f"{utilization:.2f}%"
            ))

    bitrate_table = tabulate(bitrate_table_data, headers=bitrate_headers, tablefmt="html", floatfmt=".2f")

    # Link Metrics 表格
    link_headers = ['Client ID', 'Delay(ms)', 'Loss Rate(%)', '12600_rate(Mbit)', '3150_rate(Mbit)', '785_rate(Mbit)', '200_rate(Mbit)']
    link_data = []
    for client_id, stats in link_metrics.items():
        bw_12600=stats['marks']['12600']
        bw_12600=mark2bw(bw_12600)
        bw_785 = stats['marks']['785']
        bw_785 = mark2bw(bw_785)
        bw_200 = stats['marks']['200']
        bw_200 = mark2bw(bw_200)
        bw_3150 = stats['marks']['3150']
        bw_3150 = mark2bw(bw_3150)
        link_data.append((client_id, stats['delay'], stats['loss_rate'],bw_12600,bw_3150,bw_785,bw_200))
    link_table = tabulate(link_data, headers=link_headers, tablefmt="html", floatfmt=".2f")

    client_headers = ['Client ID', 'Rebuffer Time(s)', 'Rebuffer Count', 'QoE']
    client_table_data = []
    for client_id, stats in client_stats.items():
        client_table_data.append((
            client_id,
            stats['rebuffer_time'],
            stats['rebuffer_count'],
            stats['qoe']
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

               <h2>Bitrate Statistics</h2>
               {bitrate_table}

               <h2>Link Metrics</h2>
               {link_table}

               <h2>Client Statistics</h2>
               {client_table}

           </body>
       </html>
       """


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
