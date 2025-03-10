import os
import time
import threading
from flask import Flask, Response, stream_with_context, abort, jsonify, send_file
from collections import defaultdict

app = Flask(__name__)
FILE_DIRECTORY = 'files'  # 文件存储目录
CHUNK_SIZE = 2048  # 每次传输的块大小 (2KB)
SPEED_REPORT_INTERVAL = 1  # 速度报告间隔 (秒)
STATS_5S_INTERVAL = 5
# 确保文件目录存在
os.makedirs(FILE_DIRECTORY, exist_ok=True)

# 统计数据存储
categories = ["200", "785", "3150", "12600"]
def default_stats():
    return {
        "total_bytes": 0,
        "total_time": 0,
        "avg_speed": 0,
        "total_avg_speed": 0,
        "count": 0,
        "avg_latency": 0,
        "view": [],
        "last_sec_bytes": 0,
        "last_sec_time": 0,
        "last_sec_speed": 0,
        "saved_last_sec_bytes": 0,
        "saved_last_sec_time": 0,
        "saved_sec_speed": 0,
    }

# 生成 tracks 的默认结构
def default_track():
    return {
        "total_bytes": 0,  # 总字节数
        "total_time": 0,  # 总时间
        "total_avg_speed": 0,  # 总平均速度
        "count": 0,  # 下载次数
        "avg_latency": 0,  # 平均时延
        "category": 0,  # 当前类

        "last_sec_bytes": 0,  # 当前秒的字节数
        "last_sec_time": 0,  # 当前秒的时间
        "last_sec_speed": 0,  # 当前秒的速度


        "saved_last_sec_bytes": 0,  # 保存的当前秒字节数
        "saved_last_sec_time": 0,  # 保存的当前秒时间
        "saved_sec_speed": 0,  # 保存的速度
    }


client_data = {
    '01': {
        'stats': {category: default_stats() for category in categories},
        'tracks': {i: default_track() for i in range(1, 10 + 1)}
    },
    '02': {
        'stats': {category: default_stats() for category in categories},
        'tracks': {i: default_track() for i in range(1, 10 + 1)}
    }
}

stats_lock = threading.Lock()


# 获取文件类别
def get_category(filename):
    for cat in categories:
        if cat in filename:
            return cat
    return "other"

def add_viewpoint(cat,idx,part_client_data):
    if cat in categories:
        for data in part_client_data['stats'].values():
            if idx in data["view"]:
                data["view"].remove(idx)
        part_client_data['stats'][cat]["view"].append(idx)

# 统计线程
def stats_logger():
    while True:
        time.sleep(SPEED_REPORT_INTERVAL)
        for client_id in ['01', '02']:
            part_client_data=client_data[client_id]
            for category, data in part_client_data['stats'].items():
                with stats_lock:
                    if data["last_sec_bytes"] == 0 or data["last_sec_time"] == 0:
                        data["last_sec_bytes"] = data["saved_last_sec_bytes"]
                        data["last_sec_time"] = data["saved_last_sec_time"]

                    if category in ["other", None]:
                        continue
                    data["saved_last_sec_bytes"] = data["last_sec_bytes"]
                    data["saved_last_sec_time"] = data["last_sec_time"]

                    # 清零当前秒数据
                    data["last_sec_bytes"] = 0
                    data["last_sec_time"] = 0

            for track_id, data in part_client_data['tracks'].items():
                with stats_lock:
                    if data["last_sec_bytes"]==0 or data["last_sec_time"] == 0:
                        data["last_sec_bytes"]=data["saved_last_sec_bytes"]
                        data["last_sec_time"]=data["saved_last_sec_time"]


                    # 保存当前秒的数据
                    data["saved_last_sec_bytes"] = data["last_sec_bytes"]
                    data["saved_last_sec_time"] = data["last_sec_time"]

                    # 清零当前秒数据，准备下一秒
                    data["last_sec_bytes"] = 0
                    data["last_sec_time"] = 0

                    add_viewpoint(data["category"],track_id-1,part_client_data)

# 启动统计线程
threading.Thread(target=stats_logger, daemon=True).start()

@app.route('/get_states', methods=['GET'])
def get_states():
    data_for_tracks = []
    stats_total = []

    with stats_lock:
        for track_id in range(2,11):
            data_1 = client_data['01']['tracks'][track_id]
            data_2 = client_data['02']['tracks'][track_id]
            Spd1 = 0
            if data_1['saved_last_sec_time']>0:
                Spd1 = data_1["saved_last_sec_bytes"]/data_1["saved_last_sec_time"]
            Spd2 = 0
            if data_2['saved_last_sec_time'] > 0:
                Spd2 = data_2["saved_last_sec_bytes"] / data_2["saved_last_sec_time"]
            data_for_tracks.append(
                {
                    "0.No": track_id - 1,
                    "0.cnt": f"{data_1['count'] + data_2['count']}",
                    "1.F_Spd": f"{Spd1 / 1024/1024:.1f} MB/s",
                    "1.F_kbps": data_1["category"],
                    "1.F_time": f"{data_1['saved_last_sec_time'] * 1000:.1f} ms",
                    "2.S_Spd": f"{Spd2 / 1024/1024:.1f} MB/s",
                    "2.S_Kbps": data_2["category"],
                    "2.S_time": f"{data_2['saved_last_sec_time'] * 1000:.1f} ms"
                    # "8.bytes(5s)": f"{data['five_sec_bytes'] / 1024:.1f} KB",
                    #"11.totMB": f"{data_1['total_bytes'] / 1024 / 1204:.1f} MB",
                    # "12.total_time": f"{data['total_time'] * 1000:.1f} ms",
                    #"13.avgLat": f"{data['avg_latency'] * 1000:.1f} ms",
                    #"14.avgSpd": f"{data['total_avg_speed'] / 1024:.1f} MB/s"
                }
            )

        for category in categories:
            if category == "other":
                continue
            data_1 = client_data['01']['stats'][category]
            data_2 = client_data['02']['stats'][category]
            avgSpd=0
            if data_1['total_time']+data_2['total_time']>0:
                avgSpd=(data_1['total_bytes']+data_2['total_bytes'])/(data_1['total_time']+data_2['total_time'])
            avgLat=0
            if data_1['count']+data_2['count']>0:
                avgLat=(data_1['total_time']+data_2['total_time'])/(data_1['count']+data_2['count'])
            Spd=0
            if data_1['saved_last_sec_time']+data_2['saved_last_sec_time']>0:
                Spd=(data_1['saved_last_sec_bytes']+data_2['saved_last_sec_bytes'])/(data_1['saved_last_sec_time']+data_2['saved_last_sec_time'])
            #if category == "12600":
             #   print(data_1['total_bytes'], data_1['total_time'], data_1['saved_last_sec_bytes'], data_1['saved_last_sec_time'],Spd,Spd/1024/1024*8)
            stats_total.append({
                "0.kbps": category,
                "1.cnt":f"{data_1['count']+data_2['count']}",
                #"2.totMB": f"{(data_1['total_bytes']+data_2['total_bytes']) /1024/1024:.1f} MB",
                #"2.totTime": f"{(data_1['total_time']+data_2['total_time']) * 1000:.1f} ms",
                #"2.bytes": f"{data_1['last_sec_bytes']/1024/1024:.1f} MB",
                #"2.time": f"{data_1['last_sec_time']* 1000:.1f}ms",
                "2.avgSpd": f"{avgSpd/1024/1024*8:.1f} MBits/s",
                "2.avgLat": f"{avgLat*1000:.1f} ms",
                "3.F_Viewpoint":f"{data_1['view']}",
                "3.S_Viewpoint": f"{data_2['view']}",
                "4.Spd": f"{Spd/1024/1024*8:.1f} MBits/s"
            })

    return jsonify({
        "data_for_tracks": data_for_tracks,
        "total_stats": stats_total
    })


@app.route('/<client_id>/files/<path:filename>')
def download_file(client_id,filename):
    file_path = os.path.join(FILE_DIRECTORY, filename)
    if client_id not in ['01', '02']:
        abort(404,"非法客户端")
    if not os.path.exists(file_path):
        abort(404, description="文件不存在")

    file_size = os.path.getsize(file_path)
    start_time = time.time()
    part_client_data = client_data[client_id]
    def generate():
        with open(file_path, 'rb') as f:
            while chunk := f.read(CHUNK_SIZE):  # 分块读取文件
                yield chunk
    #time.sleep(file_size/1024*0.001)
    # 创建流式响应
    response = Response(generate(), headers={
        'Content-Length': file_size,
        'Content-Disposition': f'attachment; filename={os.path.basename(filename)}'
    })

    # 注册传输完成后的回调
    if "init" not in filename and "mpd" not in filename:
        @response.call_on_close
        def record_stats():
            end_time = time.time()
            total_time = end_time - start_time + 0.005
            # 解析文件名并更新统计（同之前的逻辑）
            try:
                # 替换特殊字符并分割
                clean_name = filename.replace('.', '_')
                parts = clean_name.split('_')
                category = parts[0]
                track_id = int(parts[3][5:])  # 从类似"track5"中提取5
                timestamp = int(parts[4])
            except Exception as e:
                print(f"文件名解析失败: {str(e)}")
                return

                # 线程安全地更新统计信息
            with stats_lock:
                # 更新轨道统计
                track=part_client_data['tracks'][track_id]
                track["total_bytes"] += file_size
                track["last_sec_bytes"] += file_size
                track["last_sec_time"] += total_time
                track["total_time"] += total_time
                track["count"] += 1
                track["category"] = category

                # 计算平均速度（MB/s）
                if track["total_time"] > 0:
                    track["total_avg_speed"] = (track["total_bytes"] / track["total_time"]) / 1024

                # 计算平均延迟
                if track["count"] > 0:
                    track["avg_latency"] = track["total_time"] / track["count"]

                # 更新类别统计
                category_stats=part_client_data['stats'][category]
                category_stats["total_bytes"] += file_size
                category_stats["total_time"] += total_time
                category_stats["count"] += 1
                category_stats["last_sec_bytes"] += file_size
                category_stats["last_sec_time"] += total_time

                if category_stats["count"] > 0:
                    category_stats["avg_latency"] = category_stats["total_time"] / category_stats["count"]

    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10086, debug=True)
