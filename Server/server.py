import os
import time
import threading
from flask import Flask, Response, stream_with_context, abort, jsonify, send_file
from collections import defaultdict

app = Flask(__name__)
FILE_DIRECTORY = 'files'  # 文件存储目录
CHUNK_SIZE = 4096  # 每次传输的块大小 (4KB)
SPEED_REPORT_INTERVAL = 1  # 速度报告间隔 (秒)
STATS_5S_INTERVAL = 5
# 确保文件目录存在
os.makedirs(FILE_DIRECTORY, exist_ok=True)

# 统计数据存储
categories = ["200", "785", "3150", "12600"]
stats = defaultdict(lambda: {"total_bytes": 0, "total_time": 0, "avg_speed": 0,
                             "total_avg_speed": 0,
                             "count":0,"avg_latency": 0})  # 保存前一秒数据
tracks = {i: {
    "total_bytes": 0,      # 总字节数
    "total_time": 0,       # 总时间
    "total_avg_speed": 0,  # 总平均速度
    "count": 0,  # 下载次数
    "avg_latency": 0,  # 平均时延

    "last_sec_bytes": 0,  # 当前秒的字节数
    "last_sec_time": 0,  # 当前秒的时间
    "last_sec_speed": 0,   # 当前秒的速度
    "category": 0, #当前类


    "saved_last_sec_bytes": 0,  # 保存的当前秒字节数
    "saved_last_sec_time": 0,   # 保存的当前秒时间
    "saved_sec_speed": 0,  # 保存的速度
    "saved_category": 0,  # 保存类

    "five_sec_bytes": 0,  # 5秒的字节数
    "five_sec_time": 0,  # 5秒的时间
    "five_sec_speed": 0,  # 5秒的速度
    "five_category": 0,  # 5类


} for i in range(1, 11)}  # 10个轨道

stats_lock = threading.Lock()


# 获取文件类别
def get_category(filename):
    for cat in categories:
        if cat in filename:
            return cat
    return "other"


# 统计线程
def stats_logger():
    while True:
        time.sleep(SPEED_REPORT_INTERVAL)
        for track_id, data in tracks.items():
            with stats_lock:
                if data["last_sec_bytes"]==0 or data["last_sec_time"] == 0:
                    data["last_sec_bytes"]=data["saved_last_sec_bytes"]
                    data["last_sec_time"]=data["saved_last_sec_time"]
                    data["last_sec_speed"] = data["saved_sec_speed"]
                    data["category"] = data["saved_category"]
                    continue
                # 计算当前秒的速度
                if data["last_sec_time"] > 0:
                    data["last_sec_speed"] = (data["last_sec_bytes"] / data["last_sec_time"]) / 1024
                else:
                    data["last_sec_speed"] = 0

                # 保存当前秒的数据
                data["saved_last_sec_bytes"] = data["last_sec_bytes"]
                data["saved_last_sec_time"] = data["last_sec_time"]
                data["saved_sec_speed"] = data["last_sec_speed"]
                data["saved_category"] = data["category"]

                # 清零当前秒数据，准备下一秒
                data["last_sec_bytes"] = 0
                data["last_sec_time"] = 0
        for category, data in stats.items():
            with stats_lock:
                if category is None or category == "other":
                    continue
                # 计算总平均速度
                data["total_avg_speed"] = (data["total_bytes"] / data["total_time"]) / 1024 if data["total_bytes"] > 0 else 0
# 每5秒更新一次，把最近1秒的保存数据赋值给5秒数据
def five_sec_logger():
    while True:
        time.sleep(STATS_5S_INTERVAL)
        with stats_lock:
            for track_id, data in tracks.items():
                # 直接把最近1秒的保存数据赋值给5秒数据
                data["five_sec_bytes"] = data["saved_last_sec_bytes"]
                data["five_sec_time"] = data["saved_last_sec_time"]
                data["five_sec_speed"] = data["saved_sec_speed"]
                data["five_category"] = data["saved_category"]


# 启动统计线程
threading.Thread(target=stats_logger, daemon=True).start()
threading.Thread(target=five_sec_logger, daemon=True).start()


@app.route('/get_states', methods=['GET'])
def get_states():
    stats_1_5s = []
    stats_total = []

    with stats_lock:
        for track_id, data in tracks.items():
            if track_id == 1:
                continue
            # 1\5秒统计
            stats_1_5s.append({
                "1.No": track_id-1,
                "2.1sSpd": f"{data['last_sec_speed']/1024:.2f} MB/s",
               # "3.time(1s)": f"{data['last_sec_time'] * 1000:.2f} ms",
               # "4.bytes(1s)": f"{data['last_sec_bytes'] / 1024:.2f} KB",
                "5.cat": data["category"],
                "6.5sSpd": f"{data['five_sec_speed']/1024:.2f} MB/s",
                #"7.time(5s)": f"{data['five_sec_time'] * 1000:.2f} ms",
                #"8.bytes(5s)": f"{data['five_sec_bytes'] / 1024:.2f} KB",
                "9.5sCat": data["five_category"],
                "10.cnt": f"{data['count']}",
                "11.totKB": f"{data['total_bytes'] / 1024:.2f} KB",
                #"12.total_time": f"{data['total_time'] * 1000:.2f} ms",
                "13.avgLat": f"{data['avg_latency'] * 1000:.2f} ms",
                "14.avgSpd": f"{data['total_avg_speed']/1024:.2f} MB/s"
            })

        for category, data in stats.items():
            if category == "other":
                continue

            stats_total.append({
                "1.category": category,
                "2.count":f"{data['count']}",
                "3.total_bytes": f"{data['total_bytes'] / 1024:.2f} KB",
                "4.total_time": f"{data['total_time'] * 1000:.2f} ms",
                "5.avg_speed": f"{data['total_avg_speed']/1024:.2f} MB/s",
                "6.avg_latency": f"{data['avg_latency']*1000:.2f} ms"
            })

    return jsonify({
        "1/5s_stats": stats_1_5s,
        "total_stats": stats_total
    })


@app.route('/files/<path:filename>')
def download_file(filename):
    file_path = os.path.join(FILE_DIRECTORY, filename)

    # 检查文件是否存在
    if not os.path.exists(file_path):
        abort(404, description="文件不存在")

    file_size = os.path.getsize(file_path)
    category = get_category(filename)

    # 记录下载开始时间
    start_time = time.time()

    # 使用 send_file 直接返回文件
    try:
        response = send_file(file_path, as_attachment=True, download_name=os.path.basename(filename))
        if "init" in filename or "mpd" in filename:
            # 设置文件大小
            response.headers['Content-Length'] = file_size

            return response
        # 下载完成后进行统计
        end_time = time.time()
        total_time = max(0.001, end_time - start_time)
        string = filename.replace('.', '_')
        parts = string.split("_")
        category,track_id,timestamp =(int(parts[0]),int(parts[3][5:]),int(parts[4]))
        #print(category,track_id,timestamp)


        with stats_lock:
            tracks[track_id]["total_bytes"] += file_size
            tracks[track_id]["last_sec_bytes"] += file_size
            tracks[track_id]["last_sec_time"] += total_time
            tracks[track_id]["total_time"] += total_time
            tracks[track_id]["count"] += 1
            tracks[track_id]["category"] = category

            # 计算总平均速度
            if tracks[track_id]["total_time"] > 0:
                tracks[track_id]["total_avg_speed"] = (tracks[track_id]["total_bytes"] / tracks[track_id][
                    "total_time"]) / 1024

            # 计算平均时延
            if tracks[track_id]["count"] > 0:
                tracks[track_id]["avg_latency"] = tracks[track_id]["total_time"] / tracks[track_id]["count"]
            stats[category]["total_bytes"] += file_size
            stats[category]["total_time"] += total_time
            stats[category]["count"] += 1
            if stats[category]["count"] > 0:
                stats[category]["avg_latency"] = stats[category]["total_time"] / stats[category]["count"]

        # 设置文件大小
        response.headers['Content-Length'] = file_size

        return response
    except Exception as e:
        print(f"文件读取错误: {str(e)}")
        abort(500, description="文件读取失败")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10086, debug=True)
