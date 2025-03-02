import time
import curses
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# 共享数据存储
monitor_data = {
    "buffer_length": 0.0,
    "max_buffer_length": 10.0,
    "download_speed": 0.0,
    "slices": []  # 存储多个视频切片的信息
}

@app.route('/update', methods=['POST'])
def update_data():
    global monitor_data
    data = request.json

    # 只有 buffer_length > 0 时才更新
    new_buffer_length = data.get("buffer_length")
    if new_buffer_length and new_buffer_length > 0:
        monitor_data["buffer_length"] = new_buffer_length

    new_buffer_max = data.get("max_buffer_length")
    if new_buffer_max and new_buffer_max > 0:
        monitor_data["max_buffer_length"] = new_buffer_max

    monitor_data['download_speed'] = data.get("download_speed")

    # 处理 slices 合并逻辑
    new_slices = data.get("slices", [])
    existing_slices = {s["idx"]: s for s in monitor_data["slices"]}

    for new_slice in new_slices:
        idx = new_slice["idx"]
        if idx in existing_slices:
            existing_slices[idx].update(new_slice)  # 更新已有的 slice
        else:
            existing_slices[idx] = new_slice  # 添加新的 slice

    # 转换回列表存储
    monitor_data["slices"] = list(existing_slices.values())

    return jsonify({"status": "updated"})

@app.route('/get_status', methods=['GET'])
def get_status():
    return jsonify(monitor_data)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
