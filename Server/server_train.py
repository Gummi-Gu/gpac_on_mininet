import os
import time
import threading
from flask import Flask, Response, stream_with_context, abort, jsonify, send_file
from collections import defaultdict

app = Flask(__name__)
FILE_DIRECTORY = 'files'  # 文件存储目录
CHUNK_SIZE = 2048  # 每次传输的块大小 (2KB)
STATS_5S_INTERVAL = 5
# 确保文件目录存在
os.makedirs(FILE_DIRECTORY, exist_ok=True)
@app.route('/<client_id>/files/<path:filename>')
def download_file(client_id,filename):
    file_path = os.path.join(FILE_DIRECTORY, filename)
    if client_id not in ['01', '02']:
        abort(404,"非法客户端")
    if not os.path.exists(file_path):
        abort(404, description="文件不存在")

    file_size = os.path.getsize(file_path)
    start_time = time.time()
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

    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10086, debug=True)
