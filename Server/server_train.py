import os
from flask import Flask, abort, send_file, safe_join
from werkzeug.serving import make_server, ThreadedWSGIServer

app = Flask(__name__)
FILE_DIRECTORY = 'files'

# 确保文件目录存在
os.makedirs(FILE_DIRECTORY, exist_ok=True)
ALLOWED_CLIENTS = {'01', '02'}

@app.route('/<client_id>/files/<path:filename>')
def download_file(client_id, filename):
    if client_id not in ALLOWED_CLIENTS:
        abort(403, description="非法客户端")

    # 防止路径穿越
    file_path = safe_join(FILE_DIRECTORY, filename)
    if not file_path or not os.path.isfile(file_path):
        abort(404, description="文件不存在")

    return send_file(
        file_path,
        as_attachment=True,
        download_name=os.path.basename(file_path),
        max_age=0
    )

# 使用 ThreadedWSGIServer 启动高并发服务
class ServerThread:
    def __init__(self, host='0.0.0.0', port=10086):
        self.server = ThreadedWSGIServer(
            host,
            port,
            app # 多线程支持
        )
        self.server.daemon_threads = True  # 线程自动关闭
        #print(f"🚀 服务器已启动: http://{host}:{port}")

    def serve_forever(self):
        self.server.serve_forever()

if __name__ == '__main__':
    server = ServerThread()
    server.serve_forever()
