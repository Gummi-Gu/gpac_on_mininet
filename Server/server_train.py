import os
from flask import Flask, abort, send_file
from werkzeug.serving import make_server, ThreadedWSGIServer
from concurrent.futures import ThreadPoolExecutor  # 新增线程池库
import queue

app = Flask(__name__)
FILE_DIRECTORY = 'files'
os.makedirs(FILE_DIRECTORY, exist_ok=True)
ALLOWED_CLIENTS = {'01', '02'}

# 文件下载路由（保持不变）
@app.route('/<client_id>/files/<path:filename>')
def download_file(client_id, filename):
    if client_id not in ALLOWED_CLIENTS:
        abort(403, description="非法客户端")

    # 构造完整路径
    file_path = os.path.abspath(os.path.join(FILE_DIRECTORY, filename))

    # 确保路径没有逃出目标目录
    if not file_path.startswith(os.path.abspath(FILE_DIRECTORY) + os.sep):
        abort(403, description="非法路径")

    if not os.path.isfile(file_path):
        abort(404, description="文件不存在")

    return send_file(
        file_path,
        as_attachment=True,
        download_name=os.path.basename(file_path),
        max_age=0
    )

# 自定义支持线程池的服务器类
class ThreadPoolWSGIServer(ThreadedWSGIServer):
    """
    线程池优化版服务器，继承自 ThreadedWSGIServer
    - 使用固定大小线程池处理请求
    - 通过队列控制最大并发数
    """
    def __init__(self, host, port, app, max_workers=100):
        super().__init__(host, port, app)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.request_queue = queue.Queue(maxsize=1000)  # 防止队列无限增长

    def process_request(self, request, client_address):
        """
        重写请求处理方法：将请求提交到线程池
        """
        try:
            self.request_queue.put_nowait((request, client_address))
            self.executor.submit(self._process_request_thread, request, client_address)
        except queue.Full:
            print("请求队列已满，拒绝连接")
            self.close_request(request)

    def _process_request_thread(self, request, client_address):
        """
        实际处理请求的线程方法
        """
        try:
            super().process_request(request, client_address)
        finally:
            try:
                self.request_queue.get_nowait()
            except queue.Empty:
                pass

class ServerThread:
    def __init__(self, host='0.0.0.0', port=10086):
        self.server = ThreadPoolWSGIServer(  # 使用自定义服务器类
            host,
            port,
            app,
            max_workers=160  # 根据硬件调整（CPU核心数*10~20）
        )
        print(f"启动线程池服务器：http://{host}:{port}")

    def serve_forever(self):
        self.server.serve_forever()

if __name__ == '__main__':
    server = ServerThread()
    server.serve_forever()