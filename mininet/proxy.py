import re
import logging
import time
import threading
from flask import Flask, request, Response
from flask_cors import CORS
import requests
import util

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# 全局统计数据
track_stats = {}
stats_lock = threading.Lock()

app = Flask(__name__)
CORS(app, origins="*")

streamingMonitorClient=util.StreamingMonitorClient('http://192.168.3.22:5000')


def parse_track_id(filename):
    """从文件名解析轨道ID"""
    match = re.search(r'track(\d+)', filename)
    return match.group(1) if match else 'default'


def update_stats(track_id, duration, size):
    """更新统计信息"""
    with stats_lock:
        if track_id not in track_stats:
            track_stats[track_id] = {
                'total_delay': 0.0,
                'total_size': 0,
                'count': 0,
                'latest_delay': 0.0,
                'latest_size': 0
            }

        stats = track_stats[track_id]
        stats['total_delay'] += duration*1e3
        stats['total_size'] += size
        stats['count'] += 1
        stats['latest_delay'] = duration*1e3
        stats['latest_size'] = size
        avg_delay = stats['total_delay'] / stats['count'] if stats['count'] > 0 else 0
        avg_rate = (stats['total_size'] / stats['total_delay'] / 1e3) if stats['total_delay'] > 0 else 0
        latest_rate = (stats['latest_size'] / stats['latest_delay'] / 1e3) if stats['latest_delay'] > 0 else 0
        streamingMonitorClient.submit_track_stats(track_id,"client1",avg_delay,avg_rate,stats['latest_delay'],latest_rate)



def log_statistics():
    """定时输出统计信息"""
    while True:
        time.sleep(5)
        with stats_lock:
            for track_id, stats in track_stats.items():
                avg_delay = stats['total_delay'] / stats['count'] if stats['count'] > 0 else 0
                avg_rate = (stats['total_size'] / stats['total_delay'] / 1e3) if stats['total_delay'] > 0 else 0
                latest_rate = (stats['latest_size'] / stats['latest_delay'] / 1e3) if stats['latest_delay'] > 0 else 0

                logger.info(
                    f"TRACK {track_id} STATS: "
                    f"AvgDelay={avg_delay:.2f}ms AvgRate={avg_rate:.2f}MB/s | "
                    f"LatestDelay={stats['latest_delay']:.2f}ms LatestRate={latest_rate:.2f}MB/s"
                )


@app.route('/<path:path>', methods=['GET', 'HEAD'])
def proxy(path):
    try:
        # 解析轨道ID
        filename = path.split('/')[-1]
        track_id = parse_track_id(filename)

        # 请求计时
        start_time = time.time()

        # 转发请求
        resp = requests.request(
            method=request.method,
            url=f"http://{app.config['TARGET_SERVER']}:{app.config['TARGET_PORT']}/{path}",
            headers={k: v for k, v in request.headers if k.lower() not in ['host', 'accept-encoding']},
            stream=True,
            proxies={'http': None, 'https': None}  # 显式禁用代理
        )

        # 获取声明大小
        declared_size = int(resp.headers.get('Content-Length', 0))

        # 流式响应
        def generate():
            actual_size = 0
            try:
                for chunk in resp.iter_content(128 * 1024):
                    if chunk:
                        actual_size += len(chunk)
                        yield chunk
            finally:
                final_size = declared_size if declared_size > 0 else actual_size
                duration = time.time() - start_time
                update_stats(track_id, duration, final_size)
                resp.close()
                logger.debug(f"Processed {filename} in {duration:.2f}s")

        return Response(generate(), headers={
            k: v for k, v in resp.headers.items()
            if k.lower() not in ['transfer-encoding', 'connection']
        })

    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        return Response("Bad Gateway", status=502)


def run_server(target, port, proxy_port):
    app.config['TARGET_SERVER'] = target
    app.config['TARGET_PORT'] = port


    # 启动服务
    from werkzeug.serving import ThreadedWSGIServer
    server = ThreadedWSGIServer('0.0.0.0', proxy_port, app)
    logger.info(f"Proxy server started on :{proxy_port}")
    server.serve_forever()


if __name__ == '__main__':
    run_server("10.0.0.1", 10086, 10086)