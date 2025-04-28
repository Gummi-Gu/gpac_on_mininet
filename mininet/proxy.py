import re
import logging
import sys
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
bitrate_stats = {}
stats_lock = threading.Lock()
client_id=''
current_second = int(time.time())
current_second_bytes = 0
second_stats = {}  # 保存每秒出口总量
app = Flask(__name__)
CORS(app, origins="*")

streamingMonitorClient = util.StreamingMonitorClient('http://192.168.3.22:5000')


def parse_track_id(filename):
    """从文件名解析轨道ID"""
    match = re.search(r'track(\d+)', filename)
    return match.group(1) if match else 'default'


def parse_bitrate(filename):
    """从文件名解析比特率"""
    match = re.match(r'(\d+)_', filename)
    return match.group(1) if match else 'default'


def update_stats(track_id, bitrate, duration, size):
    """更新轨道ID和比特率的统计信息"""
    with stats_lock:
        # ===== 这里处理出口总量统计 =====
        global current_second, current_second_bytes, second_stats
        now = int(time.time())
        if now != current_second:
            # 时间到达了新的秒数，记录上一个秒的数据量
            second_stats[current_second] = current_second_bytes
            # 可选：你可以在这里打印或者上传每秒出口量
            # print(f"Second {current_second}: {current_second_bytes} bytes")
            streamingMonitorClient.submit_summary_rate_stats({'client1':current_second_bytes})
            current_second = now
            current_second_bytes = 0
        current_second_bytes += size
        # 更新track_id统计
        if track_id not in track_stats:
            track_stats[track_id] = {
                'total_delay': 0.0,
                'total_size': 0,
                'count': 0,
                'latest_delay': 0.0,
                'latest_size': 0
            }

        tstats = track_stats[track_id]
        tstats['total_delay'] += duration * 1e3
        tstats['total_size'] += size
        tstats['count'] += 1
        tstats['latest_delay'] = duration * 1e3
        tstats['latest_size'] = size

        avg_delay = tstats['total_delay'] / tstats['count'] if tstats['count'] > 0 else 0
        avg_rate = (tstats['total_size'] / tstats['total_delay'] / 1e3) if tstats['total_delay'] > 0 else 0
        latest_rate = (tstats['latest_size'] / tstats['latest_delay'] / 1e3) if tstats['latest_delay'] > 0 else 0

        streamingMonitorClient.submit_track_stats(track_id, client_id, avg_delay, avg_rate, tstats['latest_delay'], latest_rate)

        # 更新bitrate统计
        if bitrate not in bitrate_stats:
            bitrate_stats[bitrate] = {
                'total_delay': 0.0,
                'total_size': 0,
                'count': 0,
                'latest_delay': 0.0,
                'latest_size': 0
            }

        bstats = bitrate_stats[bitrate]
        bstats['total_delay'] += duration * 1e3
        bstats['total_size'] += size
        bstats['count'] += 1
        bstats['latest_delay'] = duration * 1e3
        bstats['latest_size'] = size

        avg_delay = bstats['total_delay'] / bstats['count'] if bstats['count'] > 0 else 0
        avg_rate = (bstats['total_size'] / bstats['total_delay'] / 1e3) if bstats['total_delay'] > 0 else 0
        latest_rate = (bstats['latest_size'] / bstats['latest_delay'] / 1e3) if bstats['latest_delay'] > 0 else 0

        # 提交 bitrate 的统计
        streamingMonitorClient.submit_bitrate_stats(bitrate, client_id, avg_delay, avg_rate, bstats['latest_delay'], latest_rate)

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

            for bitrate, stats in bitrate_stats.items():
                avg_delay = stats['total_delay'] / stats['count'] if stats['count'] > 0 else 0
                avg_rate = (stats['total_size'] / stats['total_delay'] / 1e3) if stats['total_delay'] > 0 else 0
                latest_rate = (stats['latest_size'] / stats['latest_delay'] / 1e3) if stats['latest_delay'] > 0 else 0

                logger.info(
                    f"BITRATE {bitrate} STATS: "
                    f"AvgDelay={avg_delay:.2f}ms AvgRate={avg_rate:.2f}MB/s | "
                    f"LatestDelay={stats['latest_delay']:.2f}ms LatestRate={latest_rate:.2f}MB/s"
                )


@app.route('/<path:path>', methods=['GET', 'HEAD'])
def proxy(path):
    try:
        # 解析轨道ID和bitrate
        filename = path.split('/')[-1]
        track_id = parse_track_id(filename)
        bitrate = parse_bitrate(filename)

        # 请求计时
        start_time = time.time()

        # 转发请求
        resp = requests.request(
            method=request.method,
            url=f"http://{app.config['TARGET_SERVER']}:{app.config['TARGET_PORT']}/{path}",
            headers={k: v for k, v in request.headers if k.lower() not in ['host', 'accept-encoding']},
            stream=True,
            proxies={'http': None, 'https': None}  # 显式禁用系统代理
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
                update_stats(track_id, bitrate, duration, final_size)
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

    # 启动日志线程
    #threading.Thread(target=log_statistics, daemon=True).start()

    # 启动服务
    from werkzeug.serving import ThreadedWSGIServer
    server = ThreadedWSGIServer('0.0.0.0', proxy_port, app)
    logger.info(f"Proxy server started on :{proxy_port}")
    server.serve_forever()


if __name__ == '__main__':
    client_id = sys.argv[1]
    run_server("10.0.0.1", 10086, 10086)
