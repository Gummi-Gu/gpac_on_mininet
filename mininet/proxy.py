import re
import logging
import sys
import time
import threading
import queue
from flask import Flask, request, Response
from flask_cors import CORS
import requests
import util
from werkzeug.serving import ThreadedWSGIServer

# 预编译正则表达式
TRACK_PATTERN = re.compile(r'track(\d+)')
BITRATE_PATTERN = re.compile(r'^(\d+)_')

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# 全局配置
BUFFER_SIZE = 1024 * 1024  # 1MB缓冲块
STATS_BATCH_SIZE = 100  # 批量处理统计事件数
CONNECTION_POOL_SIZE = 100  # 连接池大小

# 共享数据结构
track_stats = {}
bitrate_stats = {}
stats_lock = threading.Lock()
client_id = ''
current_second = int(time.time())
current_second_bytes = 0
current_second_time = 0
app = Flask(__name__)
CORS(app, origins="*")

# 监控客户端和异步队列
streamingMonitorClient = util.StreamingMonitorClient('http://192.168.3.22:5000')
stats_queue = queue.Queue(maxsize=10000)


class ConnectionPool:
    """线程安全的连接池管理"""

    def __init__(self):
        self.pool = {}
        self.lock = threading.Lock()

    def get_session(self):
        """获取线程专属的Session"""
        thread_id = threading.get_ident()
        with self.lock:
            if thread_id not in self.pool:
                session = requests.Session()
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=CONNECTION_POOL_SIZE,
                    pool_maxsize=CONNECTION_POOL_SIZE
                )
                session.mount('http://', adapter)
                session.mount('https://', adapter)
                self.pool[thread_id] = session
            return self.pool[thread_id]


connection_pool = ConnectionPool()


def parse_track_id(filename):
    match = TRACK_PATTERN.search(filename)
    return match.group(1) if match else 'default'


def parse_bitrate(filename):
    match = BITRATE_PATTERN.match(filename)
    return match.group(1) if match else 'default'


def update_stats(track_id, bitrate, duration, size):
    """非阻塞提交统计事件"""
    try:
        stats_queue.put_nowait((
            track_id,
            bitrate,
            duration * 1000,  # 转换为毫秒
            size,
            time.time()
        ))
    except queue.Full:
        logger.warning("Dropping stats event due to full queue")


def process_stats_batch(batch):
    """批量处理统计更新"""
    global current_second, current_second_bytes, current_second_time

    time_map = {}
    track_map = {}
    bitrate_map = {}

    # 聚合数据
    for track_id, bitrate, delay_ms, size, timestamp in batch:
        # 按时间聚合
        sec = int(timestamp)
        if sec not in time_map:
            time_map[sec] = {'bytes': 0, 'delay': 0}
        time_map[sec]['bytes'] += size
        time_map[sec]['delay'] += delay_ms

        # 按轨道聚合
        if track_id not in track_map:
            track_map[track_id] = {
                'count': 0, 'total_delay': 0, 'total_size': 0,
                'latest_delay': 0, 'latest_size': 0
            }
        track = track_map[track_id]
        track['count'] += 1
        track['total_delay'] += delay_ms
        track['total_size'] += size
        track['latest_delay'] = delay_ms
        track['latest_size'] = size

        # 按码率聚合
        if bitrate not in bitrate_map:
            bitrate_map[bitrate] = {
                'count': 0, 'total_delay': 0, 'total_size': 0,
                'latest_delay': 0, 'latest_size': 0
            }
        br = bitrate_map[bitrate]
        br['count'] += 1
        br['total_delay'] += delay_ms
        br['total_size'] += size
        br['latest_delay'] = delay_ms
        br['latest_size'] = size

    # 更新全局统计
    with stats_lock:
        # 处理时间统计
        for sec, data in time_map.items():
            if sec != current_second:
                if current_second_bytes > 0:
                    streamingMonitorClient.submit_summary_rate_stats({
                        client_id: {
                            'size': current_second_bytes / 1e6,
                            'time': current_second_time
                        }
                    })
                    current_second = sec
                    current_second_bytes = 0
                    current_second_time = 0
            current_second_bytes += data['bytes']
            current_second_time += data['delay']

        # 更新轨道统计
        for track_id, data in track_map.items():
            if track_id not in track_stats:
                track_stats[track_id] = data
            else:
                track = track_stats[track_id]
                track['count'] += data['count']
                track['total_delay'] += data['total_delay']
                track['total_size'] += data['total_size']
                track['latest_delay'] = data['latest_delay']
                track['latest_size'] = data['latest_size']

            # 计算并提交
            track = track_stats[track_id]
            avg_delay = track['total_delay'] / track['count']
            avg_rate = track['total_size'] / track['total_delay'] if track['total_delay'] else 0
            streamingMonitorClient.submit_track_stats(
                track_id, client_id,
                avg_delay,
                avg_rate,
                track['latest_delay'],
                track['latest_size'] / track['latest_delay'] if track['latest_delay'] else 0
            )

        # 更新码率统计
        for bitrate, data in bitrate_map.items():
            if bitrate not in bitrate_stats:
                bitrate_stats[bitrate] = data
            else:
                br = bitrate_stats[bitrate]
                br['count'] += data['count']
                br['total_delay'] += data['total_delay']
                br['total_size'] += data['total_size']
                br['latest_delay'] = data['latest_delay']
                br['latest_size'] = data['latest_size']

            # 计算并提交
            br = bitrate_stats[bitrate]
            avg_delay = br['total_delay'] / br['count']
            avg_rate = br['total_size'] / br['total_delay'] if br['total_delay'] else 0
            streamingMonitorClient.submit_bitrate_stats(
                bitrate, client_id,
                avg_delay,
                avg_rate,
                br['latest_delay'],
                br['latest_size'] / br['latest_delay'] if br['latest_delay'] else 0
            )


def stats_consumer():
    """批量消费统计事件"""
    batch = []
    while True:
        try:
            item = stats_queue.get(timeout=1)
            batch.append(item)
            if len(batch) >= STATS_BATCH_SIZE:
                process_stats_batch(batch)
                batch = []
            stats_queue.task_done()
        except queue.Empty:
            if batch:
                process_stats_batch(batch)
                batch = []
        except Exception as e:
            logger.error(f"Stats processing error: {str(e)}")


@app.route('/<path:path>', methods=['GET', 'HEAD'])
def proxy(path):
    try:
        filename = path.split('/')[-1]
        track_id = parse_track_id(filename)
        bitrate = parse_bitrate(filename)
        start_time = time.time()

        # 复用连接池
        resp = connection_pool.get_session().request(
            method=request.method,
            url=f"http://{app.config['TARGET_SERVER']}:{app.config['TARGET_PORT']}/{path}",
            headers={k: v for k, v in request.headers if k.lower() not in ['host', 'accept-encoding']},
            stream=True,
            proxies={'http': None, 'https': None}
        )

        # 流式响应
        def generate():
            actual_size = 0
            try:
                for chunk in resp.iter_content(BUFFER_SIZE):
                    if chunk:
                        actual_size += len(chunk)
                        yield chunk
            finally:
                resp.close()
                duration = time.time() - start_time
                declared_size = int(resp.headers.get('Content-Length', actual_size))
                update_stats(track_id, bitrate, duration, declared_size)
                logger.debug(f"Served {filename} ({declared_size // 1024}KB) in {duration:.2f}s")

        return Response(generate(), headers=dict(resp.headers.items()))

    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        return Response("Bad Gateway", status=502)


def run_server(target, port, proxy_port):
    app.config['TARGET_SERVER'] = target
    app.config['TARGET_PORT'] = port

    # 启动统计消费者线程池
    for _ in range(4):
        t = threading.Thread(target=stats_consumer, daemon=True)
        t.start()

    # 配置高性能服务器
    server = ThreadedWSGIServer(
        '0.0.0.0',
        proxy_port,
        app,
        threaded=True,
        processes=4  # 根据CPU核心数调整
    )
    logger.info(f"Optimized proxy running on :{proxy_port}")
    server.serve_forever()


if __name__ == '__main__':
    client_id = sys.argv[1]
    run_server("10.0.0.1", 10086, 10086)