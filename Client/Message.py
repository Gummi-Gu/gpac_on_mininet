import threading
import queue
import requests
import time


def custom_error_handler(e, data, retries):
    print(f"[错误] 异常: {str(e)}")
    print(f"    最后尝试数据: {data}")
    print(f"    已重试次数: {retries}")

class ThreadedCommunication:
    def __init__(self, url=None, send_func=None, error_callback=None,
                 max_queue_size=0, daemon=False, headers=None, timeout=10,
                 retries=3, retry_delay=1):
        """
        基于requests的线程通信类

        :param url: 目标服务地址（当使用内置HTTP发送时必需）
        :param send_func: 自定义发送函数（优先级高于url，需接受data参数）
        :param error_callback: 异常处理回调函数（参数：exception, data, retries）
        :param headers: HTTP请求头
        :param timeout: 请求超时时间（秒）
        :param retries: 失败重试次数
        :param retry_delay: 重试间隔时间（秒）
        """
        if headers is None:
            headers = {'Content-Type': 'application/json'}
        if error_callback is None:
            error_callback = custom_error_handler
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._running = False
        self._thread = None
        self._daemon = daemon
        self.url = url
        self.timeout = timeout
        self.headers = headers
        self.retries = retries
        self.retry_delay = retry_delay

        # 确定发送方式优先级
        if send_func:
            self._send_func = send_func
        elif url:
            self._send_func = self._http_send
        else:
            self._send_func = self.default_send

        self._error_callback = error_callback

    def default_send(self, data):
        """默认发送实现（打印数据）"""
        print(f"[默认发送] 数据已发送: {data}")

    def _http_send(self, data):
        """内置的requests发送实现"""
        for attempt in range(self.retries + 1):
            try:
                response = requests.post(
                    self.url,
                    json=data,
                    headers=self.headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                #print('[Message]已成功发送', response.status_code)
                return  # 成功则退出
            except requests.exceptions.RequestException as e:
                if attempt < self.retries:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise RuntimeError(f"请求失败（尝试{self.retries + 1}次）: {str(e)}") from e

    def start(self):
        """启动通信线程"""
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._run_loop)
            self._thread.daemon = self._daemon
            self._thread.start()

    def stop(self, wait=True):
        """停止通信线程"""
        self._running = False
        if wait:
            self._queue.join()
        if self._thread and self._thread.is_alive():
            self._thread.join()

    def send(self, data, block=True, timeout=None):
        """发送数据到队列"""
        try:
            self._queue.put(data, block=block, timeout=timeout)
            #print("[Message]请求已进入队列")
        except queue.Full:
            if self._error_callback:
                self._error_callback(queue.Full("队列已满"), data, 0)
            else:
                raise

    def _run_loop(self):
        """线程运行主循环"""
        while self._running or not self._queue.empty():
            try:
                data = self._queue.get(timeout=0.1)
                try:
                    self._send_func(data)
                except Exception as e:
                    if self._error_callback:
                        self._error_callback(e, data, self.retries)
                    else:
                        print(f"发送失败: {str(e)} | 数据: {data}")
                finally:
                    self._queue.task_done()
            except queue.Empty:
                continue

    @property
    def is_running(self):
        return self._running

    @property
    def queue_size(self):
        return self._queue.qsize()


# 使用示例
def test_func():
    # 错误处理回调（带重试信息）



    # 初始化通信实例
    comm = ThreadedCommunication(
        url="http://httpbin.org/post",  # 测试URL
        headers={'X-Custom-Header': 'value'},
        timeout=5,
        retries=2,
        retry_delay=1,
        error_callback=custom_error_handler(),
        max_queue_size=100
    )

    # 启动通信线程
    comm.start()

    # 发送测试数据
    sample_payload = {
        "device_id": "sensor-001",
        "timestamp": int(time.time()),
        "values": [23.5, 65.2, 1024]
    }

    # 发送10条测试数据
    for i in range(10):
        comm.send({**sample_payload, "sequence": i + 1})
        time.sleep(0.1)  # 模拟数据产生间隔

    # 等待队列处理
    time.sleep(3)

    # 停止通信
    comm.stop()
    print(f"剩余队列数量: {comm.queue_size}")