#!/usr/bin/env python3
import os

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import Controller
from mininet.link import TCLink
from mininet.log import setLogLevel, info
from mininet.cli import CLI
import time
import threading
import random

# ================== 配置参数 ==================
SERVER_IP = '10.0.0.1'
DASH_DIR = '/tmp/dash'
REQUEST_INTERVAL = 5  # 新请求间隔（秒）
TRAFFIC_CLASSES = {
    'high': {'mark': 10, 'rate': '1mbit', 'ceil': '1mbit', 'classid': '1:10'},
    'low': {'mark': 20, 'rate': '500kbit', 'ceil': '500kbit', 'classid': '1:20'}
}


# ==============================================

class DynamicTopo(Topo):
    def __init__(self):
        Topo.__init__(self)
        server = self.addHost('server', ip=SERVER_IP)
        client = self.addHost('client')
        switch = self.addSwitch('s1')
        self.addLink(server, switch, cls=TCLink, bw=10)
        self.addLink(client, switch, cls=TCLink, bw=10)


class TrafficControl:
    @staticmethod
    def setup_tc(server):
        cmds = [
            'tc qdisc del dev server-eth0 root 2>/dev/null',
            'tc qdisc add dev server-eth0 root handle 1: htb',
            'tc class add dev server-eth0 parent 1: classid 1:1 htb rate 10mbit',
            f'tc class add dev server-eth0 parent 1:1 classid {TRAFFIC_CLASSES["high"]["classid"]} htb rate {TRAFFIC_CLASSES["high"]["rate"]}',
            f'tc class add dev server-eth0 parent 1:1 classid {TRAFFIC_CLASSES["low"]["classid"]} htb rate {TRAFFIC_CLASSES["low"]["rate"]}',
            'tc filter add dev server-eth0 parent 1: protocol ip handle 10 fw flowid 1:10',
            'tc filter add dev server-eth0 parent 1: protocol ip handle 20 fw flowid 1:20',
            'tc qdisc add dev server-eth0 parent 1:10 sfq perturb 10',
            'tc qdisc add dev server-eth0 parent 1:20 sfq perturb 10'
        ]
        for cmd in cmds:
            server.cmd(cmd)
        server.cmd(
            'iptables -t mangle -A PREROUTING -m string --string "GET /high/" --algo bm --from 60 -j MARK --set-mark 10')
        server.cmd(
            'iptables -t mangle -A PREROUTING -m string --string "GET /low/" --algo bm --from 60 -j MARK --set-mark 20')


def setup_server(server):
    server.cmd(f'mkdir -p {DASH_DIR}/high {DASH_DIR}/low')
    server.cmd(f'dd if=/dev/urandom of={DASH_DIR}/high/chunk1.m4s bs=1M count=100')
    server.cmd(f'dd if=/dev/urandom of={DASH_DIR}/low/chunk1.m4s bs=1M count=50')
    server.cmd(f'cd {DASH_DIR} && python3 -m http.server 80 &')


class RequestGenerator:
    def __init__(self, client):
        self.client = client
        self.running = True
        self.active_requests = {'high': 0, 'low': 0}
        self.lock = threading.Lock()

    def _fetch(self, url_type):
        """执行单个请求并更新状态"""
        url = f'http://{SERVER_IP}/{url_type}/chunk1.m4s'
        try:
            with self.lock:
                self.active_requests[url_type] += 1
            self.client.cmd(f'curl -s {url} > /dev/null')
        finally:
            with self.lock:
                self.active_requests[url_type] -= 1

    def _generate_requests(self):
        """持续生成随机请求"""
        while self.running:
            url_type = random.choice(['high', 'low'])
            thread = threading.Thread(target=self._fetch, args=(url_type,))
            thread.start()
            time.sleep(random.uniform(0.1, REQUEST_INTERVAL))

    def start(self):
        """启动请求生成器"""
        self.thread = threading.Thread(target=self._generate_requests)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """停止所有请求"""
        self.running = False
        self.thread.join()


class TrafficMonitor:
    def __init__(self, server, request_gen):
        self.server = server
        self.request_gen = request_gen
        self.running = True

    def _get_tc_stats(self):
        """获取带宽统计信息"""
        stats = {
            'high': self.server.cmd('tc -s class show dev server-eth0 | grep "1:10"'),
            'low': self.server.cmd('tc -s class show dev server-eth0 | grep "1:20"')
        }
        return stats

    def _display(self):
        """实时显示监控信息"""
        while self.running:
            os.system('clear')
            stats = self._get_tc_stats()
            print("\n=== Server bandwidth statistics ===")
            print(f"[High] Active Requests: {self.request_gen.active_requests['high']}")
            print(stats['high'])
            print(f"\n[Low] Active Requests: {self.request_gen.active_requests['low']}")
            print(stats['low'])
            print("\npress Ctrl+C to stop...")
            time.sleep(2)

    def start(self):
        """启动监控"""
        self.thread = threading.Thread(target=self._display)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """停止监控"""
        self.running = False
        self.thread.join()


if __name__ == '__main__':
    setLogLevel('error')
    net = Mininet(topo=DynamicTopo(), controller=Controller)
    net.start()
    server, client = net.get('server', 'client')

    try:
        setup_server(server)
        TrafficControl.setup_tc(server)

        # 初始化请求生成器和监控器
        request_gen = RequestGenerator(client)
        monitor = TrafficMonitor(server, request_gen)

        # 启动线程
        request_gen.start()
        monitor.start()

        # 保持主线程运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        info("\nstop...\n")
        request_gen.stop()
        monitor.stop()
    finally:
        net.stop()