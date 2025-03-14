#!/usr/bin/env python3
import os
from time import sleep

import requests
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import Controller
from mininet.link import TCLink
from mininet.log import setLogLevel, info
from mininet.cli import CLI  # 导入 Mininet 的 CLI
import time
import threading
import random

# ================== Configuration ==================
SERVER_IP = '10.0.0.1'
PORT=1080
DASH_DIR = '/home/mininet/gpac_on_mininet/mininet/dash'
REQUEST_INTERVAL = 1  # New request interval (seconds)
TRAFFIC_CLASSES = {
    'high': {'mark': 10, 'rate': '50mbit', 'ceil': '50mbit', 'classid': '1:10'},
    'low': {'mark': 20, 'rate': '10mbit', 'ceil': '10mbit', 'classid': '1:20'}
}
FILE_SIZES = {
    'high': 5 * 1024 * 1024,  # 5MB in bytes
    'low': 1 * 1024 * 1024  # 1MB in bytes
}


# ===================================================

class DynamicTopo(Topo):
    def __init__(self):
        Topo.__init__(self)
        server = self.addHost('server', ip='10.0.0.1')
        client = self.addHost('client', ip='10.0.0.2')
        switch = self.addSwitch('s1')
        self.addLink(server, switch, cls=TCLink, bw=1000)
        self.addLink(client, switch, cls=TCLink, bw=1000)


class TrafficControl:
    @staticmethod
    def setup_tc(server):
        cmds = [
            # 清空现有配置
            'tc qdisc del dev server-eth0 root 2>/dev/null',
            # 创建HTB队列
            'tc qdisc add dev server-eth0 root handle 1: htb',
            'tc class add dev server-eth0 parent 1: classid 1:1 htb rate 200mbit',
            # 创建子类（保持原带宽设置）
            f'tc class add dev server-eth0 parent 1:1 classid {TRAFFIC_CLASSES["high"]["classid"]} htb rate {TRAFFIC_CLASSES["high"]["rate"]} ceil {TRAFFIC_CLASSES["high"]["ceil"]}',
            f'tc class add dev server-eth0 parent 1:1 classid {TRAFFIC_CLASSES["low"]["classid"]} htb rate {TRAFFIC_CLASSES["low"]["rate"]} ceil {TRAFFIC_CLASSES["low"]["ceil"]}',
            # 创建过滤器
            'tc filter add dev server-eth0 parent 1: protocol ip handle 10 fw flowid 1:10',
            'tc filter add dev server-eth0 parent 1: protocol ip handle 20 fw flowid 1:20'
        ]
        # 设置连接标记规则
        connmark_cmds = [
            # 对入口请求打连接标记
            'iptables -t mangle -A PREROUTING -p tcp --dport 1080 -m string --algo kmp --string "high" -j CONNMARK --set-mark 10',
            'iptables -t mangle -A PREROUTING -p tcp --dport 1080 -m string --algo kmp --string "low" -j CONNMARK --set-mark 20',
            #'iptables -t mangle -A PREROUTING -p tcp --dport 1080 -j CONNMARK --set-mark 10',
            #'iptables -t mangle -A PREROUTING -p tcp --dport 1080 -m string --string "/high/" --algo bm --from 60 -j CONNMARK --set-mark 10',
            #'iptables -t mangle -A PREROUTING -p tcp --dport 1080 -m string --string "/low/" --algo bm --from 60 -j CONNMARK --set-mark 20',
            # 出口方向恢复数据包标记
            'iptables -t mangle -A OUTPUT -p tcp --sport 1080 -j CONNMARK --restore-mark'
        ]

        for cmd in cmds + connmark_cmds:
            server.cmd(cmd)

    @staticmethod
    def adjust_bandwidth(server, class_type, rate, ceil):
        class_config = TRAFFIC_CLASSES[class_type]
        cmd = f'tc class change dev server-eth0 parent 1:1 classid {class_config["classid"]} htb rate {rate} ceil {ceil}'
        server.cmd(cmd)


def setup_server(server):
    server.cmd(f'mkdir -p {DASH_DIR}/high {DASH_DIR}/low')
    server.cmd(f'dd if=/dev/urandom of={DASH_DIR}/high/chunk1.m4s bs=1M count=5')
    server.cmd(f'dd if=/dev/urandom of={DASH_DIR}/low/chunk1.m4s bs=1M count=1')
    server.cmd(f'cd {DASH_DIR}')
    server.cmd(f'screen -dmS flask_server python3 {DASH_DIR}/test.py')


class RequestGenerator:
    def __init__(self, client):
        self.client = client
        self.running = True
        self.active_requests = {'high': 0, 'low': 0}
        self.completed_requests = {'high': 0, 'low': 0}
        self.total_transfer_time = {'high': 0.0, 'low': 0.0}
        self.total_data = {'high': 0, 'low': 0}
        self.last_transfer_time = {'high': 0.0, 'low': 0.0}
        self.lock = threading.Lock()

    def _fetch(self, url_type):
        """Execute single request and update metrics"""
        url = f'http://{SERVER_IP}:{PORT}/{url_type}/chunk1.m4s'

        try:
            with self.lock:
                self.active_requests[url_type] += 1

            start_time = time.time()
            self.client.cmd(f'curl -v {url} > /dev/null')
            duration = time.time() - start_time

            with self.lock:
                self.active_requests[url_type] -= 1
                self.completed_requests[url_type] += 1
                self.total_transfer_time[url_type] += duration
                self.total_data[url_type] += FILE_SIZES[url_type]
                self.last_transfer_time[url_type] = duration


        except Exception as e:

            print(f"Unknown failure: {type(e).__name__}, message: {str(e)}")

        finally:

            with self.lock:
                if self.active_requests[url_type] > 0:
                    self.active_requests[url_type] -= 1

    def _generate_requests(self):
        """Generate random requests continuously"""
        self.client.cmd(f'ping {SERVER_IP} -c 5')
        while self.running:
            url_type = random.choice(['high', 'low'])
            #threading.Thread(target=self._fetch, args=(url_type,)).start()
            self._fetch(url_type)
            time.sleep(random.uniform(1, REQUEST_INTERVAL))

    def start(self):
        """Start request generator"""
        self.thread = threading.Thread(target=self._generate_requests)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop all requests"""
        self.running = False
        self.thread.join()


class TrafficMonitor:
    def __init__(self, server, request_gen):
        self.server = server
        self.request_gen = request_gen
        self.running = True

    def _get_tc_stats(self):
        """Get bandwidth statistics"""
        return {
            'high': self.server.cmd('tc -s class show dev server-eth0 | grep "1:10"'),
            'low': self.server.cmd('tc -s class show dev server-eth0 | grep "1:20"')
        }

    def _format_speed(self, speed_bps):
        """Format speed display"""
        if speed_bps >= 1e6:
            return f"{speed_bps / 1e6:.2f} Mbps"
        elif speed_bps >= 1e3:
            return f"{speed_bps / 1e3:.2f} Kbps"
        return f"{speed_bps:.2f} bps"

    def _display(self):
        """Real-time monitoring display"""
        while self.running:
            os.system('clear')
            stats = self._get_tc_stats()

            with self.request_gen.lock:
                high_active = self.request_gen.active_requests['high']
                low_active = self.request_gen.active_requests['low']
                high_completed = self.request_gen.completed_requests['high']
                low_completed = self.request_gen.completed_requests['low']
                high_total_time = self.request_gen.total_transfer_time['high']
                low_total_time = self.request_gen.total_transfer_time['low']
                high_total_data = self.request_gen.total_data['high']
                low_total_data = self.request_gen.total_data['low']
                high_last_time = self.request_gen.last_transfer_time['high']
                low_last_time = self.request_gen.last_transfer_time['low']

            # Calculate metrics
            high_avg_speed = (high_total_data / max(0.1,high_total_time) * 8) if high_total_time > 0 else 0
            low_avg_speed = (low_total_data / max(0.1,low_total_time) * 8) if low_total_time > 0 else 0

            high_last_speed = FILE_SIZES['high'] / max(0.1,high_last_time) * 8 if high_last_time > 0 else 0
            low_last_speed = FILE_SIZES['low'] / max(0.1,low_last_time) * 8 if low_last_time > 0 else 0

            high_avg_latency = (high_total_time / high_completed * 1000) if high_completed > 0 else 0
            low_avg_latency = (low_total_time / low_completed * 1000) if low_completed > 0 else 0

            high_total_mb = high_total_data / (1024 * 1024)
            low_total_mb = low_total_data / (1024 * 1024)
            total_mb = high_total_mb + low_total_mb

            # Display metrics
            print("\n=== Real-time Network Monitoring ===")
            print("=== Traffic Class Statistics ===")
            print(f"[HIGH] Active: {high_active}\tCompleted: {high_completed}")
            print(f"       Avg Speed: {self._format_speed(high_avg_speed)}")
            print(f"       Avg Latency: {high_avg_latency:.2f} ms")
            print(f"       Last Speed: {self._format_speed(high_last_speed)}")
            print(f"       Total Data: {high_total_mb:.2f} MB")
            print("   TC Statistics:")
            print(stats['high'])

            print(f"\n[LOW] Active: {low_active}\tCompleted: {low_completed}")
            print(f"      Avg Speed: {self._format_speed(low_avg_speed)}")
            print(f"      Avg Latency: {low_avg_latency:.2f} ms")
            print(f"      Last Speed: {self._format_speed(low_last_speed)}")
            print(f"      Total Data: {low_total_mb:.2f} MB")
            print("  TC Statistics:")
            print(stats['low'])

            print("\n=== Global Statistics ===")
            print(f"Total Transferred: {total_mb:.2f} MB")
            print(f"Total Bandwidth: {self._format_speed(high_avg_speed + low_avg_speed)}")
            print("\nPress Ctrl+C to exit...")
            user_input = input("\nPress Enter continue ")


    def start(self):
        """Start monitoring"""
        self.thread = threading.Thread(target=self._display)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop monitoring"""
        self.running = False
        self.thread.join()


if __name__ == '__main__':
    #setLogLevel('error')
    net = Mininet(topo=DynamicTopo(), controller=Controller)
    net.start()
    server, client = net.get('server', 'client')

    try:
        client.cmd('screen -dm bash')  # 启动一个新的独立的 screen 会话
        server.cmd('screen -dm bash')
        setup_server(server)
        TrafficControl.setup_tc(server)
        
        
        request_gen = RequestGenerator(client)
        monitor = TrafficMonitor(server, request_gen)

        request_gen.start()
        monitor.start()

        #CLI(net)

        while True:
            time.sleep(1)
        
    except KeyboardInterrupt:
        info("\nStopping services...")
        request_gen.stop()
        monitor.stop()
       
    finally:
        net.stop()