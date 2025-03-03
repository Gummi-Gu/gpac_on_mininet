#!/usr/bin/env python3
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import Controller
from mininet.link import TCLink
from mininet.log import setLogLevel, info
from mininet.cli import CLI
import time
import threading
import requests

# ================== 配置参数 ==================
SERVER_IP = '10.0.0.1'
DASH_DIR = '/tmp/dash'
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
        """配置流量控制规则（带并发优化）"""
        cmds = [
            # 清除旧规则
            'tc qdisc del dev server-eth0 root 2>/dev/null',
            'iptables -t mangle -F',

            # 创建HTB队列
            'tc qdisc add dev server-eth0 root handle 1: htb default 1',
            'tc class add dev server-eth0 parent 1: classid 1:1 htb rate 10mbit',

            # 高码率分类
            f'tc class add dev server-eth0 parent 1:1 classid {TRAFFIC_CLASSES["high"]["classid"]} '
            f'htb rate {TRAFFIC_CLASSES["high"]["rate"]} burst 15k ceil {TRAFFIC_CLASSES["high"]["ceil"]}',

            # 低码率分类
            f'tc class add dev server-eth0 parent 1:1 classid {TRAFFIC_CLASSES["low"]["classid"]} '
            f'htb rate {TRAFFIC_CLASSES["low"]["rate"]} burst 10k ceil {TRAFFIC_CLASSES["low"]["ceil"]}',

            # 分类规则
            'tc filter add dev server-eth0 parent 1: protocol ip handle 10 fw flowid 1:10',
            'tc filter add dev server-eth0 parent 1: protocol ip handle 20 fw flowid 1:20',

            # 优化并发处理
            'tc qdisc add dev server-eth0 parent 1:10 handle 100: sfq perturb 10',
            'tc qdisc add dev server-eth0 parent 1:20 handle 200: sfq perturb 10'
        ]
        for cmd in cmds:
            server.cmd(cmd)

        # iptables规则（优化字符串匹配性能）
        server.cmd(
            f'iptables -t mangle -A PREROUTING -m string --string "GET /high/" --algo bm --from 60 -j MARK --set-mark {TRAFFIC_CLASSES["high"]["mark"]}')
        server.cmd(
            f'iptables -t mangle -A PREROUTING -m string --string "GET /low/" --algo bm --from 60 -j MARK --set-mark {TRAFFIC_CLASSES["low"]["mark"]}')
        server.cmd(
            'iptables -t mangle -A PREROUTING -m conntrack --ctstate ESTABLISHED,RELATED -j CONNMARK --restore-mark')


def setup_server(server):
    """准备测试环境（生成大文件便于观察）"""
    server.cmd(f'mkdir -p {DASH_DIR}/high {DASH_DIR}/low')
    server.cmd(f'dd if=/dev/urandom of={DASH_DIR}/high/chunk1.m4s bs=1M count=50')  # 50MB文件
    server.cmd(f'dd if=/dev/urandom of={DASH_DIR}/low/chunk1.m4s bs=1M count=30')  # 30MB文件
    server.cmd(f'cd {DASH_DIR} && python3 -m http.server 80 &')


def concurrent_requests(client):
    """执行并发请求测试"""
    urls = [
        ('high', 'http://10.0.0.1/high/chunk1.m4s'),
        ('low', 'http://10.0.0.1/low/chunk1.m4s'),
        ('high', 'http://10.0.0.1/high/chunk1.m4s'),
        ('low', 'http://10.0.0.1/low/chunk1.m4s')
    ]

    results = {'high': [], 'low': []}

    def fetch(url_pair):
        type_, url = url_pair
        start = time.time()
        try:
            # 使用curl获取完整文件并计算实际带宽
            client.cmd(f'curl -s {url} > /dev/null')
            duration = time.time() - start
            results[type_].append(duration)
        except Exception as e:
            print(f"请求失败: {str(e)}")

    threads = [threading.Thread(target=fetch, args=(url,)) for url in urls]
    for t in threads: t.start()
    for t in threads: t.join()

    # 计算统计结果
    for type_ in results:
        if results[type_]:
            avg_time = sum(results[type_]) / len(results[type_])
            print(f"{type_} 平均传输时间: {avg_time:.2f}s")


def monitor(server):
    """实时监控带宽使用情况"""
    while True:
        info('\n[实时监控] 服务器带宽统计:\n')
        info(server.cmd('tc -s class show dev server-eth0'))
        time.sleep(2)


if __name__ == '__main__':
    setLogLevel('info')
    net = Mininet(topo=DynamicTopo(), controller=Controller)
    net.start()
    server, client = net.get('server', 'client')

    try:
        # 准备服务器环境
        setup_server(server)
        TrafficControl.setup_tc(server)

        # 启动监控线程
        monitor_thread = threading.Thread(target=monitor, args=(server,))
        monitor_thread.daemon = True
        monitor_thread.start()

        # 执行并发测试
        info("\n=== 开始并发带宽测试 ===\n")
        concurrent_requests(client)

        # 保留CLI用于手动测试
        CLI(net)
    finally:
        net.stop()
        info("\n=== 测试完成，资源已清理 ===\n")