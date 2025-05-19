import os
import random
import re
import subprocess
import time
from collections import defaultdict

import util

from mininet.net import Mininet
from mininet.node import Controller, RemoteController, OVSController
from mininet.node import CPULimitedHost, Host, Node
from mininet.node import OVSKernelSwitch, UserSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink, Intf

import random
import os, stat
import json
import time
import copy
import csv
import requests
import sys
sys.path.append("...")
sys.path.append("..")
sys.path.append("../controller")
sys.path.append(".")
print(os.getcwd())
print(sys.path.__str__())
from config import Config


streamingMonitorClient=util.StreamingMonitorClient()

total_bandwidth=20

TRAFFIC_CLASSES = {
    'high': {'mark': 10, 'rate': '8mbit', 'ceil': '9mbit', 'classid': '1:10'},
    'middle': {'mark': 20, 'rate': '4mbit', 'ceil': '5mbit', 'classid': '1:20'},
    'low': {'mark': 30, 'rate': '2mbit', 'ceil': '2mbit', 'classid': '1:30'},
}
TRAFFIC_CLASSES_MARK = {
    '10.0.0.2' : {'port': 10086, '12600': 20, '3150':20, '785':30, '200':30},
    '10.0.0.3' : {'port': 10086, '12600': 10, '3150':10, '785':30, '200':30},
    #'10.0.0.4' : {'port': 10086, '12600': 10, '3150':10, '785':30, '200':30}
}
TRAFFIC_CLASSES_DELAY = {
    '10.0.0.2' : {'client': 'client1','delay': 0, 'loss':0},
    '10.0.0.3' : {'client': 'client2','delay': 0, 'loss':0},
    #'10.0.0.4' : {'client': 'client3','delay': 0, 'loss':0}
}
ip_maps={
    'client1':'0.0.0.0',
    'client2':'0.0.0.0',
    #'client3':'0.0.0.0'
}

class TrafficControl:
    @staticmethod
    def adjust(server):
        """为每个IP的每个带宽值创建独立标记和TC类"""
        traffic_classes_band=streamingMonitorClient.fetch_traffic_classes_mark()
        normalized = {}
        # 手动计算总和
        total = 0
        for ip, data in traffic_classes_band.items():
            # 提取除 'port' 外的码率字段
            bitrate_weights = {k: v for k, v in data.items() if k != 'port'}
            for v in bitrate_weights.values():
                total += v

        for ip, data in traffic_classes_band.items():
            bitrate_weights = {k: v for k, v in data.items() if k != 'port'}
            # 避免除以 0
            if total == 0:
                norm_weights = {k: 0 for k in bitrate_weights}
            else:
                norm_weights = {k: v / total for k, v in bitrate_weights.items()}

            # 组合结果
            normalized[ip] = {'port': data['port'], **norm_weights}

        traffic_classes_band=normalized
        # 阶段1：为每个(ip, 带宽)生成唯一标记
        ip_mark_mapping = defaultdict(lambda: defaultdict(lambda: {
            'mark':0,
            'bw':0
        }))
        current_mark = 10  # 起始标记值

        # 遍历所有IP和配置
        for ip, config in traffic_classes_band.items():
            # 为每个字符串对应的带宽生成独立标记
            for str_key in ['12600', '3150', '785', '200']:
                if (bw := config.get(str_key)) is not None:
                    # 每个IP的每个带宽分配唯一标记
                    ip_mark_mapping[ip][str_key] = {
                        'mark': current_mark,
                        'bw': bw*total_bandwidth
                    }
                    current_mark += 10  # 步长10保证唯一

        for ip,item0 in ip_mark_mapping.items():
            for str_key,item1 in item0.items():
                print(f'ip:{ip}',f'str_key:{str_key}',f'bw:{item1["bw"]}')

        # 阶段2：计算总带宽需求
        total_bw = total_bandwidth

        # 阶段3：清除旧配置
        server.cmd('tc qdisc del dev server-eth0 root 2>/dev/null')

        # 阶段4：构建TC配置
        tc_cmds = [
            'tc qdisc add dev server-eth0 root handle 1: htb',
            f'tc class add dev server-eth0 parent 1: classid 1:1 htb rate {int(total_bw*1.1)}mbit'
        ]

        # 为每个唯一标记创建TC类
        created_classes = set()
        for ip_config in ip_mark_mapping.values():
            for item in ip_config.values():
                mark = item['mark']
                if mark in created_classes:
                    continue

                tc_cmds.extend([
                    f'tc class add dev server-eth0 parent 1:1 classid 1:{mark} '
                    f'htb rate {item["bw"]}mbit ceil {item["bw"]}mbit',
                    f'tc filter add dev server-eth0 parent 1: protocol ip '
                    f'handle {mark} fw flowid 1:{mark}'
                ])
                created_classes.add(mark)

        # 执行TC命令
        for cmd in tc_cmds:
            server.cmd(cmd)

        # 阶段5：构建iptables规则
        iptables_cmds = [
            'iptables -t mangle -F',
            'iptables -t mangle -X',
            'iptables -t mangle -Z'
        ]

        port_set = set()

        # 生成每个IP的规则
        for ip, config in traffic_classes_band.items():
            port = config['port']
            port_set.add(port)

            mark_info = ip_mark_mapping[ip]
            for str_key in ['12600', '3150', '785', '200']:
                if str_key not in mark_info:
                    continue

                rule = (
                    f'iptables -t mangle -A PREROUTING '
                    f'-s {ip} -p tcp --dport {port} '
                    f'-m string --algo kmp --string "{str_key}" '
                    f'-j CONNMARK --set-mark {mark_info[str_key]["mark"]}'
                )
                iptables_cmds.append(rule)

        # 添加端口恢复规则
        for port in port_set:
            iptables_cmds.append(
                f'iptables -t mangle -A OUTPUT '
                f'-p tcp --sport {port} -j CONNMARK --restore-mark'
            )

        # 执行iptables命令
        for cmd in iptables_cmds:
            server.cmd(cmd)

    @staticmethod
    def adjust_loss_and_delay(net):
        """
        调整特定 IP 地址与目标主机之间的丢包率和时延。
        :param server: 服务器对象
        :param ip: 源 IP 地址（如 '10.0.0.2' 或 '10.0.0.3'）
        """
        float_dict=streamingMonitorClient.fetch_traffic_classes_delay()
        # 确保字典中有指定的 IP 地址
        for ip in float_dict:
            # 获取指定 IP 地址的配置
            config = TRAFFIC_CLASSES_DELAY[ip]
            target = config['client']
            server=net.get(target)
            # 更新字典中的丢包率和延迟（可选，模拟动态调整）
            TRAFFIC_CLASSES_DELAY[ip]['loss'] = float_dict[ip]['loss']  # 动态修改丢包率
            TRAFFIC_CLASSES_DELAY[ip]['delay'] = float_dict[ip]['delay']  # 动态修改延迟
            loss_prob = TRAFFIC_CLASSES_DELAY[ip]['loss']
            delay = TRAFFIC_CLASSES_DELAY[ip]['delay']

            # 输出当前配置
            #print(f"Adjusting {target} (IP: {ip}) with loss: {loss_prob}% and delay: {delay}ms.")
            # 删除现有的 qdisc 配置（避免冲突）
            server.cmd(f'tc qdisc del dev {target}-eth0 root 2>/dev/null')#tc qdisc del dev client1-eth0 root 2>/dev/null
            # 设置丢包率为独立的 qdisc
            server.cmd(f'tc qdisc add dev {target}-eth0 root netem delay {delay}ms loss {loss_prob}% ')#tc qdisc add dev client1-eth0 root netem delay 20ms loss 20%
            # 输出已应用的延迟和丢包率
            server.cmd(f"tc qdisc show dev {target}-eth0")
            #print(f"Applied {loss_prob}% loss and {delay}ms delay to {target} (IP: {ip}).")

    @staticmethod
    def report_traffic_classes():
        # 合并后输出
        print("Traffic Classes Configuration:")


def setup_network():
    try:
        controller = subprocess.Popen(["ryu-manager", "remote_controller.py"])
        time.sleep(3)
        net = Mininet(link=TCLink)
        c0 = net.addController(name='c0',
                               controller=RemoteController,
                               ip='127.0.0.1',
                               protocol='tcp',
                               port=6633)
        c1 = net.addController('c1')
        s0 = net.addSwitch('s0')
        s1 = net.addSwitch('s1',cls=OVSKernelSwitch)
        s2 = net.addSwitch('s2',cls=OVSKernelSwitch)
        s3 = net.addSwitch('s3',cls=OVSKernelSwitch)
        s4 = net.addSwitch('s4',cls=OVSKernelSwitch)
        s5 = net.addSwitch('s5',cls=OVSKernelSwitch)
        s6 = net.addSwitch('s6',cls=OVSKernelSwitch)
        s7 = net.addSwitch('s7',cls=OVSKernelSwitch)
        s8 = net.addSwitch('s8',cls=OVSKernelSwitch)
        s9 = net.addSwitch('s9',cls=OVSKernelSwitch)

        net.addLink(s1, s2)
        net.addLink(s3, s2)
        net.addLink(s4, s2)
        net.addLink(s5, s2)
        net.addLink(s6, s2)
        net.addLink(s7, s6)
        net.addLink(s8, s6)
        net.addLink(s9, s6)

        server = net.addHost('server', ip='10.0.0.1')
        client1 = net.addHost('client1', ip='10.0.0.2')
        client2 = net.addHost('client2', ip='10.0.0.3')

        net.addLink(server, s9, cls=TCLink, bw=1000, intfName1='server-eth0')
        net.addLink(client1, s1, cls=TCLink, bw=1000, intfName1='client1-eth0')
        net.addLink(client2, s2, cls=TCLink, bw=1000, intfName1='client2-eth0')
        net.addLink(server, s0, cls=TCLink, bw=1000, intfName1='server-eth1')
        net.addLink(client1, s0, cls=TCLink, bw=1000, intfName1='client1-eth1')
        net.addLink(client2, s0, cls=TCLink, bw=1000, intfName1='client2-eth1')
        print('network set')
        net.build()
        c0.start()
        s0.start([c1])
        s1.start([c0])
        s2.start([c0])
        s3.start([c0])
        s4.start([c0])
        s5.start([c0])
        s6.start([c0])
        s7.start([c0])
        s8.start([c0])
        s9.start([c0])
        print('network start')
        #os.system('ifconfig eth1 0.0.0.0')
        #os.system('ovs-vsctl add-port s0 eth1')
        #server.cmd('ifconfig server-eth1 0.0.0.0')
        #client1.cmd('ifconfig client1-eth1 0.0.0.0')
        #client2.cmd('ifconfig client2-eth1 0.0.0.0')
        #server.cmd('dhclient server-eth1')
        #client1.cmd('dhclient client1-eth1')
        #client2.cmd('dhclient client2-eth1')
        #client3.cmd('dhclient client3-eth1')
        print('ip request')
        server.cmd('ifconfig server-eth1 192.168.16.201/24')
        server.cmd('route add default gw 192.168.16.2 dev server-eth1')

        client1.cmd('ifconfig client1-eth1 192.168.16.202/24')
        client1.cmd('route add default gw 192.168.16.2 dev client1-eth1')

        client2.cmd('ifconfig client2-eth1 192.168.16.203/24')
        client2.cmd('route add default gw 192.168.16.2 dev client2-eth1')

        print(server.cmd('ifconfig'))
        print(client1.cmd('ifconfig'))
        print(client2.cmd('ifconfig'))
        client1.cmd('screen -dm bash_client1')
        client2.cmd('screen -dm bash_client2')
        server.cmd('screen -dm bash_server')


        def get_eth1_ip(host):
            output = host.cmd(f'ifconfig {host.name}-eth1')
            match = re.search(r'inet (\d+\.\d+\.\d+\.\d+)', output)
            if match:
                return match.group(1)
            return None

        for client_name,_ in ip_maps.items():
            client_host = net.get(client_name)
            ip_addr = get_eth1_ip(client_host)
            ip_maps[client_name] = ip_addr

        def submit_with_retry(client, ip_maps, max_retry_interval=5):
            """
            尝试向服务器提交 ip_maps，如果失败则无限重试，直到成功为止。
            每次失败后等待 max_retry_interval 秒再试一次。
            """
            while True:
                    if client.submit_ip_maps(ip_maps) is True:
                        break
                    time.sleep(max_retry_interval)
        server.cmd('cd /home/mininet/gpac_on_mininet/Server && screen -dmS server python3 server_train.py')
        client1.cmd('cd /home/mininet/gpac_on_mininet/mininet && screen -dmS proxy1 python3 proxy.py client1')
        client2.cmd('cd /home/mininet/gpac_on_mininet/mininet && screen -dmS proxy2 python3 proxy.py client2')
        print('server start')
        server.cmd('cd /home/mininet/gpac_on_mininet/mininet && screen -dmS monitor python3 monitor.py')
        server.cmd('cd /home/mininet/gpac_on_mininet/mininet && screen -dmS monitor1 python3 monitor2.py')
        print('monitor start')
        submit_with_retry(streamingMonitorClient, ip_maps)
        while True:
            streamingMonitorClient.submit_ip_maps(ip_maps)
            TrafficControl.adjust(server)
            TrafficControl.adjust_loss_and_delay(net)
            time.sleep(1)
    except KeyboardInterrupt:
        net.stop()
        pass
    finally:
        os.system("sudo mn -c")
        os.system("sudo pkill screen")



if __name__ == '__main__':
    setup_network()