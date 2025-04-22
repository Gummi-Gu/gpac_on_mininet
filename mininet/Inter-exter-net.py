import os
import random
import time

from mininet.net import Mininet
from mininet.node import Controller, OVSSwitch
from mininet.link import TCLink
from mininet.cli import CLI

TRAFFIC_CLASSES = {
    'high': {'mark': 10, 'rate': '100mbit', 'ceil': '100mbit', 'classid': '1:10'},
    'middle': {'mark': 20, 'rate': '80mbit', 'ceil': '80mbit', 'classid': '1:20'},
    'low': {'mark': 30, 'rate': '20mbit', 'ceil': '20mbit', 'classid': '1:30'},
}
TRAFFIC_CLASSES_MARK = {
    '10.0.0.2' : {'port': 10086, '12600': 10, '3150':10, '785':30},
    '10.0.0.3' : {'port': 10086, '12600': 10, '3150':10, '785':30}
}
TRAFFIC_CLASSES_DELAY = {
    '10.0.0.2' : {'client': 'client1','delay': 0, 'loss':0},
    '10.0.0.3' : {'client': 'client2','delay': 0, 'loss':0}
}


class TrafficControl:
    @staticmethod
    def setup_tc(server):
        cmds = [
            'tc qdisc del dev server-eth0 root 2>/dev/null',
            'tc qdisc add dev server-eth0 root handle 1: htb',
            'tc class add dev server-eth0 parent 1: classid 1:1 htb rate 200mbit',
            f'tc class add dev server-eth0 parent 1:1 classid {TRAFFIC_CLASSES["high"]["classid"]} htb rate {TRAFFIC_CLASSES["high"]["rate"]} ceil {TRAFFIC_CLASSES["high"]["ceil"]}',
            f'tc class add dev server-eth0 parent 1:1 classid {TRAFFIC_CLASSES["middle"]["classid"]} htb rate {TRAFFIC_CLASSES["middle"]["rate"]} ceil {TRAFFIC_CLASSES["middle"]["ceil"]}',
            f'tc class add dev server-eth0 parent 1:1 classid {TRAFFIC_CLASSES["low"]["classid"]} htb rate {TRAFFIC_CLASSES["low"]["rate"]} ceil {TRAFFIC_CLASSES["low"]["ceil"]}',
            'tc filter add dev server-eth0 parent 1: protocol ip handle 10 fw flowid 1:10',
            'tc filter add dev server-eth0 parent 1: protocol ip handle 20 fw flowid 1:20',
            'tc filter add dev server-eth0 parent 1: protocol ip handle 30 fw flowid 1:30',
        ]
        connmark_cmds = [
            'iptables -t mangle -A PREROUTING -p tcp --dport 10086 -m string --algo kmp --string "12600" -j CONNMARK --set-mark 10',
            'iptables -t mangle -A PREROUTING -p tcp --dport 10086 -m string --algo kmp --string "3150" -j CONNMARK --set-mark 20',
            'iptables -t mangle -A PREROUTING -p tcp --dport 10086 -m string --algo kmp --string "785" -j CONNMARK --set-mark 30',
            'iptables -t mangle -A OUTPUT -p tcp --sport 10086 -j CONNMARK --restore-mark'
        ]

        for cmd in cmds + connmark_cmds:
            server.cmd(cmd)

    @staticmethod
    def adjust(server, ip: str,string_dict: dict):
        """
        生成针对特定 IP 地址的 connmark 规则。
        :param ip: 源 IP 地址
        :param port: 目标端口
        :param string_dict: 一个字典，包含匹配的字符串和对应的标记（只会包含12600，3150，785）
        :return: 包含生成的 iptables 规则的列表
        """
        connmark_cmds = []

        # 确保字典只包含预期的键
        expected_strings = ["12600", "3150", "785"]

        # 更新 TRAFFIC_CLASSES_MARK 字典
        if ip in TRAFFIC_CLASSES_MARK:
            TRAFFIC_CLASSES_MARK[ip].update(string_dict)  # 合并输入的 string_dict 到指定 IP 的标记中

        # 清空之前的规则
        connmark_cmds.append('iptables -t mangle -F')  # 清空 mangle 表中的所有规则
        connmark_cmds.append('iptables -t mangle -X')  # 删除所有用户自定义链
        connmark_cmds.append('iptables -t mangle -Z')

        # 生成新的规则
        if ip in TRAFFIC_CLASSES_MARK:
            # 获取该 IP 对应的端口和标记
            port = TRAFFIC_CLASSES_MARK[ip]['port']
            for string in expected_strings:
                if string in TRAFFIC_CLASSES_MARK[ip]:
                    mark = TRAFFIC_CLASSES_MARK[ip][string]
                    rule = f'iptables -t mangle -A PREROUTING -p tcp --dport {port} -s {ip} -m string --algo kmp --string "{string}" -j CONNMARK --set-mark {mark}'
                    connmark_cmds.append(rule)

            # 恢复连接标记规则
            connmark_cmds.append(
                f'iptables -t mangle -A OUTPUT -p tcp --sport {port} -s {ip} -j CONNMARK --restore-mark')

        # 执行命令
        for cmd in connmark_cmds:
            server.cmd(cmd)

    @staticmethod
    def adjust_loss_and_delay(server, ip: str):
        """
        调整特定 IP 地址与目标主机之间的丢包率和时延。
        :param server: 服务器对象
        :param ip: 源 IP 地址（如 '10.0.0.2' 或 '10.0.0.3'）
        """
        # 确保字典中有指定的 IP 地址
        if ip not in TRAFFIC_CLASSES_DELAY:
            print(f"IP {ip} not found in TRAFFIC_CLASSES_DELAY.")
            return

        # 获取指定 IP 地址的配置
        config = TRAFFIC_CLASSES_DELAY[ip]
        target = config['client']
        loss_prob = config['loss']
        delay = config['delay']

        # 更新字典中的丢包率和延迟（可选，模拟动态调整）
        TRAFFIC_CLASSES_DELAY[ip]['loss'] = random.uniform(0, 10)  # 动态修改丢包率
        TRAFFIC_CLASSES_DELAY[ip]['delay'] = random.randint(20, 100)  # 动态修改延迟

        # 输出当前配置
        print(f"Adjusting {target} (IP: {ip}) with loss: {loss_prob}% and delay: {delay}ms.")

        # 设置丢包率
        server.cmd(f'tc qdisc add dev {target}-eth0 parent 1:10 handle 10: netem loss {loss_prob}%')

        # 设置时延
        server.cmd(f'tc qdisc add dev {target}-eth0 parent 1:10 handle 10: netem delay {delay}ms')

        # 输出已应用的延迟和丢包率
        print(f"Applied {loss_prob}% loss and {delay}ms delay to {target} (IP: {ip}).")

    @staticmethod
    def report_traffic_classes():
        # 合并后输出
        print("Merged Traffic Classes Configuration:")
        merged_traffic = {}

        # 合并 TRAFFIC_CLASSES_MARK 和 TRAFFIC_CLASSES_DELAY
        for ip, config in TRAFFIC_CLASSES_MARK.items():
            if ip not in merged_traffic:
                merged_traffic[ip] = {'client': '', 'port': config['port'], 'marks': {}, 'delay': 0, 'loss': 0}

            # 合并标记
            for string, mark in config.items():
                if string != 'port':
                    merged_traffic[ip]['marks'][string] = mark

        for ip, config in TRAFFIC_CLASSES_DELAY.items():
            if ip not in merged_traffic:
                merged_traffic[ip] = {'client': config['client'], 'port': '', 'marks': {}, 'delay': config['delay'],
                                      'loss': config['loss']}
            else:
                # 合并延迟和丢包率
                merged_traffic[ip]['client'] = config['client']
                merged_traffic[ip]['delay'] = config['delay']
                merged_traffic[ip]['loss'] = config['loss']

        # 输出合并后的配置
        for ip, config in merged_traffic.items():
            print(f"IP: {ip}")
            print(f"  Client: {config['client']}")
            print(f"  Port: {config['port']}")
            print(f"  Marks: {config['marks']}")
            print(f"  Delay: {config['delay']} ms")
            print(f"  Loss: {config['loss']} %")
            print("")  # 空行分隔


def setup_network():
    net = Mininet(controller=Controller, switch=OVSSwitch, link=TCLink)
    net.addController('c0')
    s1 = net.addSwitch('s1')
    s2 = net.addSwitch('s2')

    server = net.addHost('server', ip='10.0.0.1/24')
    client1 = net.addHost('client1', ip='10.0.0.2/24')
    client2 = net.addHost('client2', ip='10.0.0.3/24')

    net.addLink(server, s1, cls=TCLink, bw=1000, intfName1='server-eth0')
    net.addLink(client1, s1, cls=TCLink, bw=1000, intfName1='client1-eth0')
    net.addLink(client2, s1, cls=TCLink, bw=1000, intfName1='client2-eth0')
    net.addLink(server, s2, cls=TCLink, bw=1000, intfName1='server-eth1')
    net.addLink(client1, s2, cls=TCLink, bw=1000, intfName1='client1-eth1')
    net.addLink(client2, s2, cls=TCLink, bw=1000, intfName1='client2-eth1')


    net.start()

    os.system('ifconfig eth1 0.0.0.0')
    os.system('ovs-vsctl add-port s2 eth1')

    server.cmd('ifconfig server-eth1 0.0.0.0')
    client1.cmd('ifconfig client1-eth1 0.0.0.0')
    client2.cmd('ifconfig client2-eth1 0.0.0.0')
    server.cmd('dhclient server-eth1')
    client1.cmd('dhclient client1-eth1')
    client2.cmd('dhclient client2-eth1')
    print(server.cmd('ifconfig'))
    print(client1.cmd('ifconfig'))
    print(client2.cmd('ifconfig'))
    client1.cmd('screen -dm bash_client1')
    client2.cmd('screen -dm bash_client2')
    server.cmd('screen -dm bash_server')

    #server.cmd('cd /home/mininet/gpac_on_mininet/Server && screen -dmS server python3 server.py')
    #server.cmd('cd /home/mininet/gpac_on_mininet/Server && screen -dmS monitor python3 monitor.py')
    #client1.cmd('cd /home/mininet/gpac_on_mininet/mininet && screen -dmS proxy python3 proxy.py')
    #client2.cmd('cd /home/mininet/gpac_on_mininet/mininet && screen -dmS proxy python3 proxy.py')

    TrafficControl.setup_tc(server)

    try:
        while True:
            user_input = input(
                "\nEnter 'adjust' to throttle rates; 'delay' to adjust delay/loss; 'test' to test connections: ").strip().lower()
            if user_input == 'adjust':
                input_str=input('ip port mark1 mark2 mark3')
                parts = input_str.split()
                # 确保输入格式正确
                if len(parts) != 5:
                    raise ValueError("shoule be 'ip mark1 mark2 mark3'")
                ip = parts[0]  # IP 地址
                string_dict = {
                    '12600': int(parts[1]),  # 标记 12600
                    '3150': int(parts[2]),  # 标记 3150
                    '785': int(parts[3])  # 标记 785
                }
                TrafficControl.adjust(ip,string_dict)
            elif user_input == 'delay':
                ip=input('ip address')
                TrafficControl.adjust_loss_and_delay(server,ip)
            elif user_input == 'test':
                # 启动iperf服务器
                server = net.get('server')
                server.cmd("iperf -s -D")

                def test_iperf_connection(client, server_ip):
                    # 测试带宽、丢包率和时延
                    print(f"\nTesting iperf from {client.IP()} to {server_ip}...")

                    # 使用 UDP 测试 (-u)，并输出每秒的统计信息（-i 1）
                    # 设定带宽为 10Mbps（-b 10M），可以根据需要调整带宽
                    result = client.cmd(f"iperf -c {server_ip} -u -b 10M -t 10 -i 1")  # 进行10秒的UDP带宽测试
                    print(result)
                # 测试10.0.0.1到10.0.0.2 和10.0.0.3的带宽
                test_iperf_connection(net.get('client1'), '10.0.0.1')
                test_iperf_connection(net.get('client2'), '10.0.0.1')
            else:
                print("Invalid input!")
            TrafficControl.report_traffic_classes()

    except KeyboardInterrupt:
        pass
    finally:
        net.stop()
        os.system("sudo mn -c")
        os.system("sudo pkill screen")


if __name__ == '__main__':
    setup_network()