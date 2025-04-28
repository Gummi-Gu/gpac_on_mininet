import os
import random
import re
import time
import util

from mininet.net import Mininet
from mininet.node import Controller, OVSSwitch
from mininet.link import TCLink
from mininet.cli import CLI

streamingMonitorClient=util.StreamingMonitorClient('http://192.168.3.22:5000')

TRAFFIC_CLASSES = {
    'high': {'mark': 10, 'rate': '50mbit', 'ceil': '50mbit', 'classid': '1:10'},
    'middle': {'mark': 20, 'rate': '20mbit', 'ceil': '20mbit', 'classid': '1:20'},
    'low': {'mark': 30, 'rate': '10mbit', 'ceil': '10mbit', 'classid': '1:30'},
}
TRAFFIC_CLASSES_MARK = {
    '10.0.0.2' : {'port': 10086, '12600': 20, '3150':20, '785':30, '200':30},
    '10.0.0.3' : {'port': 10086, '12600': 10, '3150':10, '785':30, '200':30},
    '10.0.0.4' : {'port': 10086, '12600': 10, '3150':10, '785':30, '200':30}
}
TRAFFIC_CLASSES_DELAY = {
    '10.0.0.2' : {'client': 'client1','delay': 0, 'loss':0},
    '10.0.0.3' : {'client': 'client2','delay': 0, 'loss':0},
    '10.0.0.4' : {'client': 'client3','delay': 0, 'loss':0}
}
ip_maps={
    'client1':'0.0.0.0',
    'client2':'0.0.0.0',
    'client3':'0.0.0.0'
}



class TrafficControl:
    @staticmethod
    def setup_tc(server):
        cmds = [
            'tc qdisc del dev server-eth0 root 2>/dev/null',
            'tc qdisc add dev server-eth0 root handle 1: htb',
            'tc class add dev server-eth0 parent 1: classid 1:1 htb rate 100mbit',
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
            'iptables -t mangle -A PREROUTING -p tcp --dport 10086 -m string --algo kmp --string "200" -j CONNMARK --set-mark 30',
            'iptables -t mangle -A OUTPUT -p tcp --sport 10086 -j CONNMARK --restore-mark'
        ]

        for cmd in cmds + connmark_cmds:
            server.cmd(cmd)

    @staticmethod
    def adjust(server):
        """
        生成针对特定 IP 地址的 connmark 规则。
        :param server:
        :param ip: 源 IP 地址
        :param port: 目标端口
        :param string_dict: 一个字典，包含匹配的字符串和对应的标记（只会包含12600，3150，785）
        :return: 包含生成的 iptables 规则的列表
        """
        connmark_cmds = []
        string_dict=streamingMonitorClient.fetch_traffic_classes_mark()
        TRAFFIC_CLASSES_MARK.update(string_dict)  # 合并输入的 string_dict 到指定 IP 的标记中
        expected_strings=['12600', '3150', '785', '200']
        for ip in string_dict:
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
                        #rule = f'iptables -t mangle -A PREROUTING -p tcp --dport {port}  -m string --algo kmp --string "{string}" -j CONNMARK --set-mark {mark}'
                        rule = f'iptables -t mangle -A PREROUTING -p tcp -s {ip} --dport {port} -m string --algo kmp --string "{string}" -j CONNMARK --set-mark {mark}'
                        connmark_cmds.append(rule)

                # 恢复连接标记规则
                connmark_cmds.append(
                    f'iptables -t mangle -A OUTPUT -p tcp --sport {port} -j CONNMARK --restore-mark')

            # 执行命令
            for cmd in connmark_cmds:
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
            print(f"Adjusting {target} (IP: {ip}) with loss: {loss_prob}% and delay: {delay}ms.")
            # 删除现有的 qdisc 配置（避免冲突）
            print(server.cmd(f'tc qdisc del dev {target}-eth0 root 2>/dev/null'))#tc qdisc del dev client1-eth0 root 2>/dev/null
            # 设置丢包率为独立的 qdisc
            print(server.cmd(f'tc qdisc add dev {target}-eth0 root netem delay {delay}ms loss {loss_prob}% '))#tc qdisc add dev client1-eth0 root netem delay 20ms loss 20%
            # 输出已应用的延迟和丢包率
            print(server.cmd(f"tc qdisc show dev {target}-eth0"))
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
            streamingMonitorClient.submit_link_metrics(config['client'],config['delay'],config['loss'],TRAFFIC_CLASSES_MARK[ip])


def setup_network():
    try:
        net = Mininet(controller=Controller, switch=OVSSwitch, link=TCLink)
        net.addController('c0')
        s1 = net.addSwitch('s1')
        s2 = net.addSwitch('s2')

        server = net.addHost('server', ip='10.0.0.1/24')
        client1 = net.addHost('client1', ip='10.0.0.2/24')
        client2 = net.addHost('client2', ip='10.0.0.3/24')
        client3 = net.addHost('client3', ip='10.0.0.4/24')

        net.addLink(server, s1, cls=TCLink, bw=1000, intfName1='server-eth0')
        net.addLink(client1, s1, cls=TCLink, bw=1000, intfName1='client1-eth0')
        net.addLink(client2, s1, cls=TCLink, bw=1000, intfName1='client2-eth0')
        net.addLink(client3, s1, cls=TCLink, bw=1000, intfName1='client3-eth0')
        net.addLink(server, s2, cls=TCLink, bw=1000, intfName1='server-eth1')
        net.addLink(client1, s2, cls=TCLink, bw=1000, intfName1='client1-eth1')
        net.addLink(client2, s2, cls=TCLink, bw=1000, intfName1='client2-eth1')
        net.addLink(client3, s2, cls=TCLink, bw=1000, intfName1='client3-eth1')


        net.start()

        os.system('ifconfig eth1 0.0.0.0')
        os.system('ovs-vsctl add-port s2 eth1')

        server.cmd('ifconfig server-eth1 0.0.0.0')
        client1.cmd('ifconfig client1-eth1 0.0.0.0')
        client2.cmd('ifconfig client2-eth1 0.0.0.0')
        client3.cmd('ifconfig client3-eth1 0.0.0.0')
        server.cmd('dhclient server-eth1')
        client1.cmd('dhclient client1-eth1')
        client2.cmd('dhclient client2-eth1')
        client3.cmd('dhclient client3-eth1')
        print(server.cmd('ifconfig'))
        print(client1.cmd('ifconfig'))
        print(client2.cmd('ifconfig'))
        print(client3.cmd('ifconfig'))
        client1.cmd('screen -dm bash_client1')
        client2.cmd('screen -dm bash_client2')
        client3.cmd('screen -dm bash_client3')
        server.cmd('screen -dm bash_server')

        server.cmd('cd /home/mininet/gpac_on_mininet/Server && screen -dmS server python3 server_train.py')
        #server.cmd('cd /home/mininet/gpac_on_mininet/Server && screen -dmS monitor python3 monitor.py')
        client1.cmd('cd /home/mininet/gpac_on_mininet/mininet && screen -dmS proxy1 python3 proxy.py client1')
        client2.cmd('cd /home/mininet/gpac_on_mininet/mininet && screen -dmS proxy2 python3 proxy.py client2')
        client3.cmd('cd /home/mininet/gpac_on_mininet/mininet && screen -dmS proxy3 python3 proxy.py client3')

        TrafficControl.setup_tc(server)

        def get_eth1_ip(host):
            output = host.cmd(f'ifconfig {host.name}-eth1')
            match = re.search(r'inet (\d+\.\d+\.\d+\.\d+)', output)
            if match:
                return match.group(1)
            return None

        for client_name in ['client1', 'client2', 'client3']:
            client_host = net.get(client_name)
            ip_addr = get_eth1_ip(client_host)
            ip_maps[client_name] = ip_addr


        while True:
            streamingMonitorClient.submit_ip_maps(ip_maps)
            TrafficControl.adjust(server)
            TrafficControl.adjust_loss_and_delay(net)
            '''
            TrafficControl.report_traffic_classes()
            user_input = input(
                "\nEnter 'adjust' to throttle rates; 'delay' to adjust delay/loss; 'test' to test connections: ").strip().lower()
            if user_input == 'adjust':
                TrafficControl.adjust(server)
            elif user_input == 'delay':
                TrafficControl.adjust_loss_and_delay(net)
            elif user_input == 'test':
                server = net.get('server')

                def test_ping_connection(client, server_ip):
                    # 测试延迟和丢包情况
                    print(f"\nTesting ping from {client.IP()} to {server_ip}...")

                    # 使用 ping 测试并设置发送的包数为 10（-c 10）
                    result = client.cmd(f"ping -c 10 {server_ip}")  # 进行10次ping测试
                    print(result)

                # 测试 10.0.0.1 到 10.0.0.2 和 10.0.0.3 的延迟和丢包
                test_ping_connection(net.get('client1'), '10.0.0.1')
                test_ping_connection(net.get('client2'), '10.0.0.1')
                test_ping_connection(net.get('client3'), '10.0.0.1')
            else:
                print("Invalid input!")
            '''

    except KeyboardInterrupt:
        pass
    finally:
        net.stop()
        os.system("sudo mn -c")
        os.system("sudo pkill screen")


if __name__ == '__main__':
    setup_network()