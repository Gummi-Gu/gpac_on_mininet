import os
from mininet.net import Mininet
from mininet.node import Controller, OVSSwitch
from mininet.link import TCLink
from mininet.cli import CLI

TRAFFIC_CLASSES = {
    'high': {'mark': 10, 'rate': '50mbit', 'ceil': '50mbit', 'classid': '1:10'},
    'low': {'mark': 20, 'rate': '10mbit', 'ceil': '10mbit', 'classid': '1:20'}
}


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


def setup_network():
    net = Mininet(controller=Controller, switch=OVSSwitch, link=TCLink)

    # 添加控制器
    net.addController('c0')

    # 添加交换机
    s1 = net.addSwitch('s1')  # 内部网络交换机
    s2 = net.addSwitch('s2')  # 外部网络交换机

    # 添加服务器和客户端
    server = net.addHost('server', ip='10.0.0.1/24')  # eth0: 10.0.0.1
    client = net.addHost('client', ip='10.0.0.2/24')  # eth0: 10.0.0.2

    # eth0 连接 s1（Mininet 内部通信）
    net.addLink(server, s1, cls=TCLink, bw=1000, intfName1='server-eth0')
    net.addLink(client, s1, cls=TCLink, bw=1000, intfName1='client-eth0')

    # eth1 连接 s2（连接外部物理网络）
    net.addLink(server, s2, cls=TCLink, bw=1000, intfName1='server-eth1')
    net.addLink(client, s2, cls=TCLink, bw=1000, intfName1='client-eth1')

    # 启动网络
    net.start()

    # **配置 s2 连接物理网卡 eth1**
    os.system('ifconfig eth1 0.0.0.0')  # 释放物理网卡的 IP
    os.system('ovs-vsctl add-port s2 eth1')  # 把物理网卡 eth1 加入 s2
    print(os.system('ovs-vsctl show'))  # 查看 OVS 配置

    # **配置 server 和 client**
    # 释放 eth1 IP 并使用 DHCP 重新获取 IP（如果有 DHCP 服务器）
    server.cmd('ifconfig server-eth1 0.0.0.0')
    client.cmd('ifconfig client-eth1 0.0.0.0')
    server.cmd('dhclient server-eth1')
    client.cmd('dhclient client-eth1')

    # 确保 eth1 端口启用
    server.cmd('ifconfig server-eth1 up')
    client.cmd('ifconfig client-eth1 up')

    # 测试网络状态
    print(server.cmd('ifconfig'))
    print(client.cmd('ifconfig'))
    str="/home/mininet/gpac_on_mininet/mininet/dash"
    client.cmd('screen -dm bash')
    server.cmd('screen -dm bash')
    TrafficControl.setup_tc(server)
    server.cmd('cd /home/mininet/gpac_on_mininet/Server && screen -dmS server python3 server.py')
    server.cmd('cd /home/mininet/gpac_on_mininet/Server && screen -dmS collector python3 collector_server.py')
    server.cmd('cd /home/mininet/gpac_on_mininet/Server && screen -dmS collector python3 monitor.py')
    # 进入 CLI
    #CLI(net)

    # 关闭网络
    net.stop()

if __name__ == '__main__':
    setup_network()
