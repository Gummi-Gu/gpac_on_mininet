import os
from mininet.net import Mininet
from mininet.node import Controller, OVSSwitch
from mininet.link import TCLink
from mininet.cli import CLI

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

    # 进入 CLI
    CLI(net)

    # 关闭网络
    net.stop()

if __name__ == '__main__':
    setup_network()
