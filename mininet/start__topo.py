import os

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import Controller
from mininet.util import dumpNodeConnections
from mininet.cli import CLI  # 导入 Mininet 的 CLI

class SimpleTopo(Topo):
    def __init__(self):
        Topo.__init__(self)
        # 创建两个交换机
        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')

        # 创建三个主机
        h1 = self.addHost('h1')
        h2 = self.addHost('h2')
        h3 = self.addHost('h3')

        # 连接主机到交换机
        self.addLink(h1, s1)
        self.addLink(h2, s1)
        self.addLink(h3, s2)

        # 连接交换机之间
        self.addLink(s1, s2)

# 宿主机网络初始化
os.system('ifconfig ens37 0.0.0.0')
# 创建拓扑并启动网络
topos = { 'SimpleTopo': lambda: SimpleTopo() }
net = Mininet(topo=SimpleTopo(), controller=Controller)

# 启动网络
net.start()

# 获取网络中的主机对象（可选）
h1 = net.get('h1')
h2 = net.get('h2')
h3 = net.get('h3')
# 释放网卡
h1.cmd('ifconfig h1-eth0 0.0.0.0')
h2.cmd('ifconfig h2-eth0 0.0.0.0')
h3.cmd('ifconfig h3-eth0 0.0.0.0')
h1.cmd('dhclient h1-eth0')
h2.cmd('dhclient h2-eth0')
h3.cmd('dhclient h3-eth0')
# 可以在此处执行某些初始化命令，如果需要的话
# 比如查看接口信息
print(h1.cmd('ifconfig'))
print(h2.cmd('ifconfig'))
print(h3.cmd('ifconfig'))

# 进入 Mininet 的交互式命令行界面
CLI(net)

# 当你在 Mininet CLI 中输入 exit 时，网络将停止
net.stop()
