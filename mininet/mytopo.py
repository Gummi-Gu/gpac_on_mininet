from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import Controller
from mininet.util import dumpNodeConnections

class SimpleTopo(Topo):
    def __init__(self):
        Topo.__init__(self)
        # 创建两个交换机
        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')

        # 创建三个节点（主机）
        h1 = self.addHost('h1')
        h2 = self.addHost('h2')
        h3 = self.addHost('h3')

        # 连接交换机与主机
        self.addLink(h1, s1)
        self.addLink(h2, s1)
        self.addLink(h3, s2)

        # 连接交换机之间
        self.addLink(s1, s2)
topos = { 'SimpleTopo': lambda: SimpleTopo() }
net = Mininet(topo=SimpleTopo(), controller=Controller)

# 启动网络
net.start()