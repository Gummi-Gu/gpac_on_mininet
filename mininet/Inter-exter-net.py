import os
import time

from mininet.net import Mininet
from mininet.node import Controller, OVSSwitch
from mininet.link import TCLink
from mininet.cli import CLI

TRAFFIC_CLASSES = {
    '12600': {'mark': 10, 'rate': '80mbit', 'ceil': '80mbit', 'classid': '1:10'},
    '3150': {'mark': 20, 'rate': '65mbit', 'ceil': '65mbit', 'classid': '1:20'},
    '785': {'mark': 30, 'rate': '20mbit', 'ceil': '20mbit', 'classid': '1:30'},
    '200': {'mark': 40, 'rate': '5mbit', 'ceil': '5mbit', 'classid': '1:40'}
}


class TrafficControl:
    @staticmethod
    def setup_tc(server):
        cmds = [
            'tc qdisc del dev server-eth0 root 2>/dev/null',
            'tc qdisc add dev server-eth0 root handle 1: htb',
            'tc class add dev server-eth0 parent 1: classid 1:1 htb rate 200mbit',
            f'tc class add dev server-eth0 parent 1:1 classid {TRAFFIC_CLASSES["12600"]["classid"]} htb rate {TRAFFIC_CLASSES["12600"]["rate"]} ceil {TRAFFIC_CLASSES["12600"]["ceil"]}',
            f'tc class add dev server-eth0 parent 1:1 classid {TRAFFIC_CLASSES["3150"]["classid"]} htb rate {TRAFFIC_CLASSES["3150"]["rate"]} ceil {TRAFFIC_CLASSES["3150"]["ceil"]}',
            f'tc class add dev server-eth0 parent 1:1 classid {TRAFFIC_CLASSES["785"]["classid"]} htb rate {TRAFFIC_CLASSES["785"]["rate"]} ceil {TRAFFIC_CLASSES["785"]["ceil"]}',
            f'tc class add dev server-eth0 parent 1:1 classid {TRAFFIC_CLASSES["200"]["classid"]} htb rate {TRAFFIC_CLASSES["200"]["rate"]} ceil {TRAFFIC_CLASSES["200"]["ceil"]}',
            'tc filter add dev server-eth0 parent 1: protocol ip handle 10 fw flowid 1:10',
            'tc filter add dev server-eth0 parent 1: protocol ip handle 20 fw flowid 1:20',
            'tc filter add dev server-eth0 parent 1: protocol ip handle 30 fw flowid 1:30',
            'tc filter add dev server-eth0 parent 1: protocol ip handle 40 fw flowid 1:40'
        ]
        connmark_cmds = [
            'iptables -t mangle -A PREROUTING -p tcp --dport 10086 -m string --algo kmp --string "12600" -j CONNMARK --set-mark 10',
            'iptables -t mangle -A PREROUTING -p tcp --dport 10086 -m string --algo kmp --string "3150" -j CONNMARK --set-mark 20',
            'iptables -t mangle -A PREROUTING -p tcp --dport 10086 -m string --algo kmp --string "785" -j CONNMARK --set-mark 30',
            'iptables -t mangle -A PREROUTING -p tcp --dport 10086 -m string --algo kmp --string "200" -j CONNMARK --set-mark 40',
            'iptables -t mangle -A OUTPUT -p tcp --sport 10086 -j CONNMARK --restore-mark'
        ]

        for cmd in cmds + connmark_cmds:
            server.cmd(cmd)

    @staticmethod
    def limit_rates(server):
        print("\nApplying rate limits (30mbit) to 12600 and 3150 classes")
        cmds = [
            'tc class change dev server-eth0 parent 1:1 classid 1:10 htb rate 30mbit ceil 30mbit',
            'tc class change dev server-eth0 parent 1:1 classid 1:20 htb rate 30mbit ceil 30mbit'
        ]
        for cmd in cmds:
            server.cmd(cmd)
        print("Rate limits applied\n")
        print(server.cmd('tc class show dev server-eth0'))

    @staticmethod
    def reset_rates(server):
        print("\n🔙 Restoring original rate settings")
        cmds = [
            f'tc class change dev server-eth0 parent 1:1 classid 1:10 htb rate {TRAFFIC_CLASSES["12600"]["rate"]} ceil {TRAFFIC_CLASSES["12600"]["ceil"]}',
            f'tc class change dev server-eth0 parent 1:1 classid 1:20 htb rate {TRAFFIC_CLASSES["3150"]["rate"]} ceil {TRAFFIC_CLASSES["3150"]["ceil"]}'
        ]
        for cmd in cmds:
            server.cmd(cmd)
        print("Original settings restored\n")
        print(server.cmd('tc class show dev server-eth0'))


def setup_network():
    net = Mininet(controller=Controller, switch=OVSSwitch, link=TCLink)
    net.addController('c0')
    s1 = net.addSwitch('s1')
    s2 = net.addSwitch('s2')

    server = net.addHost('server', ip='10.0.0.1/24')
    client = net.addHost('client', ip='10.0.0.2/24')

    net.addLink(server, s1, cls=TCLink, bw=1000, intfName1='server-eth0')
    net.addLink(client, s1, cls=TCLink, bw=1000, intfName1='client-eth0')
    net.addLink(server, s2, cls=TCLink, bw=1000, intfName1='server-eth1')
    net.addLink(client, s2, cls=TCLink, bw=1000, intfName1='client-eth1')

    net.start()

    os.system('ifconfig eth1 0.0.0.0')
    os.system('ovs-vsctl add-port s2 eth1')

    server.cmd('ifconfig server-eth1 0.0.0.0')
    client.cmd('ifconfig client-eth1 0.0.0.0')
    server.cmd('dhclient server-eth1')
    client.cmd('dhclient client-eth1')
    server.cmd('ifconfig server-eth1 up')
    client.cmd('ifconfig client-eth1 up')

    server.cmd('cd /home/mininet/gpac_on_mininet/Server && screen -dmS server python3 server.py')
    server.cmd('cd /home/mininet/gpac_on_mininet/Server && screen -dmS monitor python3 monitor.py')
    client.cmd('cd /home/mininet/gpac_on_mininet/mininet && screen -dmS proxy python3 proxy.py')

    TrafficControl.setup_tc(server)

    try:
        while True:
            user_input = input(
                "\nEnter 'limit' to throttle rates, 'reset' to restore, 'exit' to quit: ").strip().lower()
            if user_input == 'limit':
                TrafficControl.limit_rates(server)
            elif user_input == 'reset':
                TrafficControl.reset_rates(server)
            else:
                print("Unknown command. Valid options: 'limit', 'reset', 'exit'")
    except KeyboardInterrupt:
        pass
    finally:
        net.stop()
        os.system("sudo mn -c")
        os.system("sudo pkill screen")


if __name__ == '__main__':
    setup_network()