import re
import time

from tabulate import tabulate

import util
streamingMonitorClient=util.StreamingMonitorClient('http://192.168.3.22:5000')
def mark2bw(x):
    if x == 10:
        return 50
    if x == 20:
        return 35
    if x == 30:
        return 10



latest_rate_history = {}
latest_delay_history = {}
while True:
    link_metrics = streamingMonitorClient.fetch_link_metrics()
    client_stats = streamingMonitorClient.fetch_client_stats()

    # Link Metrics 表格
    link_headers = ['Client ID', 'Delay(ms)', 'Loss Rate(%)', '12600_rate', '3150_rate', '785_rate','200_rate']
    link_data = []

    for client_id, stats in link_metrics.items():
        bw_12600 = mark2bw(stats['marks']['12600'])
        bw_3150 = mark2bw(stats['marks']['3150'])
        bw_785 = mark2bw(stats['marks']['785'])
        bw_200 = mark2bw(stats['marks']['200'])
        link_data.append((client_id, stats['delay'], stats['loss_rate'], bw_12600, bw_3150, bw_785, bw_200))

    link_table = tabulate(link_data, headers=link_headers, tablefmt="pretty", floatfmt=".2f")

    # Client Stats 表格
    client_headers = ['Client ID', 'Rebuffer Time(s)', 'Rebuffer Count', 'QoE']
    client_table_data = []

    for client_id, stats in client_stats.items():
        client_table_data.append((
            client_id, stats['rebuffer_time'], stats['rebuffer_count'], stats['qoe']
        ))

    client_table = tabulate(client_table_data, headers=client_headers, tablefmt="pretty", floatfmt=".2f")
    print("\nLink Metrics Table:")
    print(link_table)
    print("\nClient Stats Table:")
    print(client_table)
    time.sleep(1)
