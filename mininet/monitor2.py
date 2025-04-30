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

while True:
    link_metrics = streamingMonitorClient.fetch_link_metrics()
    client_stats = streamingMonitorClient.fetch_client_stats()
    quality_map = streamingMonitorClient.fetch_quality_map()

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
    client_headers = ['Client ID', 'Rebuffer Time(s)', 'Rebuffer Count', 'QoE', 'Avg QoE']
    client_table_data = []

    total_qoe = 0.0
    client_count = 0

    for client_id, stats in client_stats.items():
        qoe = stats['qoe']
        if isinstance(qoe, list):
            avg_qoe = sum(qoe) / len(qoe) if qoe else 0.0
        else:
            avg_qoe = qoe

        total_qoe += avg_qoe
        client_count += 1

        client_table_data.append((
            client_id,
            stats['rebuffer_time'],
            stats['rebuffer_count'],
            qoe if not isinstance(qoe, list) else f"{avg_qoe:.2f}",  # 显示为当前值或平均值
            f"{avg_qoe:.2f}"
        ))

    # 在表格最后添加一行总平均 QoE
    overall_avg_qoe = total_qoe / client_count if client_count > 0 else 0.0
    client_table_data.append(('All Clients Avg', '', '', '', f"{overall_avg_qoe:.2f}"))

    client_table = tabulate(client_table_data, headers=client_headers, tablefmt="pretty", floatfmt=".2f")

    # Quality Map 表格
    quality_headers = ['Client ID'] + [f'Quality {i}' for i in range(max(len(q) for q in quality_map.values()))]
    quality_data = []

    for client_id, qualities in quality_map.items():
        row = [client_id]
        for i in range(len(quality_headers) - 1):  # 减1是因为 'Client ID' 占了第一个位置
            row.append(qualities.get(str(i), 'N/A'))  # 若缺失某一段，填 'N/A'
        quality_data.append(row)

    quality_table = tabulate(quality_data, headers=quality_headers, tablefmt="pretty")

    print("\nQuality Map Table:")
    print(quality_table)
    print("\nLink Metrics Table:")
    print(link_table)
    print("\nClient Stats Table:")
    print(client_table)
    time.sleep(1)
