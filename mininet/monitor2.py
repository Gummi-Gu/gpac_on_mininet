import re
import time
from collections import defaultdict

from tabulate import tabulate

import util

streamingMonitorClient=util.StreamingMonitorClient()
def mark2bw(x):
    if x == 10:
        return 8
    if x == 20:
        return 4
    if x == 30:
        return 2

summary_state=defaultdict(lambda: {
    'qoe': 0.0,
    'count': 0
})

total_bandwidth=20

while True:
    client_stats = streamingMonitorClient.fetch_client_stats()
    quality_map = streamingMonitorClient.fetch_quality_map()

    # Link Metrics 表格
    link_headers = ['Client ID', '12600_rate', '3150_rate', '785_rate','200_rate']
    link_data = []

    traffic_classes_band = streamingMonitorClient.fetch_traffic_classes_mark()
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

    traffic_classes_band = normalized

    # 阶段1：为每个(ip, 带宽)生成唯一标记
    ip_mark_mapping = defaultdict(lambda: defaultdict(lambda: {
        'mark': 0,
        'bw': 0
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
                    'bw': bw * total_bandwidth
                }
                current_mark += 10  # 步长10保证唯一

    for ip, item0 in ip_mark_mapping.items():
        link_data.append((ip,item0['12600']['bw'],item0['3150']['bw'],item0['785']['bw'],item0['200']['bw']))

    link_table = tabulate(link_data, headers=link_headers, tablefmt="pretty", floatfmt=".2f")

    # Client Stats 表格
    client_headers = ['Client ID', 'Rebuffer Time(s)', 'Rebuffer Count', 'QoE', 'Avg_Qoe']
    client_table_data = []

    for client_id, stats in client_stats.items():
        summary_state[client_id]['qoe'] += stats['qoe']
        summary_state[client_id]['count'] += 1
        avg_qoe=summary_state[client_id]['qoe']/summary_state[client_id]['count']
        client_table_data.append((
            client_id, stats['rebuffer_time'], stats['rebuffer_count'], stats['qoe'], avg_qoe
        ))

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
