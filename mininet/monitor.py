import re
import time
from collections import defaultdict
from collections import defaultdict, deque
from tabulate import tabulate

import util


streamingMonitorClient=util.StreamingMonitorClient()

latest_rate_history = {}
latest_delay_history = {}
total_bandwidth=24/8

# 改为保存最近10秒的 (timestamp, delay, rate)
summary_state = defaultdict(lambda: deque())
WINDOW_SECONDS = 10

while True:

    track_stats = streamingMonitorClient.fetch_track_stats()
    summary_rate_stats = streamingMonitorClient.fetch_summary_rate_stats()
    bitrate_stats = streamingMonitorClient.fetch_bitrate_stats()
    link_metrics = streamingMonitorClient.fetch_link_metrics()
    client_stats = streamingMonitorClient.fetch_client_stats()


    track_table_data = []
    client_summary = {}

    for track_id, clients in track_stats.items():
        for client_id, stats in clients.items():
            if track_id == 'default':
                continue
            utilization = (stats['latest_rate'] / total_bandwidth) * 100  # 假设最大带宽是20

            prev_key = (track_id, client_id)
            prev_latest_rate = latest_rate_history.get(prev_key, 0.0)  # 上一秒的速率
            prev_latest_delay = latest_delay_history.get(prev_key, 0.0)  # 上一秒的时延

            # 更新历史数据
            latest_rate_history[prev_key] = stats['latest_rate']
            latest_delay_history[prev_key] = stats['latest_delay']

            if client_id not in client_summary:
                client_summary[client_id] = {
                    'total_delay': 0.0,
                    'total_latest_rate': 0.0,
                    'total_utilization': 0.0,
                    'track_count': 0,
                }
            summary = client_summary[client_id]
            summary['total_delay'] += stats['latest_delay']
            summary['total_latest_rate'] += stats['latest_rate']
            summary['total_utilization'] += utilization
            summary['track_count'] += 1
            client_id_num = ''.join(re.findall(r'\d+', client_id))
            # 加入 prev_latest_delay 和 prev_latest_rate
            if client_id == 'client1':
                track_table_data.append((
                    track_id, str(client_id_num),
                    f"{stats['avg_delay']:.1f}ms", f"{stats['avg_rate']:.1f}MB/s",
                    f"{stats['latest_delay']:.1f}ms", f"{prev_latest_delay:.1f}ms",  # 当前和上一秒的时延
                    f"{stats['latest_rate']:.1f}MB/s", f"{prev_latest_rate:.1f}MB/s",  # 当前和上一秒的速率
                    stats['resolution'], f"{utilization:.2f}%"
                ))

    for client_id, stats in client_summary.items():
        prev_key = ('sum', client_id)
        prev_latest_rate = latest_rate_history.get(prev_key, 0.0)  # 上一秒的速率
        prev_latest_delay = latest_delay_history.get(prev_key, 0.0)  # 上一秒的时延
        #summary_rate_stats[client_id]['size']=summary_rate_stats[client_id]['size']/summary_rate_stats[client_id]['time']*1e3
        utilization = (summary_rate_stats[client_id]['size']/summary_rate_stats[client_id]['time']*1000 / total_bandwidth) * 100
        latest_rate_history[prev_key] = summary_rate_stats[client_id]['size']
        latest_delay_history[prev_key] = summary_rate_stats[client_id]['time']
        client_id_num = ''.join(re.findall(r'\d+', client_id))
        now = time.time()
        summary_state[client_id].append(
            (now, summary_rate_stats[client_id]['time'], summary_rate_stats[client_id]['size']))
        # 保留最近10秒内的记录
        while summary_state[client_id] and now - summary_state[client_id][0][0] > WINDOW_SECONDS:
            summary_state[client_id].popleft()
        # 计算过去10秒内的平均
        records = summary_state[client_id]
        if records:
            avg_delay = sum(d for _, d, _ in records) / len(records)
            avg_rate = sum(r for _, _, r in records) / len(records)
        else:
            avg_delay = avg_rate = 0.0
        track_table_data.append((
            'sum', str(client_id_num),
            f"{avg_delay:.1f}ms", f"{avg_rate:.1f}MB/s",
            f"{summary_rate_stats[client_id]['time']:.1f}ms", f"{prev_latest_delay:.1f}ms",  # sum行上一秒delay为0.0
            f"{summary_rate_stats[client_id]['size']:.1f}MB/s", f"{prev_latest_rate:.1f}MB/s",  # sum行上一秒rate为0.0
            0.0, f"{utilization:.2f}%"
        ))

    # headers也同步增加
    track_headers = ['Trk', 'Clt', 'AvgDly', 'AvgRt', 'LatDly','PrvDly','LatRt', 'PrvRat','BitRt', 'Uti']
    track_table = tabulate(track_table_data, headers=track_headers, tablefmt="pretty")

    # Bitrate Stats 表格
    bitrate_headers = ['Bitrat', 'CltID', 'AvgDly', 'AvgRat', 'LatDly',
                       'LatRat', 'Avgsize','Utiliz']
    bitrate_table_data = []
    bitrate_summary = {}

    for bitrate, clients in bitrate_stats.items():
        if bitrate == 'default':
            continue
        for client_id, stats in clients.items():
            utilization = (stats['latest_rate'] / total_bandwidth) * 100  
            if client_id not in bitrate_summary:
                bitrate_summary[client_id] = {
                    'total_delay': 0.0,
                    'total_latest_rate': 0.0,
                    'total_utilization': 0.0,
                    'bitrate_count': 0,
                }
            summary = bitrate_summary[client_id]
            summary['total_delay'] += stats['latest_delay']
            summary['total_latest_rate'] += stats['latest_rate']
            summary['total_utilization'] += utilization
            summary['bitrate_count'] += 1
            bitrate_table_data.append((
                bitrate, client_id, f"{stats['avg_delay']:.1f}ms", f"{stats['avg_rate']:.1f}MB/s",
                f"{stats['latest_delay']:.1f}ms", f"{stats['latest_rate']:.1f}MB/s",
                f"{stats['avg_size']:.1f}KB",
                f"{utilization:.2f}%"
            ))

    bitrate_table = tabulate(bitrate_table_data, headers=bitrate_headers, tablefmt="pretty", floatfmt=".2f")

    print("\nTrack Stats Table:")
    print(track_table)
    print("\nBitrate Stats Table:")
    print(bitrate_table)
    #print("\nLink Metrics Table:")
    #print(link_table)
    #print("\nClient Stats Table:")
    #print(client_table)
    time.sleep(1)
