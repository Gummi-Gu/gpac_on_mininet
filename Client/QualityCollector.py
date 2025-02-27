import pandas as pd
import xml.etree.ElementTree as ET
import os

class VideoQualityMetrics:
    def __init__(self):
        # 用DataFrame来存储数据，包含时间戳列
        self.codec_h264=True
        self.codec_h265=False
        self.cpu_score=10000
        self.gpu_score=10000
        self.memory_size=8
        self.data = pd.DataFrame(columns=[
            'timestamp', 'resolution', 'frame_rate_std', 'codec_h264', 'codec_h265',
            'real_time_bandwidth', 'network_delay_change', 'packet_loss',
            'buffer_size', 'cpu_score', 'gpu_score', 'memory_size'
        ])

    def collect_data(self, timestamp, resolution=None, frame_rate_std=None, codec_h264=None, codec_h265=None,
                     real_time_bandwidth=None, network_delay_change=None, packet_loss=None,
                     buffer_size=None, cpu_score=None, gpu_score=None, memory_size=None):
        # 如果没有数据，直接添加新的一行
        if timestamp not in self.data['timestamp'].values:
            new_row = pd.DataFrame({
                'timestamp': [timestamp],
                'resolution': [resolution],
                'frame_rate_std': [frame_rate_std],
                'codec_h264': [codec_h264],
                'codec_h265': [codec_h265],
                'real_time_bandwidth': [real_time_bandwidth],
                'network_delay_change': [network_delay_change],
                'packet_loss': [packet_loss],
                'buffer_size': [buffer_size],
                'cpu_score': [cpu_score],
                'gpu_score': [gpu_score],
                'memory_size': [memory_size]
            })
            self.data = pd.concat([self.data, new_row], ignore_index=True)
        else:
            # 如果该时间戳的数据已存在，更新字段
            if resolution is not None:
                self.data.loc[self.data['timestamp'] == timestamp, 'resolution'] = resolution
            if frame_rate_std is not None:
                self.data.loc[self.data['timestamp'] == timestamp, 'frame_rate_std'] = frame_rate_std
            if codec_h264 is not None:
                self.data.loc[self.data['timestamp'] == timestamp, 'codec_h264'] = codec_h264
            if codec_h265 is not None:
                self.data.loc[self.data['timestamp'] == timestamp, 'codec_h265'] = codec_h265
            if real_time_bandwidth is not None:
                self.data.loc[self.data['timestamp'] == timestamp, 'real_time_bandwidth'] = real_time_bandwidth
            if network_delay_change is not None:
                self.data.loc[self.data['timestamp'] == timestamp, 'network_delay_change'] = network_delay_change
            if packet_loss is not None:
                self.data.loc[self.data['timestamp'] == timestamp, 'packet_loss'] = packet_loss
            if buffer_size is not None:
                self.data.loc[self.data['timestamp'] == timestamp, 'buffer_size'] = buffer_size
            if cpu_score is not None:
                self.data.loc[self.data['timestamp'] == timestamp, 'cpu_score'] = cpu_score
            if gpu_score is not None:
                self.data.loc[self.data['timestamp'] == timestamp, 'gpu_score'] = gpu_score
            if memory_size is not None:
                self.data.loc[self.data['timestamp'] == timestamp, 'memory_size'] = memory_size

    def get_data(self):
        return self.data

    def merge_data(self):
        # 合并每个时间戳的数据，选择每个字段的最后一个非空值
        self.data = self.data.groupby('timestamp', as_index=False).agg({
            'resolution': 'last',
            'frame_rate_std': 'last',
            'codec_h264': 'last',
            'codec_h265': 'last',
            'real_time_bandwidth': 'last',
            'network_delay_change': 'last',
            'packet_loss': 'last',
            'buffer_size': 'last',
            'cpu_score': 'last',
            'gpu_score': 'last',
            'memory_size': 'last'
        })

    def to_xml(self):
        # 创建 XML 根节点
        root = ET.Element("VideoMetricsData")

        # 为每一行数据创建一个<Metric>节点
        for _, row in self.data.iterrows():
            metric = ET.SubElement(root, "Metric", timestamp=str(row['timestamp']))

            for col in self.data.columns[1:]:
                ET.SubElement(metric, col).text = str(row[col])

        # 生成XML字符串并返回
        tree = ET.ElementTree(root)
        xml_str = ET.tostring(root, encoding='unicode', method='xml')
        return xml_str

    def save_to_file(self):
        # 获取文件夹路径
        folder_path = "video_metrics_files"

        # 如果文件夹不存在，则创建它
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 查找当前最大的文件编号
        files = os.listdir(folder_path)
        max_index = 0
        for file in files:
            if file.startswith("VideoMetricsData_") and file.endswith(".xml"):
                try:
                    index = int(file.split('_')[-1].split('.')[0])
                    if index > max_index:
                        max_index = index
                except ValueError:
                    continue

        # 新文件的序号
        new_file_index = max_index + 1
        new_file_name = f"VideoMetricsData_{new_file_index}.xml"
        new_file_path = os.path.join(folder_path, new_file_name)

        # 将XML写入文件
        with open(new_file_path, "w") as f:
            xml_str = self.to_xml()
            f.write(xml_str)

        print(f"数据已保存到文件: {new_file_path}")
