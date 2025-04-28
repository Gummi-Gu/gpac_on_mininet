# client_monitor.py
import requests
import time
import random
from datetime import datetime
import json
from typing import Dict, Optional

class StreamingMonitorClient:
    def __init__(self, server_url: str = "http://localhost:5000"):
        self.base_url = server_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _send_data(self, endpoint: str, data: Dict) -> bool:
        """通用数据提交方法"""
        try:
            response = self.session.post(
                f"{self.base_url}/{endpoint}",
                data=json.dumps(data),
                timeout=5
            )
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"提交数据失败: {str(e)}")
            return False

    def submit_track_stats(
        self,
        track_id: str,
        avg_delay: float,
        avg_rate: float,
        latest_delay: float,
        latest_rate: float
    ) -> bool:
        """提交流轨道统计信息"""
        payload = {
            "track_id": track_id,
            "avg_delay": avg_delay,
            "avg_rate": avg_rate,
            "latest_delay": latest_delay,
            "latest_rate": latest_rate
        }
        return self._send_data("track_stats", payload)

    def submit_link_metrics(
        self,
        link_id: str,
        delay: float,
        loss_rate: float
    ) -> bool:
        """提交网络链路指标"""
        payload = {
            "link_id": link_id,
            "delay": delay,
            "loss_rate": loss_rate  # 小数形式，如0.05表示5%
        }
        return self._send_data("link_metrics", payload)

    def submit_chunk_quality(
        self,
        chunk_id: str,
        bitrate: int,
        resolution: str,
        buffer_time: float,
        quality_score: int
    ) -> bool:
        """提交视频分块质量数据"""
        payload = {
            "chunk_id": chunk_id,
            "bitrate": bitrate,
            "resolution": resolution,
            "buffer_time": buffer_time,
            "quality_score": quality_score
        }
        return self._send_data("chunk_quality", payload)

class DataGenerator:
    """模拟数据生成器"""

    @staticmethod
    def generate_track_stats(track_id: str) -> Dict:
        return {
            "track_id": track_id,
            "avg_delay": random.uniform(10, 100),
            "avg_rate": random.uniform(1, 5),
            "latest_delay": random.uniform(5, 150),
            "latest_rate": random.uniform(0.5, 6)
        }

    @staticmethod
    def generate_link_metrics(link_id: str) -> Dict:
        return {
            "link_id": link_id,
            "delay": random.uniform(20, 500),
            "loss_rate": random.uniform(0, 0.2)
        }

    @staticmethod
    def generate_chunk_quality(chunk_id: str) -> Dict:
        resolutions = ["1280x720", "1920x1080", "3840x2160"]
        return {
            "chunk_id": chunk_id,
            "bitrate": random.choice([2000, 4000, 8000]),
            "resolution": random.choice(resolutions),
            "buffer_time": random.uniform(1, 10),
            "quality_score": random.randint(50, 100)
        }