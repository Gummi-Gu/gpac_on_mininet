# client_monitor.py
import requests
import time
import random
from datetime import datetime
import json
from typing import Dict, Optional, Any


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

    def _get_data(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """通用数据获取方法"""
        try:
            response = self.session.get(
                f"{self.base_url}/{endpoint}",
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
            else:
                print(f"获取数据失败，状态码: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"获取数据失败: {str(e)}")
            return None

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

    def submit_headmap(self,data) -> bool:
        payload = {
            "data": data
        }
        return self._send_data("update_heatmap", payload)

    def submit_client_stats(
        self,
        client_id: str,
        rebuffer_time: float,
        rebuffer_count: int,
        qoe: float
    ) -> bool:
        """提交网络链路指标"""
        payload = {
            'client_id': client_id,
            'rebuffer_time': rebuffer_time,
            'rebuffer_count': rebuffer_count,
            'qoe': qoe
        }
        return self._send_data("client_stats", payload)

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
            "loss_rate": loss_rate,  # 小数形式，如0.05表示5%
        }
        return self._send_data("link_metrics", payload)

    def submit_chunk_qualities(self, resolutions,client) -> bool:
        """批量提交视频分块的分辨率数据"""
        payload = {str(i+1): res for i,res in enumerate(resolutions)}
        payload['client']=client
        return self._send_data("chunk_quality", payload)

    def fetch_client_stats(self) -> Optional[Dict[str, Any]]:
        return self._get_data("get/client_stats")

    def fetch_buffer(self) -> Optional[Dict[str, Any]]:
        return self._get_data("get/rebuffer_config")

    def fetch_quality(self) -> Optional[Dict[int, int]]:
        return self._get_data("get/quality_map")

    def fetch_ip_maps(self) -> Optional[Dict[str, str]]:
        """从服务器拉取 IP 映射表"""
        return self._get_data("get/ip_maps")
