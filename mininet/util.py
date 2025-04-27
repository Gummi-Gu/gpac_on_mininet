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
        client_id: str,
        avg_delay: float,
        avg_rate: float,
        latest_delay: float,
        latest_rate: float
    ) -> bool:
        """提交流轨道统计信息"""
        payload = {
            "track_id": track_id,
            "client_id": client_id,
            "avg_delay": avg_delay,
            "avg_rate": avg_rate,
            "latest_delay": latest_delay,
            "latest_rate": latest_rate
        }
        return self._send_data("track_stats", payload)

    def submit_link_metrics(self, client_id: str, delay: float, loss_rate: float, marks: dict) -> bool:
        """提交网络链路指标"""
        payload = {
            "client_id": client_id,
            "delay": delay,
            "loss_rate": loss_rate,
            "marks": marks
        }
        return self._send_data("link_metrics", payload)

    def submit_ip_maps(self, ip_maps: Dict[str, str]) -> bool:
        """提交 IP 映射表到服务器"""
        payload = ip_maps
        return self._send_data("update/ip_maps", payload)

    def fetch_traffic_classes_mark(self) -> Optional[Dict[str, Any]]:
        """获取 TRAFFIC_CLASSES_MARK 数据"""
        return self._get_data("get/traffic_classes_mark")

    def fetch_traffic_classes_delay(self) -> Optional[Dict[str, Any]]:
        """获取 TRAFFIC_CLASSES_DELAY 数据"""
        return self._get_data("get/traffic_classes_delay")
