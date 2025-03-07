import subprocess
import threading
import random
import time
import os

SERVER_IP="10.0.0.1"
PORT=1080
FILE_SIZES=10*1024*1024
REQUEST_INTERVAL=1


class RequestGenerator:
    def __init__(self):
        self.running = True
        self.classes = [12600, 3150, 785, 200]
        self.active_requests = {c: 0 for c in self.classes}
        self.completed_requests = {c: 0 for c in self.classes}
        self.total_transfer_time = {c: 0.0 for c in self.classes}
        self.total_data = {c: 0 for c in self.classes}
        self.last_transfer_time = {c: 0.0 for c in self.classes}
        self.lock = threading.Lock()

    def _fetch(self, url_type):
        """Execute single request and update metrics"""
        url = f'http://{SERVER_IP}:{PORT}/files/{url_type}.txt'
        curl_cmd = f'curl -s -o /dev/null "{url}" > /dev/null 2>&1'

        try:
            with self.lock:
                self.active_requests[url_type] += 1

            start_time = time.time()
            # 执行curl命令并检查返回值
            exit_code = os.system(curl_cmd)
            if exit_code != 0:
                raise RuntimeError(f"Curl failed with exit code {exit_code >> 8}")

            duration = time.time() - start_time

            with self.lock:
                self.active_requests[url_type] -= 1
                self.completed_requests[url_type] += 1
                self.total_transfer_time[url_type] += duration
                self.total_data[url_type] += FILE_SIZES[url_type]
                self.last_transfer_time[url_type] = duration

        except Exception as e:
            print(f"Request failed: {str(e)}")
        finally:
            with self.lock:
                if self.active_requests[url_type] > 0:
                    self.active_requests[url_type] -= 1

    def _generate_requests(self):
        """Generate random requests continuously"""
        # 执行ping测试
        os.system(f'ping -c 5 {SERVER_IP} > /dev/null 2>&1')

        while self.running:
            url_type = random.choice(self.classes)
            threading.Thread(target=self._fetch, args=(url_type,)).start()
            time.sleep(random.uniform(1, REQUEST_INTERVAL))

    def start(self):
        """Start request generator"""
        self.thread = threading.Thread(target=self._generate_requests)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop all requests"""
        self.running = False
        self.thread.join()


class TrafficMonitor:
    def __init__(self, request_gen):
        self.request_gen = request_gen
        self.running = True
        self.class_map = {
            12600: "1:10",
            3150: "1:20",
            785: "1:30",
            200: "1:40"
        }

    def _format_speed(self, speed_bps):
        """Format speed display"""
        if speed_bps >= 1e6:
            return f"{speed_bps / 1e6:.2f} Mbps"
        if speed_bps >= 1e3:
            return f"{speed_bps / 1e3:.2f} Kbps"
        return f"{speed_bps:.2f} bps"

    def _display(self):
        """Real-time monitoring display"""
        while self.running:
            os.system('clear')

            with self.request_gen.lock:
                data = {
                    cls: {
                        'active': self.request_gen.active_requests[cls],
                        'completed': self.request_gen.completed_requests[cls],
                        'total_time': self.request_gen.total_transfer_time[cls],
                        'total_data': self.request_gen.total_data[cls],
                        'last_time': self.request_gen.last_transfer_time[cls]
                    }
                    for cls in self.request_gen.classes
                }

            total_mb = 0
            print("\n=== Real-time Network Monitoring ===")

            for cls in self.request_gen.classes:
                # Calculate metrics
                avg_speed = (data[cls]['total_data'] / max(0.1, data[cls]['total_time']) * 8) if data[cls]['total_time'] > 0 else 0
                last_speed = (FILE_SIZES[cls] / max(0.1, data[cls]['last_time']) * 8) if data[cls]['last_time'] > 0 else 0
                avg_latency = (data[cls]['total_time'] / data[cls]['completed'] * 1000) if data[cls]['completed'] > 0 else 0
                cls_mb = data[cls]['total_data'] / (1024 * 1024)
                total_mb += cls_mb

                # Display metrics
                print(f"\n=== Class {cls} ===")
                print(f"Active: {data[cls]['active']}\tCompleted: {data[cls]['completed']}")
                print(f"Avg Speed: {self._format_speed(avg_speed)}")
                print(f"Avg Latency: {avg_latency:.2f} ms")
                print(f"Last Speed: {self._format_speed(last_speed)}")
                print(f"Total Data: {cls_mb:.2f} MB")
                print("TC Statistics:")

            print("\n=== Global Statistics ===")
            print(f"Total Transferred: {total_mb:.2f} MB")
            print("\nPress Ctrl+C to exit...")

            input("\nEnter")

    def start(self):
        """Start monitoring"""
        self.thread = threading.Thread(target=self._display)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop monitoring"""
        self.running = False
        self.thread.join()