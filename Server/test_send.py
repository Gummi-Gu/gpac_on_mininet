import subprocess
import threading
import random
import time
import os

SERVER_IP="10.0.0.1"
PORT=10086
FILE_SIZES=1*1024*1024
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
        curl_cmd = f'curl -s {url} -o NUL'

        try:
            with self.lock:
                self.active_requests[url_type] += 1

            start_time = time.time()
            # 执行curl命令并检查返回值
            exit_code = os.system(curl_cmd)
            if exit_code != 0:
                raise RuntimeError(f"Curl failed with exit code {exit_code}")
            #time.sleep(random.uniform(0.01,0.05))
            duration = time.time() - start_time

            with self.lock:
                self.active_requests[url_type] -= 1
                self.completed_requests[url_type] += 1
                self.total_transfer_time[url_type] += duration
                self.total_data[url_type] += FILE_SIZES
                self.last_transfer_time[url_type] = duration

        except Exception as e:
            print(f"Request failed: {str(e)}")

    def _generate_requests(self):
        """Generate random requests continuously"""
        # 执行ping测试
        os.system(f'ping -c 5 {SERVER_IP} > /dev/null 2>&1')

        while self.running:
            url_type = random.choice(self.classes)
            self._fetch(url_type)
            #threading.Thread(target=self._fetch, args=(url_type,)).start()
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
        """Real-time monitoring display with tabulate tables"""
        from tabulate import tabulate

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
            class_table = []
            tc_table = []

            # Prepare class data for tables
            for cls in self.request_gen.classes:
                # Calculate metrics
                avg_speed = (data[cls]['total_data'] / max(0.1, data[cls]['total_time']) * 8) if data[cls]['total_time'] > 0 else 0
                last_speed = (FILE_SIZES / max(0.001, data[cls]['last_time']) * 8) if data[cls]['last_time'] > 0 else 0
                avg_latency = (data[cls]['total_time'] / data[cls]['completed'] * 1000) if data[cls]['completed'] > 0 else 0
                cls_mb = data[cls]['total_data'] / (1024 * 1024)
                total_mb += cls_mb

                # Format for table display
                class_table.append([
                    cls,
                    data[cls]['active'],
                    data[cls]['completed'],
                    f"{self._format_speed(avg_speed):>8}",
                    f"{avg_latency:.2f} ms",
                    f"{self._format_speed(last_speed):>8}",
                    f"{data[cls]['last_time']*1000:.2f}ms",
                    f"{cls_mb:.2f} MB"
                ])

            # 构建表格输出
            print("\n=== Real-time Network Monitoring ===")

            # 主监控表
            print("\n--- Class Statistics ---")
            print(tabulate(
                class_table,
                headers=['Cls', 'Act', 'Fin', 'AvgSpd', 'AvgLat', 'LastSpd', 'LastTime','Data'],
                tablefmt='grid',
                stralign='right'
            ))

            # 全局统计
            print("\n=== Global Statistics ===")
            print(tabulate(
                [[f"{total_mb:.2f} MB"]],
                headers=['Total Transferred'],
                tablefmt='grid'
            ))

            print("\nPress Ctrl+C to exit...")
            input("\nPress Enter to refresh...")
    def start(self):
        """Start monitoring"""
        self.thread = threading.Thread(target=self._display)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop monitoring"""
        self.running = False
        self.thread.join()


if __name__ == '__main__':
    try:
        request_gen = RequestGenerator()
        monitor = TrafficMonitor(request_gen)

        request_gen.start()
        monitor.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        request_gen.stop()
        monitor.stop()
