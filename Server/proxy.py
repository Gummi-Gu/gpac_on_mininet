from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import requests
from urllib.parse import urljoin

# 配置目标服务器地址 (需要替换为你的实际服务器地址)
TARGET_SERVER = "http://ip1:port"  # 示例: "http://192.168.1.100:80"


class ProxyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            # 构造目标URL
            target_url = urljoin(TARGET_SERVER, self.path)

            # 转发请求头 (过滤掉不需要的标头)
            headers = {key: self.headers[key] for key in self.headers if key.lower() not in ['host', 'accept-encoding']}

            # 发送请求到目标服务器
            response = requests.get(
                target_url,
                headers=headers,
                stream=True,
                timeout=10
            )

            # 设置响应状态码
            self.send_response(response.status_code)

            # 转发响应头 (过滤掉不需要的标头)
            for header in response.headers:
                if header.lower() not in ['transfer-encoding', 'connection', 'content-encoding']:
                    self.send_header(header, response.headers[header])
            self.end_headers()

            # 流式传输响应内容
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    self.wfile.write(chunk)

        except requests.exceptions.RequestException as e:
            self.send_error(502, f"Bad Gateway: {str(e)}")
        except Exception as e:
            self.send_error(500, f"Internal Server Error: {str(e)}")


def run_server(port=8000):
    server_address = ('', port)
    httpd = ThreadingHTTPServer(server_address, ProxyHandler)
    print(f"Starting reverse proxy server on port {port}...")
    httpd.serve_forever()


if __name__ == '__main__':
    run_server()