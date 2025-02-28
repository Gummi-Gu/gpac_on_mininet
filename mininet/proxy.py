import sys
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
from urllib.parse import urlparse, urljoin


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProxyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            # 解析目标服务器URL
            parsed_url = urlparse(urljoin(self.server.target_server, self.path))
            target_host = parsed_url.hostname
            target_port = parsed_url.port or 80  # 默认为 80

            # 日志记录目标服务器的请求信息
            logger.info(f"Forwarding request to {target_host}:{target_port}{parsed_url.path}")

            # 设置连接到目标服务器
            conn = http.client.HTTPConnection(target_host, target_port, timeout=10)

            # 转发请求头
            headers = {key: self.headers[key] for key in self.headers if key.lower() not in ['host', 'accept-encoding']}

            # 向目标服务器发送GET请求
            conn.request("GET", parsed_url.path, headers=headers)

            # 获取目标服务器的响应
            response = conn.getresponse()

            # 日志记录目标服务器的响应状态
            logger.info(f"Received response {response.status} from {target_host}:{target_port}")

            # 设置响应状态码
            self.send_response(response.status)

            # 转发响应头
            for header, value in response.getheaders():
                if header.lower() not in ['transfer-encoding', 'connection', 'content-encoding']:
                    self.send_header(header, value)
            self.end_headers()

            # 流式传输响应内容
            while chunk := response.read(8192):
                self.wfile.write(chunk)

            # 关闭连接
            conn.close()

        except Exception as e:
            # 捕获异常并记录日志
            logger.error(f"Error processing request: {str(e)}")
            self.send_error(502, f"Bad Gateway: {str(e)}")


def run_server(target_server, target_port, port=8000):
    server_address = ('', port)
    httpd = ThreadingHTTPServer(server_address, ProxyHandler)
    httpd.target_server = target_server  # 将目标服务器地址传递给处理器
    httpd.target_port = target_port  # 将目标服务器端口传递给处理器
    logger.info(f"Starting reverse proxy server on port {port} with target {target_server}:{target_port}...")
    httpd.serve_forever()


if __name__ == '__main__':
    # 从命令行参数获取目标服务器的地址和端口
    if len(sys.argv) < 3:
        logger.error("Usage: python proxy_server.py <target-server-url> <target-server-port>")
        sys.exit(1)

    target_server = sys.argv[1]  # 获取目标服务器地址
    target_port = int(sys.argv[2])  # 获取目标服务器端口

    # 启动代理服务器
    run_server(target_server, target_port)
