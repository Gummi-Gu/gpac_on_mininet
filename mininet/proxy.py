import sys
import logging
from flask import Flask, request, Response
from flask_cors import CORS, cross_origin
import requests

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app,
     origins="*",
     methods=["GET", "HEAD", "OPTIONS"],
     allow_headers=["Range", "Content-Type"],
     expose_headers=["Content-Range", "Content-Length", "Content-Type"],
     max_age=86400)

app.config.update({
    'TARGET_SERVER': None,
    'TARGET_PORT': None,
    'STREAM_CHUNK_SIZE': 1024 * 128  # 128KB
})


@app.route('/', defaults={'path': ''}, methods=['GET', 'HEAD'])
@app.route('/<path:path>', methods=['GET', 'HEAD'])
@cross_origin()  # 为每个路由单独应用CORS
def proxy(path):
    try:
        # 构建目标URL
        target_url = f"http://{app.config['TARGET_SERVER']}:{app.config['TARGET_PORT']}/{path}"
        logger.info(f"Proxying {request.method} to: {target_url}")

        # 转发请求头（保留Range头）
        headers = {
            key: value
            for key, value in request.headers
            if key.lower() not in ['host', 'accept-encoding']
        }

        # 发起代理请求
        resp = requests.request(
            method=request.method,
            url=target_url,
            headers=headers,
            stream=True,
            timeout=30
        )

        # 构建响应头
        excluded_headers = ['content-encoding', 'connection', 'transfer-encoding']
        response_headers = {
            key: value
            for key, value in resp.raw.headers.items()
            if key.lower() not in excluded_headers
        }

        # 流式传输生成器
        def generate():
            try:
                for chunk in resp.iter_content(app.config['STREAM_CHUNK_SIZE']):
                    if chunk:
                        yield chunk
            finally:
                resp.close()
                logger.debug("Connection closed")

        return Response(
            generate(),
            status=resp.status_code,
            headers=response_headers
        )

    except requests.exceptions.RequestException as e:
        logger.error(f"Proxy error: {str(e)}")
        return Response(f"Bad Gateway: {str(e)}", status=502)


def run_server(target_server, target_port, proxy_port=8000):
    app.config['TARGET_SERVER'] = target_server
    app.config['TARGET_PORT'] = target_port

    # 启动服务器
    from werkzeug.serving import ThreadedWSGIServer
    server = ThreadedWSGIServer('0.0.0.0', proxy_port, app)
    logger.info(f"Starting CORS-enabled proxy on :{proxy_port}")
    server.serve_forever()


if __name__ == '__main__':
    if len(sys.argv) < 3:
        logger.error("Usage: python proxy.py <target-server> <target-port> [proxy-port]")
        sys.exit(1)

    run_server(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8000)