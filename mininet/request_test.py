import sys
import time

import requests
# 从命令行获取 URL 参数
if len(sys.argv) > 1:
    url = sys.argv[1]
else:
    print("Usage: python script.py <URL>")
    sys.exit(1)


# 请求并处理异常
try:
    response = requests.get(url)
    print(response.status_code)
except requests.exceptions.Timeout as e:
    print('Timeout error:', e)
except requests.exceptions.TooManyRedirects as e:
    print('Too many redirects:', e)
except requests.exceptions.RequestException as e:
    print('Request error:', e)
except Exception as e:
    print('Unknown error:', e)
