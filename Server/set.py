import requests
import json

# 配置服务器地址
SERVER_URL = "http://127.0.0.1:5000"  # 改成你实际的IP和端口

def update(endpoint, data):
    url = f"{SERVER_URL}/update/{endpoint}"
    try:
        response = requests.post(url, json=data)
        print(f"请求接口: {url}")
        print(f"请求数据: {json.dumps(data, indent=2)}")
        print(f"响应: {response.status_code} {response.json()}")
    except Exception as e:
        print(f"请求失败: {e}")

# --- 修改这里的内容来设置不同的配置 ---

# 1. 更新 traffic_classes_mark
traffic_classes_mark_update = {
    '10.0.0.2': {'port': 10086, '12600': 10, '3150': 20, '785': 30, '200': 30},
    '10.0.0.3': {'port': 10086, '12600': 10, '3150': 20, '785': 30, '200': 30},
    '10.0.0.4': {'port': 10086, '12600': 10, '3150': 20, '785': 30, '200': 30}

}

# 2. 更新 traffic_classes_delay
traffic_classes_delay_update = {
    '10.0.0.2': {'client': 'client1', 'delay': 1, 'loss': 1},
    '10.0.0.3': {'client': 'client4', 'delay': 1, 'loss': 2},
    '10.0.0.4': {'client': 'client4', 'delay': 1, 'loss': 2}
}

# 3. 更新 rebuffer_config
rebuffer_config_update = {
    'client1': {'re_buffer': 1000000, 'play_buffer': 1000000},
    'client2': {'re_buffer': 1000000, 'play_buffer': 1000000},
    'client3': {'re_buffer': 1000000, 'play_buffer': 1000000}
}

# 4. 更新 quality_map
quality_map_update = {
    'client1': {0: 1, 1: 2, 2: 3, 3: 3},
    'client2': {0: 0, 1: 1, 2: 2, 3: 3},
    'client3': {0: 0, 1: 1, 2: 2, 3: 3}
}

# --- 选择你要更新的内容，执行更新 ---

# 更新 traffic_classes_mark
update("traffic_classes_mark", traffic_classes_mark_update)

# 更新 traffic_classes_delay
update("traffic_classes_delay", traffic_classes_delay_update)

# 更新 rebuffer_config
update("rebuffer_config", rebuffer_config_update)

# 更新 quality_map
update("quality_map", quality_map_update)
