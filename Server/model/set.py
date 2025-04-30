import requests
import json

# 配置服务器地址
SERVER_URL = "http://127.0.0.1:5000"  # 改成你实际的IP和端口

def update(endpoint, data):
    url = f"{SERVER_URL}/update/{endpoint}"
    try:
        response = requests.post(url, json=data)
        #print(f"请求接口: {url}")
        #print(f"请求数据: {json.dumps(data, indent=2)}")
        #print(f"响应: {response.status_code} {response.json()}")
    except Exception as e:
        print(f"请求失败: {e}")

# --- 修改这里的内容来设置不同的配置 ---

# 1. 更新 traffic_classes_mark
traffic_classes_mark_update = {
    '10.0.0.2': {'port': 10086, '12600': 10, '3150': 20, '785': 30, '200': 30},
    '10.0.0.3': {'port': 10086, '12600': 0, '3150': 0, '785': 0, '200': 0},
    '10.0.0.4': {'port': 10086, '12600': 0, '3150': 0, '785': 0, '200': 0}

}

def increase_mark(ip, bit_class):
    """将指定IP的带宽类别值提升10个单位（最高30）"""
    if ip in traffic_classes_mark_update:
        if bit_class in traffic_classes_mark_update[ip]:
            current = traffic_classes_mark_update[ip][bit_class]
            new_value = min(current + 10, 30)  # 确保不超过30
            traffic_classes_mark_update[ip][bit_class] = new_value
            #print(f"已增加 {ip} 的 {bit_class} 到 {new_value}")
        #else:
            #print(f"错误：{bit_class} 不存在于 {ip}")
    else:
        print(f"错误：{ip} 不存在于配置")

def decrease_mark(ip, bit_class):
    """将指定IP的带宽类别值降低10个单位（最低10）"""
    if ip in traffic_classes_mark_update:
        if bit_class in traffic_classes_mark_update[ip]:
            current = traffic_classes_mark_update[ip][bit_class]
            new_value = max(current - 10, 10)  # 确保不低于10
            traffic_classes_mark_update[ip][bit_class] = new_value
            #print(f"已降低 {ip} 的 {bit_class} 到 {new_value}")
        #else:
            #print(f"错误：{bit_class} 不存在于 {ip}")
    else:
        print(f"错误：{ip} 不存在于配置")


# 2. 更新 traffic_classes_delay
traffic_classes_delay_update = {
    '10.0.0.2': {'client': 'client1', 'delay': 0, 'loss': 0},
    '10.0.0.3': {'client': 'client4', 'delay': 0, 'loss': 0},
    '10.0.0.4': {'client': 'client4', 'delay': 0, 'loss': 0}
}

# 3. 更新 rebuffer_config
rebuffer_config_update = {
    'client1': {'re_buffer': 1000000, 'play_buffer': 1000000},
    'client2': {'re_buffer': 1000000, 'play_buffer': 1000000},
    'client3': {'re_buffer': 1000000, 'play_buffer': 1000000}
}

def increase_buffer(client, buffer_type):
    """增加缓冲区大小（步长100,000，上限3,000,000）"""
    if client in rebuffer_config_update:
        if buffer_type in rebuffer_config_update[client]:
            current = rebuffer_config_update[client][buffer_type]
            new_value = min(current + 100000, 3000000)
            rebuffer_config_update[client][buffer_type] = new_value
            #print(f"已增加 {client} 的 {buffer_type}：{current} -> {new_value}")
        #else:
            #print(f"错误：{buffer_type} 不存在（应为 re_buffer/play_buffer）")
    else:
        print(f"错误：客户端 {client} 不存在")

def decrease_buffer(client, buffer_type):
    """减少缓冲区大小（步长100,000，下限1,000,000）"""
    if client in rebuffer_config_update:
        if buffer_type in rebuffer_config_update[client]:
            current = rebuffer_config_update[client][buffer_type]
            new_value = max(current - 100000, 1000000)
            rebuffer_config_update[client][buffer_type] = new_value
            #print(f"已减少 {client} 的 {buffer_type}：{current} -> {new_value}")
        #else:
            #print(f"错误：{buffer_type} 不存在（应为 re_buffer/play_buffer）")
    else:
        print(f"错误：客户端 {client} 不存在")

# 4. 更新 quality_map
quality_map_update = {
    'client1': {0: 0, 1: 1, 2: 2, 3: 3},
    'client2': {0: 0, 1: 1, 2: 2, 3: 3},
    'client3': {0: 0, 1: 1, 2: 2, 3: 3}
}

def increase_quality(client, level):
    """提升客户端某层级的质量等级（上限3）"""
    if client in quality_map_update:
        if level in quality_map_update[client]:
            current = quality_map_update[client][level]
            new_value = min(current + 1, 3)
            quality_map_update[client][level] = new_value
            #print(f"已提升 {client} 的层级 {level}：{current} -> {new_value}")
        #else:
            #print(f"错误：层级 {level} 不存在于 {client}")
    else:
        print(f"错误：客户端 {client} 不存在")

def decrease_quality(client, level):
    """降低客户端某层级的质量等级（下限0）"""
    if client in quality_map_update:
        if level in quality_map_update[client]:
            current = quality_map_update[client][level]
            new_value = max(current - 1, 0)
            quality_map_update[client][level] = new_value
            #print(f"已降低 {client} 的层级 {level}：{current} -> {new_value}")
        #else:
            #print(f"错误：层级 {level} 不存在于 {client}")
    else:
        print(f"错误：客户端 {client} 不存在")
# --- 选择你要更新的内容，执行更新 ---

def reset():
    # 更新 traffic_classes_mark
    update("traffic_classes_mark", traffic_classes_mark_update)

    # 更新 traffic_classes_delay
    update("traffic_classes_delay", traffic_classes_delay_update)

    # 更新 rebuffer_config
    update("rebuffer_config", rebuffer_config_update)

    # 更新 quality_map
    update("quality_map", quality_map_update)

reset()