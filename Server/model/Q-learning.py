import json
import time

import requests
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque, defaultdict
from Server.util import StreamingMonitorClient
from mininet.proxy import client_id

streamingMonitorClient=StreamingMonitorClient()
# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def fairness(q):
    """
    计算 Jain's Fairness Index
    参数:
        q: 一维数组或列表，表示各个实体的指标（比如带宽、速率等）
    返回:
        一个0到1之间的数，越接近1越公平
    """
    q = np.array(q)
    numerator = np.sum(q) ** 2
    denominator = len(q) * np.sum(q ** 2)
    if denominator == 0:
        return 0  # 防止除零
    return numerator / denominator


import numpy as np


def log_normalize(x, min_val=0, max_val=1, eps=1e-8):
    """
    对x进行log归一化，映射到[0, 1]之间。
    参数:
        x: 输入值（可以是标量或数组）
        min_val: log归一化的最小值（如果不给，会根据x动态计算）
        max_val: log归一化的最大值（如果不给，会根据x动态计算）
        eps: 防止log(0)出现的微小值
    返回:
        归一化后的值
    """
    x = np.array(x)  # 支持单个值或数组

    # 加eps防止出现log(0)
    log_x = np.log(x + eps)

    # 自动确定归一化区间
    if min_val is None:
        min_val = np.min(log_x)
    if max_val is None:
        max_val = np.max(log_x)

    norm_x = (log_x - min_val) / (max_val - min_val + eps)
    return norm_x


class DQN(nn.Module):
    """Q网络结构定义"""

    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.fc(x)


class StreamingOptimizer:
    def __init__(self):
        # 初始化系统配置
        self.traffic_classes_mark = {
            '10.0.0.2': {'port': 10086, '12600': 10, '3150': 30, '785': 30, '200': 30},
            '10.0.0.3': {'port': 10086, '12600': 10, '3150': 20, '785': 30, '200': 30},
            '10.0.0.4': {'port': 10086, '12600': 10, '3150': 20, '785': 30, '200': 30}
        }

        self.quality_map = {
            'client1': {0: 1, 1: 2, 2: 3, 3: 3},
            'client2': {0: 0, 1: 1, 2: 2, 3: 3},
            'client3': {0: 0, 1: 1, 2: 2, 3: 3}
        }

        self.rebuffer_config = {
            'client1': {'re_buffer': 1000000, 'play_buffer': 1000000},
            'client2': {'re_buffer': 1000000, 'play_buffer': 1000000},
            'client3': {'re_buffer': 1000000, 'play_buffer': 1000000}
        }

        self.bitrate_stats = defaultdict(lambda: defaultdict(lambda: {
            'avg_delay': 0.0,
            'avg_rate': 0.0,
            'latest_delay': 0.0,
            'latest_rate': 0.0,
            'resolution': '',
            'last_update': None
        }))

        self.track_stats = defaultdict(lambda: defaultdict(lambda: {
            'avg_delay': 0.0,
            'avg_rate': 0.0,
            'latest_delay': 0.0,
            'latest_rate': 0.0,
            'resolution': '',
            'last_update': None
        }))

        self.summary_rate_stats = {
            'client1': {'size': 0.0, 'time': 0.0},
            'client2': {'size': 0.0, 'time': 0.0},
            'client3': {'size': 0.0, 'time': 0.0}
        }

        self.client_stats = defaultdict(lambda: {
            'rebuffer_time': 0.0,
            'rebuffer_count': 0.0,
            'qoe': 0.0,
            'last_update': None
        })

        self.rebuff_event=0.0

        # 强化学习参数
        self.state_size = 35  # 状态向量维度（根据实际特征计算）
        self.action_size = 60  # 总动作数量
        self.batch_size = 32
        self.buffer_size = 10000
        self.discount_factor = 0.95
        self.learning_rate = 0.001
        self.exploration_rate = 1.0
        self.min_exploration_rate = 0.01
        self.exploration_decay = 0.995


        # 初始化Q网络和目标网络
        self.q_network = DQN(self.state_size, self.action_size).to(device)
        self.target_network = DQN(self.state_size, self.action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # 优化器和损失函数
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        # 经验回放池
        self.replay_buffer = deque(maxlen=self.buffer_size)

    # --- 辅助函数（与之前实现的相同）---
    def increase_mark(self,ip, bit_class):
        """将指定IP的带宽类别值提升10个单位（最高30）"""
        if ip in self.traffic_classes_mark:
            if bit_class in self.traffic_classes_mark[ip]:
                current = self.traffic_classes_mark[ip][bit_class]
                new_value = min(current + 10, 30)  # 确保不超过30
                self.traffic_classes_mark[ip][bit_class] = new_value
                print(f"已增加 {ip} 的 {bit_class} 到 {new_value}")
                update("traffic_classes_mark", self.traffic_classes_mark)
            else:
                print(f"错误：{bit_class} 不存在于 {ip}")
        else:
            print(f"错误：{ip} 不存在于配置")

    def decrease_mark(self, ip, bit_class):
        """将指定IP的带宽类别值降低10个单位（最低10）"""
        if ip in self.traffic_classes_mark:
            if bit_class in self.traffic_classes_mark[ip]:
                current = self.traffic_classes_mark[ip][bit_class]
                new_value = max(current - 10, 10)  # 确保不低于10
                self.traffic_classes_mark[ip][bit_class] = new_value
                print(f"已降低 {ip} 的 {bit_class} 到 {new_value}")
                update("traffic_classes_mark", self.traffic_classes_mark)
            else:
                print(f"错误：{bit_class} 不存在于 {ip}")
        else:
            print(f"错误：{ip} 不存在于配置")

    def increase_quality(self, client, level):
        """提升客户端某层级的质量等级（上限3）"""
        if client in self.quality_map:
            if level in self.quality_map[client]:
                current = self.quality_map[client][level]
                new_value = min(current + 1, 3)
                self.quality_map[client][level] = new_value
                print(f"已提升 {client} 的层级 {level}：{current} -> {new_value}")
                update("quality_map", self.quality_map)
            else:
                print(f"错误：层级 {level} 不存在于 {client}")
        else:
            print(f"错误：客户端 {client} 不存在")

    def decrease_quality(self, client, level):
        """降低客户端某层级的质量等级（下限0）"""
        if client in self.quality_map:
            if level in self.quality_map[client]:
                current = self.quality_map[client][level]
                new_value = max(current - 1, 0)
                self.quality_map[client][level] = new_value
                print(f"已降低 {client} 的层级 {level}：{current} -> {new_value}")
                update("quality_map", self.quality_map)
            else:
                print(f"错误：层级 {level} 不存在于 {client}")
        else:
            print(f"错误：客户端 {client} 不存在")

    def increase_buffer(self, client, buffer_type):
        """增加缓冲区大小（步长100,000，上限3,000,000）"""
        if client in self.rebuffer_config:
            if buffer_type in self.rebuffer_config[client]:
                current = self.rebuffer_config[client][buffer_type]
                new_value = min(current + 100000, 3000000)
                self.rebuffer_config[client][buffer_type] = new_value
                print(f"已增加 {client} 的 {buffer_type}：{current} -> {new_value}")
                update("rebuffer_config", self.rebuffer_config)
            else:
                print(f"错误：{buffer_type} 不存在（应为 re_buffer/play_buffer）")
        else:
            print(f"错误：客户端 {client} 不存在")

    def decrease_buffer(self, client, buffer_type):
        """减少缓冲区大小（步长100,000，下限1,000,000）"""
        if client in self.rebuffer_config:
            if buffer_type in self.rebuffer_config[client]:
                current = self.rebuffer_config[client][buffer_type]
                new_value = max(current - 100000, 1000000)
                self.rebuffer_config[client][buffer_type] = new_value
                print(f"已减少 {client} 的 {buffer_type}：{current} -> {new_value}")
                update("rebuffer_config", self.rebuffer_config)
            else:
                print(f"错误：{buffer_type} 不存在（应为 re_buffer/play_buffer）")
        else:
            print(f"错误：客户端 {client} 不存在")

    # --- 新增核心方法 ---
    def get_current_state(self):
        """将系统状态编码为特征向量：先采样5次取平均，再编码"""
        self.rebuff_event = 0
        self.client_stats = streamingMonitorClient.fetch_client_stats()
        for client in ['client1', 'client2', 'client3']:
            self.rebuff_event += (
                    self.client_stats[client]['rebuffer_time'] + self.client_stats[client]['rebuffer_count'])

        # 存5次原始数据
        traffic_classes_mark_list = []
        quality_map_list = []
        client_stats_list = []
        rebuffer_config_list = []
        track_stats_list = []
        bitrate_stats_list = []
        summary_rate_stats_list = []

        for _ in range(5):
            traffic_classes_mark_list.append(streamingMonitorClient.fetch_traffic_classes_mark())
            quality_map_list.append(streamingMonitorClient.fetch_quality_map())
            client_stats_list.append(streamingMonitorClient.fetch_client_stats())
            rebuffer_config_list.append(streamingMonitorClient.fetch_rebuffer_config())
            track_stats_list.append(streamingMonitorClient.fetch_track_stats())
            bitrate_stats_list.append(streamingMonitorClient.fetch_bitrate_stats())
            summary_rate_stats_list.append(streamingMonitorClient.fetch_summary_rate_stats())

            time.sleep(1)  # 每秒采一次

        # 对每个指标取平均
        # 为了简单，我们只处理需要的字段，逐项平均
        avg_traffic_classes_mark = {}
        avg_quality_map = {}
        avg_client_stats = {}
        avg_rebuffer_config = {}
        avg_track_stats = {}
        avg_bitrate_stats = {}
        avg_summary_rate_stats = {}

        # traffic_classes_mark 平均
        for ip in ['10.0.0.2', '10.0.0.3', '10.0.0.4']:
            avg_traffic_classes_mark[ip] = {}
            for bw in ['12600', '3150', '785', '200']:
                avg_traffic_classes_mark[ip][bw] = np.mean(
                    [sample[ip][bw] for sample in traffic_classes_mark_list]
                )

        # quality_map 平均
        for client in ['client1', 'client2', 'client3']:
            avg_quality_map[client] = {}
            for level in [0, 1, 2, 3]:
                avg_quality_map[client][level] = np.mean(
                    [sample[client][level] for sample in quality_map_list]
                )

        # client_stats 平均
        for client in ['client1', 'client2', 'client3']:
            avg_client_stats[client] = {
                'rebuffer_time': np.mean([sample[client]['rebuffer_time'] for sample in client_stats_list]),
                'rebuffer_count': np.mean([sample[client]['rebuffer_count'] for sample in client_stats_list])
            }

        # rebuffer_config 平均
        for client in ['client1', 'client2', 'client3']:
            avg_rebuffer_config[client] = {
                're_buffer': np.mean([sample[client]['re_buffer'] for sample in rebuffer_config_list]),
                'play_buffer': np.mean([sample[client]['play_buffer'] for sample in rebuffer_config_list])
            }

        # track_stats 平均
        for track_id in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
            avg_track_stats[track_id] = {}
            for client in ['client1', 'client2', 'client3']:
                avg_track_stats[track_id][client] = {
                    'latest_delay': np.mean([sample[track_id][client]['latest_delay'] for sample in track_stats_list]),
                    'latest_rate': np.mean([sample[track_id][client]['latest_rate'] for sample in track_stats_list]),
                }

        # bitrate_stats 平均
        for bitrate in ['12600', '3150', '785', '200']:
            avg_bitrate_stats[bitrate] = {}
            for client in ['client1', 'client2', 'client3']:
                avg_bitrate_stats[bitrate][client] = {
                    'latest_delay': np.mean([sample[bitrate][client]['latest_delay'] for sample in bitrate_stats_list]),
                    'latest_rate': np.mean([sample[bitrate][client]['latest_rate'] for sample in bitrate_stats_list]),
                }

        # summary_rate_stats 平均
        for client in ['client1', 'client2', 'client3']:
            avg_summary_rate_stats[client] = {
                'size': np.mean([sample[client]['size'] for sample in summary_rate_stats_list]),
                'time': np.mean([sample[client]['time'] for sample in summary_rate_stats_list]),
            }

        # 用平均值构造 state
        state = []

        for ip in ['10.0.0.2', '10.0.0.3', '10.0.0.4']:
            for bw in ['12600', '3150', '785', '200']:
                state.append(avg_traffic_classes_mark[ip][bw] / 30)

        for client in ['client1', 'client2', 'client3']:
            for level in [0, 1, 2, 3]:
                state.append(avg_quality_map[client][level] / 3)

        for client in ['client1', 'client2', 'client3']:
            state.append(avg_client_stats[client]['rebuffer_time'] / 100)
            state.append(avg_client_stats[client]['rebuffer_count'] / 100)

        for client in ['client1', 'client2', 'client3']:
            state.append(avg_rebuffer_config[client]['re_buffer'] / 3000000)
            state.append(avg_rebuffer_config[client]['play_buffer'] / 3000000)

        for client in ['client1', 'client2', 'client3']:
            delay = 0
            rate = 0
            for track_id in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
                delay += avg_track_stats[track_id][client]['latest_delay']
                rate += avg_track_stats[track_id][client]['latest_rate']
            delay = delay / 10 / 1000
            rate = rate / 10 / 10
            state.append(delay)
            state.append(rate)

        for client in ['client1', 'client2', 'client3']:
            for bitrate in ['12600', '3150', '785', '200']:
                state.append(avg_bitrate_stats[bitrate][client]['latest_delay'] / 1000)
                state.append(avg_bitrate_stats[bitrate][client]['latest_rate'] / 10)

        for client in ['client1', 'client2', 'client3']:
            state.append(avg_summary_rate_stats[client]['size'] / 10)
            state.append(avg_summary_rate_stats[client]['time'] / 1000)

        # 更新 rebuff_event
        temp = 0
        self.client_stats = streamingMonitorClient.fetch_client_stats()
        for client in ['client1', 'client2', 'client3']:
            temp += (
                    self.client_stats[client]['rebuffer_time'] + self.client_stats[client]['rebuffer_count'])
        self.rebuff_event = temp - self.rebuff_event

        return np.array(state)

    def calculate_reward(self, prev_state):
        """计算即时奖励"""
        # 基于以下因素计算奖励：
        # 1. 带宽利用率（越高越好）
        # 2. 视频卡顿次数（越少越好）
        # 3. 平均视频质量（越高越好）
        # 4. 带宽公平性
        # 5. Qoe总值
        # 6. Qoe公平性

        bandwidth_util = sum(
            self.summary_rate_stats[client]['size']
            for client in ['client1', 'client2', 'client3']
        ) / 10  # 最大可能值

        qoe=sum(
            self.client_stats[client]['qoe']
            for client in ['client1', 'client2', 'client3']
        ) / (50 * 3)

        fairness_band=fairness([self.summary_rate_stats[client]['size']
            for client in self.summary_rate_stats])

        fairness_qoe=fairness([self.client_stats[client]['qoe']
            for client in self.summary_rate_stats])

        avg_quality = sum(
            self.quality_map[client][level]
            for client in self.quality_map
            for level in [0, 1, 2, 3]
        ) / (3 * 12)  # 归一化

        # 假设通过监控获取缓冲次数（这里使用随机模拟）
        rebuffer_events =self.rebuff_event/200

        reward = (
                0.3 * log_normalize(bandwidth_util) +
                0.3 * fairness_band+
                0.3 * fairness_qoe+
                0.3 * avg_quality +
                0.5 * qoe-
                0.5 * rebuffer_events
        )
        return reward

    def decode_action(self, action_id):
        """将动作ID正确解码为可执行参数"""
        # 计算各动作类型的区间
        traffic_mark_actions = 3 * 4 * 2  # 3 IPs, 4 bit classes, 2 ops → 24
        quality_map_actions = 3 * 4 * 2  # 3 clients, 4 levels, 2 ops → 24
        buffer_config_actions = 3 * 2 * 2  # 3 clients, 2 buffer types, 2 ops →12

        # 确定动作类型
        if action_id < traffic_mark_actions:
            # traffic_mark: [0, 23]
            A = 0
            remainder = action_id
        elif action_id < traffic_mark_actions + quality_map_actions:
            # quality_map: [24, 47]
            A = 1
            remainder = action_id - traffic_mark_actions
        else:
            # buffer_config: [48, 59]
            A = 2
            remainder = action_id - (traffic_mark_actions + quality_map_actions)

        # 解析参数
        if A == 0:
            # traffic_mark: IP (3) × bit_class (4) × operation (2)
            B = remainder // (4 * 2)
            remainder %= (4 * 2)
            C = remainder // 2
            D = remainder % 2
            params = (
                "traffic_mark",
                ['10.0.0.2', '10.0.0.3', '10.0.0.4'][B],
                ['12600', '3150', '785', '200'][C],
                "increase" if D == 0 else "decrease"
            )
        elif A == 1:
            # quality_map: client (3) × level (4) × operation (2)
            B = remainder // (4 * 2)
            remainder %= (4 * 2)
            C = remainder // 2
            D = remainder % 2
            params = (
                "quality_map",
                ['client1', 'client2', 'client3'][B],
                C,  # level 0-3
                "increase" if D == 0 else "decrease"
            )
        else:
            # buffer_config: client (3) × buffer_type (2) × operation (2)
            B = remainder // (2 * 2)
            remainder %= (2 * 2)
            C = remainder // 2
            D = remainder % 2
            params = (
                "buffer_config",
                ['client1', 'client2', 'client3'][B],
                ['re_buffer', 'play_buffer'][C],
                "increase" if D == 0 else "decrease"
            )
        return params

    def execute_action(self, action_id):
        """执行解码后的动作"""
        try:
            action_type, target, param, operation = self.decode_action(action_id)

            if action_type == "traffic_mark":
                if operation == "increase":
                    self.increase_mark(target, param)
                else:
                    self.decrease_mark(target, param)

            elif action_type == "quality_map":
                if operation == "increase":
                    self.increase_quality(target, param)
                else:
                    self.decrease_quality(target, param)

            elif action_type == "buffer_config":
                if operation == "increase":
                    self.increase_buffer(target, param)
                else:
                    self.decrease_buffer(target, param)

            return True
        except Exception as e:
            print(f"Action execution failed: {e}")
            return False

    def store_experience(self, state, action, reward, next_state, done):
        """存储经验到回放池"""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample_experience(self):
        """从回放池中随机采样经验"""
        return random.sample(self.replay_buffer, min(len(self.replay_buffer), self.batch_size))

    def update_q(self, states, actions, rewards, next_states, dones):
        """批量更新Q值（修改后的版本）"""
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.BoolTensor(dones).to(device)

        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones.float()) * self.discount_factor * next_q_values

        # 计算损失
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)  # 梯度裁剪
        self.optimizer.step()

    def choose_action(self, state):
        """改进的ε-greedy策略"""
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, self.action_size - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def decay_exploration(self):
        """指数衰减探索率"""
        self.exploration_rate = max(self.min_exploration_rate,
                                    self.exploration_rate * self.exploration_decay)

    def train(self, episodes=1000):
        """完整的训练流程"""
        for episode in range(episodes):
            if state is None:
                state = self.get_current_state()
            else:
                state=prev_state
            action_id = self.choose_action(state)
            prev_state = state.copy()

            # 执行动作并获取奖励
            success = self.execute_action(action_id)
            reward = self.calculate_reward(prev_state) if success else -10

            # 获取新状态
            next_state = self.get_current_state()
            done = False  # 假设连续任务

            # 存储经验
            self.store_experience(prev_state, action_id, reward, next_state, done)
            prev_state=next_state

            # 经验回放学习
            if len(self.replay_buffer) >= self.batch_size:
                batch = self.sample_experience()
                for exp in batch:
                    self.update_q(*exp)

            # 更新探索率
            self.decay_exploration()

            # 定期同步目标网络
            if episode % 100 == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            # 输出训练进度
            if episode % 50 == 0:
                print(f"Episode {episode}: Epsilon {self.exploration_rate:.2f} | Recent Reward {reward:.2f}")


if __name__ == "__main__":
    agent = StreamingOptimizer()
    agent.train(episodes=1000)