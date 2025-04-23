import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import threading
import requests
import time


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class SimpleRL:
    def __init__(self, track_stats, link_metrics, client_stats, input_size, output_size):
        self.track_stats = track_stats
        self.link_metrics = link_metrics
        self.client_stats = client_stats
        self.actions = [
            "modify_client_bit_rate",
            "modify_client_buffer_size",
            "modify_global_bandwidth_weight",
        ]
        self.input_size = input_size
        self.output_size = output_size

        # 初始化 Q 网络和目标 Q 网络
        self.q_network = QNetwork(input_size, output_size)
        self.target_network = QNetwork(input_size, output_size)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

        # 强化学习的参数
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.01

        # 经验回放池
        self.replay_buffer = []
        self.buffer_size = 10000
        self.batch_size = 32

        # 启动数据更新线程
        self.update_thread = threading.Thread(target=self.update_data, daemon=True)
        self.update_thread.start()

    def get_current_state(self):
        self.update_data()
        return self.parse_data()

    def parse_data(self):
        pass
        return (self.client_stats,self.track_stats,self.link_metrics)


    def update_data(self):
        """ 每秒钟从监控系统获取并更新数据 """
        while True:
            try:
                track_stats_response = requests.get('http://localhost:5000/get_track_stats')  # 假设接口
                link_metrics_response = requests.get('http://localhost:5000/get_link_metrics')  # 假设接口
                client_stats_response = requests.get('http://localhost:5000/get_client_stats')
                if client_stats_response.status_code == 200:
                    self.client_stats = client_stats_response.json()

                if track_stats_response.status_code == 200:
                    self.track_stats = track_stats_response.json()  # 更新 track_stats

                if link_metrics_response.status_code == 200:
                    self.link_metrics = link_metrics_response.json()  # 更新 link_metrics

            except Exception as e:
                print(f"Error updating data: {e}")

            time.sleep(1)  # 每秒更新一次

    def store_experience(self, state, action, reward, next_state, done):
        """ 存储经验到回放池 """
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample_experience(self):
        """ 从回放池中随机采样经验 """
        return random.sample(self.replay_buffer, self.batch_size)

    def update_q(self, state, action, reward, next_state, done):
        """ 更新 Q 值 """
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)

        q_values = self.q_network(state)
        next_q_values = self.target_network(next_state)

        q_value = q_values[action]

        if done:
            target = reward
        else:
            target = reward + self.discount_factor * torch.max(next_q_values)

        loss = self.loss_fn(q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def choose_action(self, state):
        """ 根据 ε-greedy 策略选择动作 """
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(range(self.output_size))  # 探索
        else:
            state = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                q_values = self.q_network(state)
            return torch.argmax(q_values).item()  # 利用

    def decay_exploration(self):
        """ 衰减探索率 """
        if self.exploration_rate > self.min_exploration_rate:
            self.exploration_rate *= self.exploration_decay

    def execute_action(self, action, **kwargs):
        """ 执行动作 """
        # 动作执行与之前的代码类似
        if action == 0:  # 修改客户端比特率
            pass

        elif action == 2:  # 修改客户端缓冲区大小
            pass

        elif action == 3:  # 修改视频块的带宽分配权重
            pass



    def train(self, episodes):
        """ 训练过程 """
        for _ in range(episodes):
            state = self.get_current_state()
            action = self.choose_action(state)
            reward = self.calculate_reward(state)

            # 执行动作
            action_params = {
                pass
            }
            self.execute_action(action, **action_params)

            # 存储经验
            next_state = self.get_current_state()  # 假设下一个状态是当前状态
            done = False  # 假设没有结束条件
            self.store_experience(state, action, reward, next_state, done)

            # 从经验回放池中随机采样并更新 Q 网络
            if len(self.replay_buffer) > self.batch_size:
                experiences = self.sample_experience()
                for exp in experiences:
                    state, action, reward, next_state, done = exp
                    self.update_q(state, action, reward, next_state, done)

            self.decay_exploration()

            # 每隔一定步数更新目标网络
            if _ % 100 == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

    def calculate_reward(self, state):
        """ 假设奖励计算方法：延迟越低，奖励越高 """
        return pass


# 初始化 track_stats 和 link_metrics（从监控系统中获取的初始数据）
track_stats = defaultdict(lambda: defaultdict(lambda: {
    'avg_delay': 0.0,
    'avg_rate': 0.0,
    'latest_delay': 0.0,
    'latest_rate': 0.0,
    'resolution': '',
    'last_update': None
}))

link_metrics = defaultdict(lambda: {
    'delay': 0.0,
    'loss_rate': 0.0,
    'marks': {},
    'last_update': None
})

client_stats = defaultdict(lambda: {
    'rebuffer_time': 0.0,
    'rebuffer_count': 0.0,
    'qoe':0.0,
    'last_update': None
})

# 初始化输入输出维度
input_size = 0  # 假设状态空间大小
output_size = len(["modify_client_bit_rate", "modify_client_buffer_size",
                   "modify_global_bandwidth_weight"])

# 实例化 RL 模型并进行训练
rl_model = SimpleRL(track_stats, link_metrics, input_size, output_size)
rl_model.train(1000)  # 训练 1000 次
