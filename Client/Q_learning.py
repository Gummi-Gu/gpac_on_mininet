import queue
import time

import requests
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from util import StreamingMonitorClient

# ====================== 设备配置 ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== 参数 ======================
streamingMonitorClient=StreamingMonitorClient()
client_name='client1'
client_ip='10.0.0.1'
total_bandwidth=16
alpha = 0.3
beta = 0.3
lambda_ = 0.2
theta = 0.2
gamma = 0.1

# ====================== 经验回放缓冲区 ======================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            torch.FloatTensor(state).to(device),
            torch.tensor([action], dtype=torch.long).to(device),
            torch.FloatTensor([reward]).to(device),
            torch.FloatTensor(next_state).to(device),
            torch.FloatTensor([done]).to(device)
        ))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None

        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return (
            torch.stack(states),
            torch.cat(actions),
            torch.cat(rewards),
            torch.stack(next_states),
            torch.cat(dones)
        )


# ====================== DQN 模型定义 ======================
class ClientDQN(nn.Module):
    def __init__(self, input_dim=3, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# ====================== 客户端智能体 ======================
class DQNClient:
    def __init__(self, client_id, gamma=0.99):
        self.client_id = client_id  # 质量等级映射表

        # 模型参数
        self.input_dim = 4  # [预测带宽, 缓冲区, 实际带宽, 实际时延]
        self.output_dim = 4  # 可选分辨率等级数量（0-3）

        # 初始化模型
        self.policy_net = ClientDQN(self.input_dim, self.output_dim).to(device)
        self.target_net = ClientDQN(self.input_dim, self.output_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)

        # 训练参数
        self.gamma = gamma
        self.epsilon = 0.9
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32

        # 经验回放
        self.memory = ReplayBuffer(10000)

        # 本地环境状态
        self.current_state = None
        self.current_quality = {}

    def get_state(self, env_data):
        """从环境数据中提取本地状态并进行归一化"""
        # env_data 应包含：预测带宽，缓冲区状态，实际带宽
        predicted_bw = env_data['predicted_bandwidth']
        buffer_level = env_data['buffer_level']
        actual_band = env_data['actual_bandwidth']
        actual_delay = env_data['actual_delay']
        predicted_bw_sum=0
        for i,j in predicted_bw.items():
            predicted_bw_sum+=j
        # 状态归一化
        state = np.array([
            predicted_bw_sum,  # 假设最大带宽20000kbps
            buffer_level / 10.0,  # 假设缓冲区最大10秒
            actual_band / 2,
            actual_delay / 1000
        ], dtype=np.float32)

        return torch.FloatTensor(state).to(device)

    def select_action(self, state):
        """ε-贪婪策略选择动作"""
        if np.random.random() < self.epsilon:
            actions = {
                'bw_factor': round(np.random.uniform(0.2, 0.9), 2),  # 保留两位小数可选
                'quality_up': list(np.random.randint(0, 4, 4)),
                'buffer': np.random.randint(1,6)
            }
            #return np.random.randint(self.output_dim)
            return actions
        else:
            with torch.no_grad():
                q_values = self.policy_net(state)
                return q_values.argmax().item()

    def update_model(self):
        """执行一步DQN训练"""
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return

        states, actions, rewards, next_states, dones = batch

        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # 计算当前Q值
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # 计算损失
        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    def sync_target_network(self):
        """同步目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def receive_global_model(self, global_state_dict):
        """接收来自服务器的全局模型"""
        self.policy_net.load_state_dict(global_state_dict)
        self.sync_target_network()

    def get_local_parameters(self):
        """返回当前模型参数（用于联邦聚合）"""
        return self.policy_net.state_dict()


# ====================== 环境模拟接口 ======================
class LocalEnvSimulator:
    """简化的本地环境模拟器"""

    def __init__(self):
        # 初始状态
        self.qoe = 0
        self.predicted_bandwidth = {}  # 预测带宽
        self.buffer_level = 1.5  # 缓冲区级别（秒）
        self.actual_bandwidth = 10  # 实际可用带宽
        self.actual_delay=0
        self.buffer_queue=queue.Queue(maxsize=10)
        self.qoe_queue=queue.Queue(maxsize=10)

    def update(self, endpoint, data):

        url = f"127.0.0.1:5000/update/{endpoint}"
        try:
            response = requests.post(url, json=data)
        except Exception as e:
            print(f"请求失败: {e}")

    def step(self, action):
        """执行动作并返回新状态和奖励"""
        # 应用动作
        total_bw=action['bw_factor']
        traffic_classes_mark_update = {
            client_ip: {'port': 10086,
                        '12600': total_bw*self.predicted_bandwidth['12600'],
                        '3150': total_bw*self.predicted_bandwidth['3150'],
                        '785': total_bw*self.predicted_bandwidth['785'],
                        '200': total_bw*self.predicted_bandwidth['200']},
        }
        self.update("traffic_classes_mark",traffic_classes_mark_update)

        quality_map=action['quality_up']
        quality_map_update = {
            client_name: {0:quality_map[0],1:quality_map[1],2:quality_map[2],3:quality_map[3]}
        }
        self.update("quality_map", quality_map_update)

        rebuffer_config_update = {
            client_name: {'re_buffer': action['buffer'], 'play_buffer': action['buffer']+1},
        }
        self.update("rebuffer_config", rebuffer_config_update)

        state=self.get_state()

        # 这里简化奖励计算，实际应根据QoE公式计算
        reward = self.calculate_reward()

        '''
        # 更新环境状态（示例逻辑）
        self.predicted_bandwidth = self.predicted_bandwidth * 0.9 + np.random.normal(0, 500)
        self.buffer_level = max(0, self.buffer_level + (np.random.rand() - 0.5))
        self.actual_bandwidth = self.actual_bandwidth * 0.95 + np.random.normal(0, 300)
        '''

        return state, reward, False  # 假设不会终止

    def calculate_reward(self):
        # === 1. 缓冲区稳定性 ===
        self.buffer_queue.put(self.buffer_level)
        buffer_list = list(self.buffer_queue.queue)
        P_buffer = np.std(buffer_list) if len(buffer_list) >= 2 else 0

        # === 2. 当前帧 QoE 获取 ===
        temp = streamingMonitorClient.fetch_client_states()
        qoe_list = [buffer['qoe'] for buffer in temp.values()]
        N = len(qoe_list)
        S_qoe = self.qoe

        # === 3. QoE 稳定性（跨 step）===
        self.qoe_queue.put(S_qoe)
        qoe_queue_list = list(self.qoe_queue.queue)
        P_qoe = np.std(qoe_queue_list) if len(qoe_queue_list) >= 2 else 0

        # === 4. QoE 公平性 ===
        fairness_qoe = (self.qoe ** 2) / (N * sum(q ** 2 for q in qoe_list) + 1e-6) if N > 0 else 0

        # === 5. 带宽利用率 ===
        total_used_bw = self.actual_bandwidth*8 # 使用总带宽
        bandwidth_efficiency = total_used_bw / total_bandwidth

        # === 7. 奖励函数组合 ===
        reward = (
                S_qoe
                + alpha * fairness_qoe
                + beta * bandwidth_efficiency
                - lambda_ * P_qoe
                - theta * P_buffer
        )

        return reward

    def get_state(self):
        avg_num={'12600':0,'3150':0,'785':0,'200':0}
        avg_size={'12600':0,'3150':0,'785':0,'200':0}
        avg_band=0
        avg_delay=0
        avg_buffer={'rebuffer':0,'play':0}
        avg_qoe=0
        for _ in range(5):
            time.sleep(1)
            temp=streamingMonitorClient.fetch_track_stats()
            for track,states in temp.items():
                if track == 'default':
                    continue
                for client_id , state in states.items():
                    if client_id != client_name :
                        continue
                    avg_num[str(state['resolution'])]+=1
            temp=streamingMonitorClient.fetch_bitrate_stats()
            for bitrate, clients in temp.items():
                if bitrate == 'default':
                    continue
                for client_id, stats in clients.items():
                    if client_id != client_name:
                        continue
                    avg_size[bitrate]+=stats['avg_size']
            temp=streamingMonitorClient.fetch_summary_rate_stats()
            for client_id, stats in temp.items():
                if client_id != client_name:
                    continue
                avg_band+=stats['size']
                avg_delay+=stats['time']
            temp=streamingMonitorClient.fetch_client_states()
            for client_id, buffer in temp.items():
                if client_id != client_name:
                    continue
                avg_qoe+=buffer['qoe']
            temp=streamingMonitorClient.fetch_rebuffer_config()
            for client_id, buffer in temp.items():
                if client_id != client_name:
                    continue
                avg_buffer['rebuffer']+=buffer['re_buffer']
                avg_buffer['play']+=buffer['play_buffer']

        for key in avg_num:
            avg_num[key] /= 5
        for key in avg_size:
            avg_size[key] /= 5
        for key in avg_buffer:
            avg_buffer[key] /= 5
        avg_band /= 5
        avg_delay /= 5
        avg_qoe /= 5

        sum = {'12600': 0, '3150': 0, '785': 0, '200': 0}
        for i, j in avg_num.items():
            sum[i] = j * avg_size[i]
        total=0
        for i in sum.values():
            total += i
        for k in sum:
            sum[k] = sum[k] / total if total != 0 else 0  # 防止除以0

        self.predicted_bandwidth = sum

        self.buffer_level=(avg_buffer['rebuffer']+avg_buffer['play'])/2

        self.actual_bandwidth=avg_band

        self.actual_delay=avg_delay

        self.qoe=avg_qoe

        return {
            'predicted_bandwidth': self.predicted_bandwidth,
            'buffer_level': self.buffer_level,
            'actual_bandwidth': self.actual_bandwidth,
            'actual_delay': self.actual_delay,
            'qoe': self.qoe
        }


# ====================== 客户端训练循环 ======================
def client_training_loop(client_id, num_episodes=1000):
    # 初始化
    client = DQNClient(client_id)
    env = LocalEnvSimulator()

    # 训练循环
    for episode in range(num_episodes):
        state = env.get_state()
        total_reward = 0

        # 模拟单次训练（实际应替换为与真实环境交互）
        for _ in range(1):  # 假设每个episode包含10个决策步骤
            # 获取状态并选择动作
            state_tensor = client.get_state(state)
            action = client.select_action(state_tensor)

            # 执行动作并获取新状态
            new_state, reward, done = env.step(action)
            total_reward += reward

            # 存储经验
            client.memory.push(
                state_tensor.cpu().numpy(),
                action,
                reward,
                client.get_state(new_state).cpu().numpy(),
                done
            )

            # 更新状态
            state = new_state

            # 训练模型
            if len(client.memory.buffer) > client.batch_size:
                loss = client.update_model()

        # 定期同步目标网络
        if episode % 10 == 0:
            client.sync_target_network()

        # 输出训练信息
        print(f"Client {client_id} Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {client.epsilon:.2f}")


# ====================== 联邦接口 ======================
class FederatedServer:
    def __init__(self, num_clients):
        self.global_model = ClientDQN()
        self.client_models = [ClientDQN() for _ in range(num_clients)]

    def aggregate(self):
        """联邦平均聚合"""
        global_dict = self.global_model.state_dict()
        for key in global_dict:
            global_dict[key] = torch.mean(
                torch.stack([client.state_dict()[key] for client in self.client_models]),
                dim=0
            )
        self.global_model.load_state_dict(global_dict)

    def distribute_model(self):
        """分发全局模型到各客户端"""
        for client in self.client_models:
            client.load_state_dict(self.global_model.state_dict())


# ====================== 使用示例 ======================
if __name__ == "__main__":
    # 单个客户端本地训练
    client_training_loop(client_id=client_name, num_episodes=100)