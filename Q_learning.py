import logging
import os
import queue
import subprocess
import sys
import time
from datetime import datetime
from math import sqrt, exp

import requests
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from util import StreamingMonitorClient
from filterpy.kalman import KalmanFilter

# ====================== 设备配置 ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== 参数 ======================
streamingMonitorClient=StreamingMonitorClient()
client_name='client1'
client_ip='10.0.0.2'
total_bandwidth=20
alpha = 5
beta = 2
lambda_ = 1
theta = 0.2
gamma = 0.1
save_dir = "./saved_models"
os.makedirs(save_dir, exist_ok=True)

# ====================== 日志 =======================
def set_log():
    logging.basicConfig(
        filename=f'log_{client_name}.txt',
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def log_metrics(S_qoe, delay, fairness_qoe, fairness_band, bandwidth_efficiency,
                P_qoe, P_buffer, bw):
    # delay 除以 100
    delay_adjusted = (delay*1000)//100*100

    log_message = (
        f"S_qoe={S_qoe}, delay={delay_adjusted:.4f}, fairness_qoe={fairness_qoe}, "
        f"fairness_band={fairness_band}, bandwidth_efficiency={bandwidth_efficiency}, "
        f"P_qoe={P_qoe}, P_buffer={P_buffer}, bw={bw}"
    )
    logging.info(log_message)
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
    def __init__(self, input_dim=5, output_dim=54):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# ====================== 缩减后的动作空间设计 ======================
class ReducedActionSpace:
    def __init__(self):
        # 离散化带宽因子（3种选择）
        self.bw_options = {
            0: 0.5,# 保守带宽利用
            1: 0.65,
            2: 0.85,
            3: 1, # 平衡模式
            4: 1.3,
            5: 1.7,
            6: 2  # 激进模式
        }

        # 质量调整策略（3种全局策略）
        self.quality_strategies = {
            0: [0, 0, 1, 3],  # 性价比模式
            1: [0, 1, 2, 3],  # 正常模式
            2: [0, 3, 3, 3]   # 全高模式
        }

        # 缓冲区配置（2种预设）
        self.buffer_presets = {
            0: 1, # 低延迟模式
            1: 3   # 流畅模式
        }

    def get_action(self, action_idx):
        """将离散动作编号解码为具体参数组合"""
        bw_idx = action_idx // (len(self.quality_strategies) * len(self.buffer_presets))
        remaining = action_idx % (len(self.quality_strategies) * len(self.buffer_presets))

        quality_idx = remaining // len(self.buffer_presets)
        buffer_idx = remaining % len(self.buffer_presets)

        if bw_idx > 3:
            quality_idx=max(1,quality_idx)
        elif bw_idx < 3:
            quality_idx=min(1,quality_idx)

        return {
            'bw_factor': self.bw_options[bw_idx],
            'quality_up': self.quality_strategies[quality_idx],
            'buffer': self.buffer_presets[buffer_idx]
        }

    @property
    def action_space_size(self):
        return len(self.bw_options) * len(self.quality_strategies) * len(self.buffer_presets)

# ====================== 客户端智能体 ======================
class DQNClient:
    def __init__(self, client_id, gamma, ep, init_model_path=None):
        self.client_id = client_id  # 质量等级映射表

        # 模型参数
        self.action_space = ReducedActionSpace()
        self.output_dim = self.action_space.action_space_size  # 18
        self.input_dim = 5  # [预测带宽, 缓冲区, 实际带宽, 实际时延， Qoe]

        # 初始化模型
        self.policy_net = ClientDQN(self.input_dim, self.output_dim).to(device)
        self.target_net = ClientDQN(self.input_dim, self.output_dim).to(device)

        # 新增模型加载逻辑
        if init_model_path is not None:
            if os.path.exists(init_model_path):
                print(f"Loading initial model from {init_model_path}")
                self.policy_net.load_state_dict(torch.load(init_model_path, map_location=device))
            else:
                print(f"Warning: Model file {init_model_path} not found, using random initialization")

        self.target_net.load_state_dict(self.policy_net.state_dict())

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)

        # 训练参数
        self.gamma = gamma
        self.epsilon = ep
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
        qoe = env_data['qoe']
        predicted_bw_sum=0
        for i,j in predicted_bw.items():
            predicted_bw_sum+=j
        predicted_bw_sum/=1000
        # 状态归一化
        state = np.array([
            predicted_bw_sum/2,
            buffer_level / 3.5,
            actual_band / 2,
            actual_delay / 1000,
            sqrt(max(0.0,qoe/8))
        ], dtype=np.float32)

        print("\n========== state ==========\n")
        print("Predicted Bandwidth:", predicted_bw_sum/2)
        print("Buffer Level:", buffer_level / 3.5)
        print("Actual Bandwidth:", actual_band / 2)
        print("Actual Delay:", actual_delay / 1000)
        print("QoE:", sqrt(max(0.0,qoe/8)))

        return torch.FloatTensor(state).to(device)

    def select_action(self, state):
        """ε-贪婪策略选择动作"""
        if np.random.random() < self.epsilon:
            if state[3]<0.4:
                return np.random.randint(int(self.output_dim*4/7))
            elif state[3] > 0.6:
                return np.int32(self.output_dim*3/7)+np.random.randint(int(self.output_dim*4/7))
            else:
                return np.random.randint(self.output_dim)
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
        if self.epsilon<0.5:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        else:
            self.epsilon -= 0.005
        

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

    def request_global_model(self):
        data = {"model_weights": self.get_local_parameters()}
        response = requests.post(f"http://localhost:5000/upload_client_model/{client_ip}", json=data)
        response = requests.get("http://localhost:5000/get_global_model")
        weights = {k: torch.tensor(v) for k, v in response.json()["model_weights"].items()}
        self.receive_global_model(weights)
        pass


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
        self.action_space = ReducedActionSpace()
        self.avg_qoe_list=[]
        self.avg_band_list = []

    def update(self, endpoint, data):

        url = f"http://127.0.0.1:5000/update/{endpoint}"
        try:
            response = requests.post(url, json=data)
        except Exception as e:
            print(f"请求失败: {e}")

    def step(self, action):
        """执行动作并返回新状态和奖励"""
        action=self.action_space.get_action(action)
        # 应用动作
        total_bw=action['bw_factor']
        quality_map=action['quality_up']
        bitrate_mapping = {
            0: '200',
            1: '785',
            2: '3150',
            3: '12600'
        }
        temp={}
        # 遍历所有 quality 等级

        sum_band=self.predicted_bandwidth.copy()
        total=0
        for i in sum_band.values():
            total += i
        for k in sum_band:
            sum_band[k] = sum_band[k] / total if total != 0 else 0  # 防止除以0

        for quality_value in range(4):
            # 找出所有 value 为当前 quality 的 key
            keys = [k for k, v in enumerate(quality_map) if v == quality_value]

            # 映射成比特率（字符串）
            mapped_keys = [bitrate_mapping[k] for k in keys]

            # 累加 self.predicted_bandwidth 中这些比特率对应的值
            total = sum(sum_band.get(k, 0) for k in mapped_keys)

            temp[bitrate_mapping[quality_value]] = total
        sum_band=temp

        traffic_classes_mark_update = {
            client_ip: {'port': 10086,
                        '12600': total_bw*sum_band['12600'],
                        '3150': total_bw*sum_band['3150'],
                        '785': total_bw*sum_band['785'],
                        '200': total_bw*sum_band['200']},
        }
        self.update("traffic_classes_mark",traffic_classes_mark_update)
        quality_map_update = {
            client_name: {0:quality_map[0],1:quality_map[1],2:quality_map[2],3:quality_map[3]}
        }
        self.update("quality_map", quality_map_update)
        rebuffer_config_update = {
            client_name: {'re_buffer': action['buffer'], 'play_buffer': action['buffer']+1},
        }
        self.update("rebuffer_config", rebuffer_config_update)

        #等待生效
        print("########## 等待生效 ##########")
        time.sleep(8)


        state=self.get_state()
        # 这里简化奖励计算，实际应根据QoE公式计算
        reward = self.calculate_reward(action['bw_factor'])
        '''
        # 更新环境状态（示例逻辑）
        self.predicted_bandwidth = self.predicted_bandwidth * 0.9 + np.random.normal(0, 500)
        self.buffer_level = max(0, self.buffer_level + (np.random.rand() - 0.5))
        self.actual_bandwidth = self.actual_bandwidth * 0.95 + np.random.normal(0, 300)
        '''

        return state, reward, False  # 假设不会终止

    def calculate_reward(self,bw):
        # === 1. 缓冲区稳定性 ===
        if self.buffer_queue.qsize() == 10:
            self.buffer_queue.get()
        self.buffer_queue.put(self.buffer_level)
        buffer_list = list(self.buffer_queue.queue)
        P_buffer = np.std(buffer_list) if len(buffer_list)>2 else 0 # 标准化后计算标准差
        P_buffer = 1 / (1 + np.exp(-P_buffer))

        # === 2. 当前 QoE 获取 ===
        S_qoe = sqrt(max(0.0,self.qoe/8))

        # === 3. QoE 稳定性（跨 step）===
        if self.qoe_queue.qsize() == 10:
            self.qoe_queue.get()
        self.qoe_queue.put(S_qoe)
        qoe_queue_list = list(self.qoe_queue.queue)
        P_qoe = np.std(qoe_queue_list)  if len(qoe_queue_list)>2 else 0
        P_qoe = 1 / (1 + np.exp(-P_qoe))# 计算标准差

        # === 4. QoE 公平性 ===
        qoe_list=self.avg_qoe_list
        for _ in range(1):
            qoe_list.append(random.randint(2,8))
        N = len(qoe_list)
        qoe = [q if q > 0 else 0 for q in qoe_list]
        fairness_qoe = (sum(qoe) ** 2) / (len(qoe) * sum(q**2 for q in qoe) + 1e-6)


        # === 5. 带宽利用率 ===
        delay=0
        for d in self.avg_band_list:
            delay=max(delay,d[1])
        total_used_bw = sum(q[0] for q in self.avg_band_list)/delay*1000*8 # 使用总带宽
        bandwidth_efficiency = total_used_bw / total_bandwidth

        # === 6. 时延 ===
        delay=self.actual_delay / 1000

        # === 7. 带宽公平性 ===
        band_list=self.avg_band_list
        for _ in range(0):
            band_list.append(1)
        N = len(band_list)
        fairness_band = (sum(q[0] for q in band_list) ** 2) / (N * sum(q[0] ** 2 for q in band_list) + 1e-6) if N > 0 else 0

        # 公平性函数：低于0.8急剧下降（sigmoid）
        def fairness_score(x, k=40):
            return 1 / (1 + exp(-k * (x - 0.8)))

        # QoE 函数：基于 q 和 fq 的值计算
        def qoe_score(q, fq):
            # 在 QoE 计算中直接考虑公平性，fq < 0.8 会急剧影响 QoE
            fair_qoe = q * fairness_score(fq)
            return 1 / (1 + exp(-10 * (fair_qoe - 0.5)))

        # 平底型时延奖励函数：最优在 [0.4, 0.6]
        def delay_score(d, k=10):
            left = 1 / (1 + exp(-k * (d - 0.4)))
            right = 1 / (1 + exp(-k * (d - 0.6)))
            score = (left - right)*2.6
            return max(0.0, min(score, 1.0))  # 限定在 [0,1]

        def compute_reward(q, d, fq, fb, bu, sq, sb,
                           alpha1=5, alpha2=2, alpha3=1):
            Q = qoe_score(q, fq)  # 计算 QoE（已经包含公平性）
            D = delay_score(d)  # 计算时延奖励
            F_fair = fairness_score(fb)  # 计算带宽公平性
            BU = bu  # 计算带宽利用率
            total=alpha1+alpha2+alpha3
            reward = alpha1 * (Q + D) + alpha2 * (F_fair + BU) - alpha3 * (sq + sb)
            reward /= total

            print(f"\n========== reward ==========\n"
                  f"Q:{Q,q,fairness_score(fq),qoe_list},\n"
                  f"D:{delay,D},\n"
                  f"B_fair:{F_fair},\n"
                  f"BU:{bu,band_list},\n"
                  f"P_qoe:{sq},\n"
                  f"P_buffer:{sb},\n")
            return reward


        # === 8. 奖励函数组合 ===
        reward = compute_reward(S_qoe,delay,fairness_qoe,fairness_band,bandwidth_efficiency,P_qoe,P_buffer,alpha,beta,lambda_)

        log_metrics(S_qoe, delay, fairness_qoe, fairness_band,
                    bandwidth_efficiency, P_qoe, P_buffer, bw)

        return reward

    def get_state(self):
        avg_num={'12600':0,'3150':0,'785':0,'200':0}
        avg_size={'12600':[],'3150':[],'785':[],'200':[]}
        avg_band=0
        avg_delay=0
        avg_buffer={'rebuffer':0,'play':0}
        avg_qoe=0
        all_qoe_per_round = []
        avg_qoe_list = []
        all_band_per_round = []
        avg_band_list = []
        N=8
        print("########## 开始收集 ##########")
        for _ in range(N):
            time.sleep(1)
            temp=streamingMonitorClient.fetch_orign_quality_tiled()
            for i in temp[client_name]:
                if i == 0:
                    avg_num['200']+=1
                elif i == 1:
                    avg_num['785']+=1
                elif i == 2:
                    avg_num['3150']+=1
                elif i == 3:
                    avg_num['12600']+=1

            temp=streamingMonitorClient.fetch_bitrate_stats()
            for bitrate, clients in temp.items():
                if bitrate == 'default':
                    continue
                for client_id, stats in clients.items():
                    if client_id != client_name:
                        continue
                    avg_size[bitrate].append(stats['avg_size'])
            temp=streamingMonitorClient.fetch_summary_rate_stats()
            for client_id, stats in temp.items():
                if client_id != client_name:
                    continue
                avg_band+=stats['size']/stats['time']*1000
                avg_delay+=stats['time']
            band_list = []
            for client_id, stats in temp.items():
                band = (stats['size'],stats['time'])  # 获取客户端的 带宽情况
                if band is not None:
                    band_list.append(band)
            if band_list:
                all_band_per_round.append(band_list)
            temp=streamingMonitorClient.fetch_client_states()
            for client_id, buffer in temp.items():
                if client_id != client_name:
                    continue
                avg_qoe+=buffer['qoe']
            qoe_list = []
            for client_id, buffer in temp.items():
                qoe = buffer.get('qoe')  # 获取客户端的 QoE 值
                if qoe is not None:
                    qoe_list.append(qoe)
            if qoe_list:
                all_qoe_per_round.append(qoe_list)
            temp=streamingMonitorClient.fetch_rebuffer_config()
            for client_id, buffer in temp.items():
                if client_id != client_name:
                    continue
                avg_buffer['rebuffer']+=buffer['re_buffer']
                avg_buffer['play']+=buffer['play_buffer']

        for key in avg_num:
            avg_num[key] /= N

        def predict_next(data):
            kf = KalmanFilter(dim_x=2, dim_z=1)
            kf.F = np.array([[1, 1], [0, 1]])
            kf.H = np.array([[1, 0]])
            kf.x = np.array([[data[0]], [0.]])
            kf.P *= 1000.
            kf.Q = np.eye(2)
            kf.R = np.array([[5]])
            for z in data: kf.predict(); kf.update(z)
            kf.predict()
            return float(kf.x[0])
        avg_size = {k: predict_next(v) if len(v) >= N else 1 for k, v in avg_size.items()}

        for key in avg_buffer:
            avg_buffer[key] /= N
        avg_band /= N
        avg_delay /= N
        avg_qoe /= N

        for i,_ in enumerate(all_qoe_per_round[0]):
            _avg_qoe=0
            for qoe_list in all_qoe_per_round:
                _avg_qoe+=qoe_list[i]
            _avg_qoe/=N
            avg_qoe_list.append(_avg_qoe)
        for i,_ in enumerate(all_band_per_round[0]):
            _avg_band=0
            _avg_time=500
            for band_list in all_band_per_round:
                _avg_band+=band_list[i][0]
                _avg_time=max(band_list[i][1],_avg_band)
            _avg_band/=N
            avg_band_list.append((_avg_band,_avg_time))

        sum_band = {'12600': 0, '3150': 0, '785': 0, '200': 0}
        for i, j in avg_num.items():
            sum_band[i] = j * avg_size[i]

        self.predicted_bandwidth = sum_band

        self.buffer_level=(avg_buffer['rebuffer']+avg_buffer['play'])/2

        self.actual_bandwidth=avg_band

        self.actual_delay=avg_delay

        self.qoe=avg_qoe

        self.avg_qoe_list=avg_qoe_list

        self.avg_band_list=avg_band_list
            
        return {
            'predicted_bandwidth': self.predicted_bandwidth,
            'buffer_level': self.buffer_level,
            'actual_bandwidth': self.actual_bandwidth,
            'actual_delay': self.actual_delay,
            'qoe': self.qoe
        }


# ====================== 客户端训练循环 ======================

project_root = os.path.dirname(os.path.abspath(__file__))  # 项目根目录
venv_python = r"D:\Users\GummiGu\PycharmProjects\gpac_on_mininet\.venv\Scripts\python.exe"
modules = [
    #'Client.main2',
    #'Client.main1'
]
process_infos = []

def start_single_client(venv_python, module, project_root):
    print(f"Starting {module}...")
    p = subprocess.Popen(
        [venv_python, "-m", module],
        cwd=project_root
    )
    return {"module": module, "process": p, "start_time": time.time()}
def restart_client(process_info, venv_python, project_root):
    p = process_info["process"]
    module = process_info["module"]
    if p.poll() is None:
        print(f"Terminating process {p.pid} ({module})...")
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(f"Process {p.pid} did not terminate, killing...")
            p.kill()
    return start_single_client(venv_python, module, project_root)


def client_training_loop(client_id, num_episodes, ep,init_model,reason=True):
    # 初始化
    client = DQNClient(client_id, 0.5, ep, init_model)
    env = LocalEnvSimulator()
    if reason == True:
        for episode in range(num_episodes):
            time.sleep(5)
            state = env.get_state()
            for i in range(5):
                state_tensor = client.get_state(state)
                # 这里不更新 epsilon，也不存 replay，也不调用 update_model
                action_idx = client.select_action(state_tensor)
                #action_idx = 21 # ceshi
                action = client.action_space.get_action(action_idx)
                print(f"[Episode {ep}] 选择动作：{action}")
                new_state, reward, done = env.step(action_idx)
                state=new_state
                print(f"[Episode {ep}] reward={reward:.4f}\n")
            for i, info in enumerate(process_infos):
                p = info["process"]
                if p.poll() is not None:
                    print(f"Process {p.pid} ({info['module']}) exited early. Restarting...")
                    process_infos[i] = start_single_client(venv_python, info["module"], project_root)
                else:
                    print(f"Process {p.pid} ({info['module']}) reached 60s. Restarting...")
                    process_infos[i] = restart_client(info, venv_python, project_root)
    else:
        # 训练循环
        for episode in range(num_episodes):
            time.sleep(5)
            state = env.get_state()
            total_reward = 0

            # 模拟单次训练（实际应替换为与真实环境交互）
            for i in range(5):  # 假设每个episode包含10个决策步骤
                # 获取状态并选择动作
                print(f"\n==========参数==========\n"
                    f"episodes: {episode,i}\n"
                    f"epsilon:{client.epsilon}")
                state_tensor = client.get_state(state)
                action = client.select_action(state_tensor)
                print("\n==========动作==========\n"
                    f"action:{env.action_space.get_action(action)} \n")
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

                print(f"\n========== 结果 ==========\n"
                    f"reward: {reward}\n")

                print("\n==================== next ====================\n")
            #重启
            for i, info in enumerate(process_infos):
                p = info["process"]
                if p.poll() is not None:
                    print(f"Process {p.pid} ({info['module']}) exited early. Restarting...")
                    process_infos[i] = start_single_client(venv_python, info["module"], project_root)
                else:
                    print(f"Process {p.pid} ({info['module']}) reached 60s. Restarting...")
                    process_infos[i] = restart_client(info, venv_python, project_root)


            # 定期同步目标网络
            if episode % 10 == 0:
                client.sync_target_network()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = os.path.join(save_dir, f"client_dqn_ep{episode}_{total_reward}_{timestamp}.pt")
                torch.save(client.policy_net.state_dict(), model_path)
                print(f"[模型保存] Episode {episode} 模型已保存至 {model_path}")
                #client.request_global_model()


            # 输出训练信息
            print(f"Client {client_id} Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {client.epsilon:.2f}")


# ====================== 联邦接口 ======================
'''
class FederatedServer:
    def __init__(self, num_clients):
        self.global_model = ClientDQN()
        self.client_models = {f"client{i}":ClientDQN() for i in range(num_clients)}

    def aggregate(self):
        """联邦平均聚合"""
        global_dict = self.global_model.state_dict()
        for key in global_dict:
            global_dict[key] = torch.mean(
                torch.stack([client.state_dict()[key] for idx , client in self.client_models]),
                dim=0
            )
        self.global_model.load_state_dict(global_dict)

    def distribute_model(self):
        """分发全局模型到各客户端"""
        for idx, client in self.client_models:
            #client.load_state_dict(self.global_model.state_dict())
            client.receive_global_model(self.global_model.get_local_parameters())

    def set_client_model(self,idx,state__dict):
        """读入客户端的新模型"""
        self.client_models[idx].receive_global_model(state__dict)

'''
# ====================== 使用示例 ======================
if __name__ == "__main__":
    client_ip = sys.argv[1] if len(sys.argv) > 1 else '10.0.0.2'
    client_name = sys.argv[2] if len(sys.argv) > 2 else 'client1'
    init_model = sys.argv[3] if len(sys.argv) > 3 else  './saved_models/client_dqn_ep50_6.793304088961961_20250509_133940.pt'  # 新增第三个参数
    print(f"start: {client_ip},{client_name}")
    if client_name == 'client1':
        modules.append("Client.main1")
    elif client_name == 'client2':
        modules.append("Client.main2")
    elif client_name == 'client3':
        modules.append("Client.main3")
    set_log()
    for module in modules:
        process_infos.append(start_single_client(venv_python, module, project_root))
    time.sleep(10)
    try:
    # 启动训练时传入初始模型
        client_training_loop(
            client_id=client_name,
            num_episodes=60,
            ep=1,
            init_model=init_model,
            reason=True
        )
    except BaseException as e:
        # 捕获 KeyboardInterrupt、SystemExit 以及其他所有异常
        print(f"\nExit received ({type(e).name}): {e}")
    finally:
        print("\nKeyboard interrupt received. Stopping all clients...")
        for info in process_infos:
            p = info["process"]
            if p.poll() is None:
                print(f"Terminating process {p.pid}...")
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"Process {p.pid} did not terminate, killing...")
                    p.kill()