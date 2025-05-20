from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from collections import defaultdict


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


class FederatedServer:
    def __init__(self, num_clients):
        self.num_clients = num_clients
        # 加载预训练模型
        pretrained = torch.load('./client_dqn_ep50_6.793304088961961_20250509_133940.pt',
                                map_location='cpu')
        # 初始化全局模型
        self.global_model = ClientDQN()
        self.global_model.load_state_dict(pretrained)

        # 初始化客户端模型
        self.client_models = {}
        for i in range(num_clients):
            client_id = f"client{i}"
            model = ClientDQN()
            model.load_state_dict(pretrained)
            self.client_models[client_id] = model

        self.upload_counter = defaultdict(int)

    def aggregate(self):
        """修正后的联邦平均聚合"""
        global_dict = self.global_model.state_dict()
        for key in global_dict:
            # 收集所有客户端的参数（排除未更新的客户端）
            client_params = [
                client.state_dict()[key].float()
                for client in self.client_models.values()
            ]
            if client_params:
                global_dict[key] = torch.mean(torch.stack(client_params), dim=0)
        self.global_model.load_state_dict(global_dict)

    def distribute_model(self):
        """模型分发方法"""
        global_state = self.global_model.state_dict()
        for client in self.client_models.values():
            client.load_state_dict(global_state)

    def check_aggregation(self, client_id):
        """检查是否达到聚合条件"""
        self.upload_counter[client_id] += 1
        # 统计有效上传次数
        valid_uploads = sum(1 for cnt in self.upload_counter.values() if cnt > 0)
        return valid_uploads >= self.num_clients


app = Flask(__name__)
NUM_CLIENTS = 5
server = FederatedServer(NUM_CLIENTS)


@app.route('/get_global_model', methods=['GET'])
def handle_get_global_model():
    """获取最新全局模型"""
    try:
        global_state = server.global_model.state_dict()
        serialized = {k: v.tolist() for k, v in global_state.items()}
        return jsonify({
            "status": "success",
            "model_weights": serialized
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/upload_client_model/<string:client_id>', methods=['POST'])
def handle_upload_client_model(client_id):
    """处理模型上传并自动触发聚合"""
    try:
        # 验证客户端ID
        if client_id not in server.client_models:
            return jsonify({"status": "error", "message": "Invalid client ID"}), 400

        # 解析上传数据
        uploaded_data = request.json.get('model_weights')
        if not uploaded_data:
            return jsonify({"status": "error", "message": "No weights provided"}), 400

        # 转换张量格式
        converted_weights = {}
        for key, value in uploaded_data.items():
            converted_weights[key] = torch.tensor(value, dtype=torch.float32)

        # 更新客户端模型
        server.client_models[client_id].load_state_dict(converted_weights)

        # 检查聚合条件
        if server.check_aggregation(client_id):
            server.aggregate()
            server.distribute_model()
            # 重置计数器
            server.upload_counter.clear()
            return jsonify({
                "status": "aggregated",
                "message": "Global model updated and distributed"
            })

        return jsonify({
            "status": "success",
            "message": f"Model updated for {client_id}",
            "remaining": NUM_CLIENTS - sum(server.upload_counter.values())
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, threaded=True)