import os
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# 启用CORS支持，允许所有来源
CORS(app)

# 设置文件根目录
FILE_DIRECTORY = 'files'  # 这里是存放文件的目录，请根据实际情况修改

# 文件服务器路由
@app.route('/files/<path:filename>')
def serve_file(filename):
    # 返回请求的文件
    return send_from_directory(FILE_DIRECTORY, filename)

@app.route('/dash', methods=['GET'])
def dash():
    # 获取传递的 'value' 参数
    value = request.args.get('value', type=int)

    if value is None:
        return jsonify({"error": "Missing 'value' parameter"}), 400

    # 返回 value + 1 的 JSON 响应
    return jsonify({"value": value + 1})



if __name__ == '__main__':
    if not os.path.exists(FILE_DIRECTORY):
        os.makedirs(FILE_DIRECTORY)  # 确保文件目录存在
    app.run(debug=True, host='0.0.0.0', port=1256)

