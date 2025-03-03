from flask import Flask, send_from_directory
import os

app = Flask(__name__)

# 设置文件根目录，假设 DASH_DIR 是文件存放的目录
DASH_DIR = '/home/mininet/gpac_on_mininet/dash'

@app.route('/<folder>/<filename>')
def serve_file(folder, filename):
    # 如果请求的是 high 或 low 目录下的文件
    print(folder,filename)
    if folder not in ['high', 'low']:
        return "Directory not found", 404

    # 获取完整的文件路径
    file_path = os.path.join(DASH_DIR, folder, filename)
    print(file_path)
    # 检查文件是否存在
    if not os.path.isfile(file_path):
        return "File not found", 404

    # 返回该目录下的文件
    return send_from_directory(os.path.join(DASH_DIR, folder), filename)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1080, debug=True)
