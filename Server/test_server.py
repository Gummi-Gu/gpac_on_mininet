from flask import Flask, send_from_directory
import os

app = Flask(__name__)

# 设置文件根目录，假设 DASH_DIR 是文件存放的目录
DASH_DIR = '/home/mininet/gpac_on_mininet/Server/files'
#DASH_DIR = 'files'

@app.route('/files/<filename>')
def serve_file(filename):
    # 获取完整的文件路径
    file_path = DASH_DIR

    # 检查文件是否存在
    if not os.path.isfile(os.path.join(file_path, filename)):
        return "File not found", 404

    # 返回该目录下的文件
    return send_from_directory(file_path, filename)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10086, debug=True)
