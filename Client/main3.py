import sys
import threading
import Client.Factory as Factory
import Client.model.pre as re
sys.path.append("C:/Users/GummiGu/毕业设计/代码/gpac/share/python")
import libgpac as gpac

#initialize gpac
gpac.init()
#indicate we want to start with min bw by using global parameters
gpac.set_args(["Ignored", "--start_with=max_bw"])
Factory.press_start = 0
Factory.clientname='client3'
Factory.Factory_init()


def run_pipeline():
    # 创建一个自定义的 filter session
    fs = Factory.fs
    re.start()
    ip=Factory.ip_maps[Factory.clientname]
    # 加载视频源
    if len(sys.argv) > 1:
        src = fs.load_src(sys.argv[1])
    else:
        src = fs.load_src(f"http://{ip}:10086/01/files/dash_tiled.mpd")

    # 加载自定义 filter 并设置其源
    my_filter = Factory.bufferFilter
    my_filter.set_source(src)

    # 运行 pipeline
    fs.run()

    # 清理
    fs.delete()
    gpac.close()

def main():
    t = threading.Thread(target=run_pipeline)
    t.start()
    print("视频处理线程已启动。")

if __name__ == '__main__':
    main()