import sys
import cv2
import Client.Factory as Factory
import Client.model.pre as re
sys.path.append("C:/Users/GummiGu/毕业设计/代码/gpac/share/python")
import libgpac as gpac

#initialize gpac
gpac.init()
#indicate we want to start with min bw by using global parameters
gpac.set_args(["Ignored", "--start_with=max_bw"])
Factory.press_start = 0


def main():
    # cv2.namedWindow('360 View')
    # cv2.setMouseCallback('360 View', mouse_callback)

    # create a custom filter session
    fs = Factory.fs
    re.start()
    # load a source filter
    # if a parameter is passed to the script, use this as source
    if len(sys.argv) > 1:
        src = fs.load_src(sys.argv[1])
    # otherwise load one of our DASH sequences
    else:
        src = fs.load_src("http://127.0.0.1:10086/01/files/dash_tiled.mpd")
        #src = fs.load_src("http://192.168.16.174:10081/01/files/dash_tiled.mpd")

    # load our custom filter and assign its source
    my_filter = Factory.bufferFilter
    my_filter.set_source(src)

    # and run
    fs.run()

    # fs.print_graph()

    fs.delete()
    gpac.close()

if __name__ == '__main__':
    main()