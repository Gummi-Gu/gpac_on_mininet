import BufferFilter
import DASH
import Rendering
import Interfaces
import Message


width=4096#3840
height=2048#1920
Winname=None
press_start=True
dash=None
viewpoint=None
fs=None
bufferFilter=None
dash_interface=None
render=None
Monitor_URL= "http://127.0.0.1:10087/update"
ThreadedCommunication=Message.ThreadedCommunication

def init():
    global dash,viewpoint,fs,bufferFilter,dash_interface,render
    dash= DASH.MyCustomDASHAlgo()
    viewpoint= Rendering.Viewpoint(press_start)
    fs = BufferFilter.MyFilterSession()
    bufferFilter= BufferFilter.MyFilter(fs)
    render= Rendering.Render()
    dash_interface=Interfaces.DashInterface()