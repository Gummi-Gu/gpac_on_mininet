import BufferFilter
import DASH
import Rendering
import Interfaces
import Message

dash=None
viewpoint=None
fs=None
bufferFilter=None
dash_interface=None
render=None
Monitor_URL= "http://localhost:5000/update"
ThreadedCommunication=Message.ThreadedCommunication

def init():
    global dash,viewpoint,fs,bufferFilter,dash_interface,render
    dash= DASH.MyCustomDASHAlgo()
    viewpoint= Rendering.Viewpoint()
    fs = BufferFilter.MyFilterSession()
    bufferFilter= BufferFilter.MyFilter(fs)
    render= Rendering.Render()
    dash_interface=Interfaces.DashInterface()