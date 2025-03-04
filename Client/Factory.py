import BufferFilter
import DASH
import Rendering
import Interfaces

dash=None
viewpoint=None
fs=None
bufferFilter=None
quantitycollector=None
dash_interface=None

def init():
    global dash,viewpoint,fs,bufferFilter,dash_interface
    dash_interface=Interfaces.DashInterface()
    dash= DASH.MyCustomDASHAlgo()
    viewpoint= Rendering.Viewpoint()
    fs = BufferFilter.MyFilterSession()
    bufferFilter= BufferFilter.MyFilter(fs)