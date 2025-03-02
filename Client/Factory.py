import BufferFilter
import DASH
import Rendering
import Interfaces

dash=None
rendering=None
fs=None
bufferFilter=None
dashinterface=None

def init():
    global dash,rendering,fs,bufferFilter, dashinterface
    dash= DASH.MyCustomDASHAlgo()
    rendering= Rendering.Rendering()
    fs = BufferFilter.MyFilterSession()
    bufferFilter= BufferFilter.MyFilter(fs)
    dashinterface=Interfaces.DashInterface()
