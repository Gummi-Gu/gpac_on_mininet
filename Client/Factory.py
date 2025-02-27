import BufferFilter
import DASH
import Viewpoint
import QualityCollector

dash=None
viewpoint=None
fs=None
bufferFilter=None
quantitycollector=None

def init():
    global dash,viewpoint,fs,bufferFilter,quantitycollector
    dash= DASH.MyCustomDASHAlgo()
    viewpoint= Viewpoint.Viewpoint()
    fs = BufferFilter.MyFilterSession()
    bufferFilter= BufferFilter.MyFilter(fs)
    quantitycollector=QualityCollector.VideoQualityMetrics()
