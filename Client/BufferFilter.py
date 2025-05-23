import sys
import time

import cv2
from pyarrow import timestamp

import Client.Factory  as Factory

sys.path.append("C:/Users/GummiGu/毕业设计/代码/gpac/share/python")
import libgpac as gpac

class MyFilterSession(gpac.FilterSession):
    def __init__(self, flags=0, blacklist=None, nb_threads=0, sched_type=0):
        gpac.FilterSession.__init__(self, flags, blacklist, nb_threads, sched_type)

    def on_filter_new(self, f):
        # bind the dashin filter to our algorithm object
        if f.name == "dashin":
            f.bind(Factory.dash)

class MyFilter(gpac.FilterCustom):
    def __init__(self, session):
        gpac.FilterCustom.__init__(self, session, "PYRawVid")
        # indicate what we accept and produce - here, raw video input only (this is a sink)
        self.tmp_pck = None
        self.push_cap("StreamType", "Visual", gpac.GF_CAPS_INPUT)
        self.push_cap("CodecID", "Raw", gpac.GF_CAPS_INPUT)

        self.max_buffer = 10000000
        self.play_buffer = 1
        self.re_buffer = 1
        self.buffer=0
        self.buffering = True
        self.rebuff_time=0
        self.rebuff_count=0
        self.rebuff_sum_time=0
        self.dur=0.0

    def set_rebuffer_playbuffer(self,v1,v2):
        self.re_buffer=v1
        self.play_buffer=v2

    # configure input PIDs
    def configure_pid(self, pid, is_remove):
        if is_remove:
            return 0
        if pid in self.ipids:
            print('PID reconfigured')
        else:
            print('PID configured')

            #1- setup buffer levels - the max_playout_us and min_playout_us are only informative for the filter session
            #but are forwarded to the DASH algo
            evt = gpac.FilterEvent(gpac.GF_FEVT_BUFFER_REQ)
            evt.buffer_req.max_buffer_us = self.max_buffer
            evt.buffer_req.max_playout_us = self.play_buffer*8000000
            evt.buffer_req.min_playout_us = self.re_buffer*8000000
            pid.send_event(evt)

            #2-  we are a sink, we MUST send a play event
            evt = gpac.FilterEvent(gpac.GF_FEVT_PLAY)
            pid.send_event(evt)

        # get width, height, stride and pixel format - get_prop may return None if property is not yet known
        # but this should not happen for these properties with raw video, except StrideUV which is NULL for non (semi) planar YUV formats
        self.width = pid.get_prop('Width')
        self.height = pid.get_prop('Height')
        self.pixfmt = pid.get_prop('PixelFormat')
        self.stride = pid.get_prop('Stride')
        self.stride_uv = pid.get_prop('StrideUV')
        self.timescale = pid.get_prop('Timescale')
        return 0

    # process

    def process(self):
        start_time = time.time()
        if self.rebuff_time!=0:
            dur_time=start_time-self.rebuff_time
            if  dur_time - self.dur > 0.4:
                    self.rebuff_sum_time += (dur_time - self.dur - 0.2)
                    self.rebuff_count+=1
        self.rebuff_time=start_time
        #only one PID in this example
        for pid in self.ipids:
            title = ""
            if pid.eos:
                pass
                # not done, check buffer levels
            else:
                buffer=self.buffer
                if buffer<self.re_buffer*self.timescale:
                    self.buffering = True
                elif buffer>self.play_buffer*self.timescale:
                    self.buffering = False
                title=f'buffer: {self.buffer/self.timescale:.2f}s - {self.buffer/(self.play_buffer*self.timescale)*100:.2f}%'
            pck = pid.get_packet()
            if pck is None:
                break
            if pck.frame_ifce:
                self.tmp_pck = pck.clone(self.tmp_pck)
                if self.tmp_pck == None:
                    raise Exception("Packet clone failed")
                data = self.tmp_pck.data
            else:
                data = pck.data
            #convert to cv2 image for some well known formats
            #note that for YUV formats here, we assume stride luma is width and stride chroma is width/2
            #shoe img in 360 view
            if self.pixfmt == 'nv12':
                yuv = data.reshape((self.height * 3 // 2, self.width))
                rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_NV12)
            dur = pck.dur
            dur /= self.timescale
            #Factory.render.render(rgb,title)
            self.buffer=Factory.render.push_frame(rgb,title,dur)
            #print(buffer)
            Factory.videoSegmentStatus.set_rgb(rgb)
            #get packet duration for later sleep
            if self.buffering is False:
                time.sleep(dur*5)
            self.dur=dur
            # dummy player, this does not take into account the time needed to draw the frame, so we will likely drift
            pid.drop_packet()

        return 0