import sys
import time

import cv2

import Factory

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
        self.play_buffer = 1000000
        self.re_buffer = 100000
        self.buffering = True


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
            evt.buffer_req.max_playout_us = self.play_buffer
            evt.buffer_req.min_playout_us = self.re_buffer
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
        start_time=time.time()
        #only one PID in this example
        for pid in self.ipids:
            title = 'GPAC cv2'

            if pid.eos:
                pass
            #not done, check buffer levels
            else:
                buffer = pid.buffer
                if self.buffering:
                    #playout buffer not yet filled
                    if buffer < self.play_buffer:
                        pc = 100 * buffer / self.play_buffer
                        #$title += " - buffering " + str(int(pc)) + ' %'
                        break

                    #playout buffer refilled
                    #title += " - resuming"
                    self.buffering = False

                if self.re_buffer:
                    #playout buffer underflow
                    if buffer < self.re_buffer:
                        #title += " - low buffer, pausing"
                        self.buffering = True
                        break

                #show max buffer level
                if self.max_buffer > self.play_buffer:
                        pc = buffer / self.max_buffer * 100
                        #title += " - buffer " + str(int(buffer/1000000)) + 's ' + str(int(pc)) + ' %'

            pck = pid.get_packet()
            if pck is None:
                break
            #frame interface, data is in GPU memory or internal to decoder, try to grab it
            #we do so by creating a clone of the packet, reusing the same clone at each call to reduce memory allocations
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

            Factory.render.render(rgb,title)

            #get packet duration for later sleep
            dur = pck.dur
            dur /= self.timescale

            pid.drop_packet()

            cv2.waitKey(1)

            # dummy player, this does not take into account the time needed to draw the frame, so we will likely drift
            time.sleep(max(0,dur-(time.time() - start_time)))
            #print("[BufferFilter]",time.time() - start_time)
        return 0