from select import select

import Client.Factory as Factory

class MyCustomDASHAlgo:
    #get notifications when a DASH period starts or ends
    def __init__(self):
        self.srd_position=[]
        self.srd_quantity=[]
        self.comm = Factory.ThreadedCommunication(
            url=Factory.Monitor_URL,
            timeout=5,
            retries=2,
            retry_delay=1,
            max_queue_size=100
        )
        #self.comm.start()

    def on_period_reset(self, type):
        print('period reset type ' + str(type))

    #get notification when a new group (i.e., set of adaptable qualities, `AdaptationSet` in DASH) is created. Some groups may be left unexposed by the DASH client
    #the qualities are sorted from min bandwidth/quality to max bandwidth/quality
    def on_new_group(self, group):
        #print('new group ' + str(group.idx) + ' qualities ' + str(len(group.qualities)) + ' codec ' + group.qualities[0].codec);
        srd=group.SRD
        #print(f'{srd.x} {srd.y}')
        self.srd_position.append(srd)

    #perform adaptation logic - return value is the new quality index, or -1 to keep as current, -2 to discard  (debug, segments won't be fetched/decoded)
    def on_rate_adaptation(self, group, base_group, force_low_complexity, stats):
        #print('We are adapting on group ' + str(group.idx) )
        # print('' + str(stats))
        # Perform adaptation, check group.SRD to perform spatial adaptation, ...
        #
        # In this example we simply cycle through qualities
        # Send the newq value via GET request to 127.0.0.1:12567/dash
        select_num=Factory.dash_interface.set_quality(group.idx,self.srd_position)
        #Factory.videoSegmentStatus.set_quality_tiled(group.idx,select_num)
        #select_num=1
        #sample_slices = [
        #    {"idx": group.idx, "bitrate": group.qualities[select_num].bandwidth,
        #     "download_speed": group.qualities[select_num].bandwidth, "fps": str(group.qualities[select_num].fps),
        #     "resolution": f'{group.qualities[select_num].width}x{group.qualities[select_num].height}'}
        #]
        #self.comm.send(self.packet_video_data(stats.buffer, stats.buffer_max, stats.download_rate, sample_slices))
        #print(f"[DASH]{stats.buffer} {stats.buffer_min} {stats.buffer_max} {stats.download_rate}")
        #print(f'[DASH]For {group.idx} We choose {select_num}')
        return select_num

    # this callback is optional, use it only if your algo may abort a running transfer (this can be very costly as it will require closing and reopening the HTTP connection for HTTP 1.1  )
    #   -1 to continue download
    #   or -2 to abort download but without retrying to download a segment at lower quality for the same media time
    #   or the index of the new quality to download for the same media time
    def on_download_monitor(self, group, stats):
        print('download monitor group ' + str(group.idx) + ' stats ' + str(stats) );
        return -1


    def packet_video_data(self,buffer_length, max_buffer_length, download_speed,slices):
        """
        发送视频传输数据到 Flask 服务器。

        :param buffer_length: 当前 buffer 长度（秒）
        :param max_buffer_length: 最大 buffer 长度（秒）
        :param slices: 一个包含视频切片信息的列表，每个切片是一个字典
        """
        data = {
            "buffer_length": buffer_length,
            "max_buffer_length": max_buffer_length,
            "download_speed": download_speed,
            "slices": slices
        }
        return data
