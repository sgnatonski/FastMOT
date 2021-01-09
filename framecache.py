import json
import asyncio
import cv2
import time
from collections import deque, OrderedDict
from os import path
from json import dumps, loads
from threading import Thread
import struct

from nats.aio.client import Client as NATS
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE
import proto.gen.detections_pb2

def read_counter():
    return loads(open("counter.json", "r").read()) + 1 if path.exists("counter.json") else 0


def write_counter():
    with open("counter.json", "w") as f:
        f.write(dumps(counter))


counter = read_counter()
write_counter()

class FixSizeOrderedDict(OrderedDict):
    def __init__(self, *args, max=0, **kwargs):
        self._max = max
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        if self._max > 0:
            if len(self) > self._max:
                self.popitem(False)

class FrameCache:
    def __init__(self, size, cacheSize, natsUrl):
        self.size = size    
        self.natsUrl = natsUrl    
        self.frame_queue = FixSizeOrderedDict(max=cacheSize)
        self.jpeg = TurboJPEG()

        thr1 = Thread(target=self.start)
        thr1.start()
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.loop.run_until_complete(self.nc.drain())
        self.loop.close()

    def start(self):
        self.loop = asyncio.new_event_loop()
        self.loop.run_until_complete(self.run())
        self.loop.run_forever()
    
    async def run(self):
        self.nc = NATS()

        await self.nc.connect(self.natsUrl, loop=self.loop)

        def frame_handler(msg):

            data = json.loads(msg.data.decode())
            frame_id = int(data['frame'])
            frame = self.frame_queue.get(frame_id)

            if frame is None:
                diff = frame_id - next(reversed(self.frame_queue))
                print("Frame {frame} not found in cache. Frame diff = {diff}".format(frame=frame_id, diff=diff))
                asyncio.run_coroutine_threadsafe(
                    self.nc.publish(msg.reply, bytearray(0)),
                    loop=self.loop)
            else:
                quality = int(data['quality'] or 40)
                crop_x = int(data['crop']['x'] or 0)
                crop_y = int(data['crop']['y'] or 0)
                crop_w = int(data['crop']['w'] or 0)
                crop_h = int(data['crop']['h'] or 0)
                
                if crop_w == 0 and crop_h == 0:
                    jpg = self.jpeg.encode(frame, quality=quality)
                if crop_w > 0 and crop_h > 0:
                    jpg = self.jpeg.encode(frame, quality=quality)
                    if crop_x < 0:
                        crop_x = 0
                    if crop_y < 0:
                        crop_y = 0
                    height, width = frame.shape[:2]
                    if crop_x + crop_w > width:
                        crop_w = width - crop_x
                    if crop_y + crop_h > height:
                        crop_h = height - crop_y
                    jpg = self.jpeg.crop(jpg, crop_x, crop_y, crop_w, crop_h)
                   
                asyncio.run_coroutine_threadsafe(
                    self.nc.publish(msg.reply, jpg),
                    loop=self.loop)

        await self.nc.subscribe("frame", cb=frame_handler)

    def publishTracks(self, frame_id, frame, tracks):
        self.frame_queue[frame_id] = frame

        frame = proto.gen.detections_pb2.Frame()
        frame.frame = int(frame_id)
        frame.ex = counter

        for track in tracks:
            det = frame.det.add()
            det.id = int(track.trk_id)
            det.cl = int(track.label)
            det.x = int(track.tlbr[0])
            det.y = int(track.tlbr[1])
            det.w = int(track.tlbr[2] - track.tlbr[0])
            det.h = int(track.tlbr[3] - track.tlbr[1])
            det.ft = track.smooth_feature.tobytes()

        asyncio.run_coroutine_threadsafe(
            self.nc.publish("detections", frame.SerializeToString()),
            loop=self.loop)