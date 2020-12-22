from collections import deque, OrderedDict
import json
import asyncio
from nats.aio.client import Client as NATS
from nats.aio.errors import ErrConnectionClosed, ErrTimeout, ErrNoServers
from fastmot.models import LABEL_MAP
import cv2
import time
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE


from threading import Thread

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
    def __init__(self, size, cacheSize):
        self.size = size        
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

        await self.nc.connect("nats://127.0.0.1:4222", loop=self.loop)

        def frame_handler(msg):
            #tic = time.perf_counter()

            data = str(msg.data.decode()).split(",")
            frame_id = int(data[0])
            frame = self.frame_queue.get(frame_id)

            if frame is None:
                diff = frame_id - next(reversed(self.frame_queue))
                print("Frame {frame} not found in cache. Frame diff = {diff}".format(frame=frame_id, diff=diff))
                asyncio.run_coroutine_threadsafe(
                    self.nc.publish(msg.reply, bytearray(0)),
                    loop=self.loop)
            else:
                jpg = self.jpeg.encode(frame, quality=40)
                asyncio.run_coroutine_threadsafe(
                    self.nc.publish(msg.reply, jpg),
                    loop=self.loop)
            
            #toc = time.perf_counter()
            #elapsed_time = toc - tic
            #print("Frame handle = {elapsed}".format(elapsed=elapsed_time))

        await self.nc.subscribe("frame", cb=frame_handler)

    def appendFrame(self, frame_id, frame):
        self.frame_queue[frame_id] = frame

    def publishTracks(self, frame_id, tracks):
        payload = {
            "frame": int(frame_id),
            "det": []
        }
        for track in tracks:
            payloadTrack = {
                "id": int(track.trk_id),
                "cl": LABEL_MAP[track.label],
                "pr": 1, # todo confidence
                "x": int(track.tlbr[0]),
                "y": int(track.tlbr[1]),
                "w": int(track.tlbr[2] - track.tlbr[0]),
                "h": int(track.tlbr[3] - track.tlbr[1])
            }

            payload["det"].append(payloadTrack)

        json_payload = json.dumps(payload)

        asyncio.run_coroutine_threadsafe(
            self.nc.publish("detections", bytearray(json_payload, 'utf-8')),
            loop=self.loop)