import time
import VAS_system_mp
import asyncio

MODE_IP_CAM = 0
MODE_VIDEO = 1

VAS = VAS_system_mp.VAS_system()

# RTSP URL example: "rtsp://admin:BCT!234$@192.168.205.215:554/profile3/media.smp"
username = "admin"
password = "BCT!234$"

video_start = time.time()

Video_Info = [username, password]
while True:
    try:
        status = VAS.system_Operation(Video_Info, MODE_IP_CAM)
        if not status:
            time.sleep(5)
            continue
    except KeyboardInterrupt:
        break
