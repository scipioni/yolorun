import asyncio
import glob
import logging
import os
import platform
import signal
import socket
import struct
import time
import shutil

from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory

import configargparse
import cv2 as cv
import numpy as np
from shared_ndarray2 import SharedNDArray
from .timing import timing

try:
    from pypylon import genicam, pylon
except:
    pass

from .timing import timing

# from dnnutils.utils import get_bboxes, get_bboxes_txt

from . import aioudp
from .bboxes import BBox, BBoxes

log = logging.getLogger(__name__)


class Grabber:
    name = "generic"

    def __init__(self, config):
        self.config = config
        self._started_at = time.time()
        self.counter = 0
        self.frame_id = 0
        self.fps_mean = 25.0
        self.key = ""
        self.video = False
        self.is_running = True
        self.grey = False
        self.session = int(1000 * time.time())
        # atexit.register(self.close)

    async def start(self):
        """overload me"""
        pass

    def get(self, key=None):
        self.counter += 1

        elapsed = time.time() - self._started_at
        fps = 1 / elapsed
        if self.counter > 1:
            self.fps_mean = (
                self.fps_mean * ((self.counter - 1) / self.counter) + fps / self.counter
            )
        self._started_at = time.time()
        if self.counter % (2 * int(self.fps_mean)) == 0:  # ogni 2 secondi circa
            log.info(
                "camera-id=%s %.1f FPS (mean %.1f) "
                % (self.config.camera_id, fps, self.fps_mean)
            )

        if not key and self.config.show:
            key = cv.waitKey(1) & 0xFF
        if key:
            self.key = chr(key)
        if key in (ord("q"), 27):
            return (None, "", [])
        elif key in (ord("h"),):
            self.config.show = not self.config.show
        elif self.key in ("d",):
            self.config.debug = not self.config.debug

        return (True, key, [])

    def getBboxes(self, classes):
        return []

    def close(self):
        log.info("close")

    def move(self, filename, path):
        if not os.path.isdir(path):
            if os.path.exists(path):
                os.remove(path)
            os.makedirs(path)
        log.info("move %s to %s", filename, path)

        classes_txt = os.path.join(os.path.dirname(filename), "classes.txt")
        if os.path.exists(classes_txt):
            shutil.copy(classes_txt, path)

        for f in glob.glob(filename.split(".")[0] + ".*"):
            shutil.move(f, path)

    def postprocess(self, frame):
        if self.config.crop:
            h, w = frame.shape[:2]
            l = min(h, w)
            x1 = int((w - l) / 2)
            y1 = int((h - l) / 2)
            return frame[y1 : y1 + l, x1 : x1 + l]
        return frame


class DummyGrabber(Grabber):
    name = "dummy"

    def __init__(self, config):
        super().__init__(config)

    async def get(self, key=None):
        (do_continue, buff, _) = super().get(key=key)
        if not do_continue:
            return (None, "", [])
        return (np.zeros((1024, 1024, 3), np.uint8), self.counter, [])


class WebcamGrabber(Grabber):
    name = "webcam"

    def __init__(self, config):
        super().__init__(config)
        self._vid = None

    async def get(self, key=None):
        (do_continue, buff, _) = super().get(key=key)
        if not do_continue:
            return (None, "", [])
        if not self._vid:
            self._vid = cv.VideoCapture(self.config.url or 0)

        ret, frame = self._vid.read()
        if not ret:
            return (None, "", [])

        return (frame, self.counter, [])


class RtspGrabber(Grabber):
    name = "rtsp"

    def __init__(self, config):
        super().__init__(config)
        self._vid = None

    async def get(self, key=None):
        (do_continue, buff, _) = super().get(key=key)
        if not do_continue:
            return (None, self.counter, [])
        if not self._vid:
            # connection_string = f"""rtspsrc location={self.config.url} protocols={self.config.protocol} latency=0 ! rtph264depay ! h264parse ! tee name=h264
            #     h264. ! queue ! {self.config.decoder} ! videorate ! video/x-raw,framerate={framerate}/{divisor} ! videoconvert ! video/x-raw, format=BGR  ! videoconvert ! appsink drop=true
            #     """
            url = self.config.images[0]
            self._vid = cv.VideoCapture(
                f"rtspsrc location={url} protocols=tcp latency=0 ! rtph264depay ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1",
                cv.CAP_GSTREAMER,
            )
            if not self._vid.isOpened():
                return (None, self.counter, [])
            log.info("... OK, connected")
        ret, frame = self._vid.read()
        if not ret:
            return (None, self.counter, [])
        frame = self.postprocess(frame)
        return (frame, self.counter, [])


class FileGrabber(Grabber):
    def __init__(self, config, files):
        super().__init__(config)
        # self.i = 0
        self.current = -1
        self.current_video = -99
        self.isNew = True
        self.setFiles(files)

    def setFiles(self, files):
        self.video = None

        for index, filename in enumerate(files):
            if filename[0] != "/":
                files[index] = filename  # os.path.join(self.config.path, filename)

        if files and (
            ".mp4" in files[0]
            or ".mkv" in files[0]
            or ".mov" in files[0]
            or ".webm" in files[0]
        ):
            self.video = files[0]
            log.info("video detected: %s" % self.video)
        else:
            with timing("glob", count=1):
                if files and "*" in files[0]:
                    files = [f for f in glob.iglob(files[0])]

            self.files = files
            log.info("detected %d files", len(files))

        if self.video:
            self.cap = cv.VideoCapture(self.video)
        else:
            self.cap = None

        self.current_frame = None
        self.current_bboxes = None

    # def getBboxes(self, classes):
    #     if self.current_frame is None or self.video:
    #         return []
    #     (h, w) = self.current_frame.shape[:2]
    #     bboxes = get_bboxes(
    #         self.files[self.current].replace(".jpg", ".xml"), classes=classes
    #     )
    #     if not bboxes:
    #         bboxes = get_bboxes_txt(
    #             self.files[self.current].replace(".jpg", ".txt"), w, h, classes=classes
    #         )
    #     return bboxes

    async def get(self, key=None):
        (do_continue, buff, _) = super().get(key=key)
        self.isNew = self.current_frame is None
        if not do_continue:
            return (None, "", [])

        if not (self.config.show and self.config.step):
            self.current += 1
            self.isNew = True

        if self.current < 0:
            self.current = 0

        if self.key in ("n", "."):
            self.current += 1
            self.isNew = True
        elif key == ord(" "):
            self.config.save = not self.config.save
        elif self.key in ("p", ","):
            self.current -= 1
        elif self.key in ("c",):
            self.config.step = not self.config.step

        if self.video:
            if self.current_video != self.current:
                ret, img = self.cap.read()
                if img is None:
                    return (None, "", [])
                # self.current += 1
                if self.grey:
                    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                self.current_video = self.current
                img = self.postprocess(img)
                self.current_frame = img
            if self.key in ("i",):
                log.info(
                    "filename: path=%s shape=%s", self.video, self.current_frame.shape
                )
        elif 0 <= self.current < len(self.files):
            if self.isNew:
                with timing("cv.imread"):
                    # https://github.com/libvips/pyvips/issues/179#issuecomment-618936358
                    filename = self.files[self.current]
                    img = cv.imread(filename)
                    if img is None:
                        return (None, "", None)
                    # img = np.zeros((1, 1, 1), np.uint8)
                    filetxt = filename.split(".")[0] + ".txt"

                    h, w = img.shape[:2]
                    self.current_bboxes = BBoxes()
                    self.current_bboxes.import_txt(filetxt, w, h)
                if self.grey:
                    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                self.current_frame = img
            if self.key in ("i",):
                log.info(
                    "filename: path=%s shape=%s",
                    self.files[self.current],
                    self.current_frame.shape,
                )
        else:
            return (None, "", self.current_bboxes)

        if self.key in ("s",):
            filename = os.path.join(
                self.config.save_path, "%s.jpg" % int(time.time() * 1000)
            )
            log.info("save image: %s", filename)
            cv.imwrite(filename, self.current_frame, [cv.IMWRITE_JPEG_QUALITY, 100])
        elif self.key in ("r",):
            print(f"delete file {self.files[self.current]}")
            name = self.files[self.current].split(".")[0]
            for _f in glob.glob(f"{name}*"):
                os.remove(_f)
            self.files.pop(self.current)
            self.current += 1
            self.isNew = True

        if self.video:
            return (
                self.current_frame,
                "%s-%d.jpg"
                % (os.path.basename(self.video).split(".")[0], self.current),
                [],
            )
        else:
            return self.current_frame, self.files[self.current], self.current_bboxes


class DcamGrabber(Grabber):
    name = "dcam"

    def __init__(self, config):
        super().__init__(config)
        self.video = "dcam"
        self.shm = None
        self.sock_listen = None
        self.camera_id = int(self.config.camera_id[0])
        self._counter = 0
        # camera = self.setup_dc1394()
        # if camera:
        #     camera.close()

        #     self.cap = cv.VideoCapture(0, cv.CAP_FIREWIRE)

    # def setUpListen(self):
    #     if self.camera_id == 1: # se sono la camera 1 ascolto la camera 0
    #         self.sock_listen = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    #         self.sock_listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    #         self.sock_listen.bind(('', self.config.socket_port + 0)) # ascolto la camera 0

    def poll(self):
        sock_emit = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        # sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2
        # self.setUpListen()

        if self.cap.isOpened():
            while self.is_running:
                (do_continue, buff) = super().get()
                if not do_continue:
                    break
                ret, frame = self.cap.read()  # questo blocca
                if frame is None:
                    log.warn("skip null frame")
                    time.sleep(0.1)
                    continue
                self.pushFrame(frame)

                # if self.sock_listen: # ci blocchiamo fino all'arrivo dell'altro frame
                #     with timing("receive"):
                #         self.sock_listen.recv(1024)

                self.emit(sock_emit, frame.shape[:2])
                if self.config.show:
                    cv.imshow("image", frame)
            self.close()

    def pushFrame(self, frame):
        if self.shm is None or (
            self.config.check_shm and self.frame_id % self.config.check_shm == 0
        ):
            name = "frame_load_%s" % self.camera_id
            try:
                shm = SharedMemory(name=name, create=True, size=frame.size)
            except:
                shm = SharedMemory(name=name, create=False, size=frame.size)
            finally:
                self.shm = SharedNDArray(shm.name, frame.shape, frame.dtype)
                log.debug(f"shared {name} created size={frame.nbytes}")

        shared = self.shm.get()
        shared[:] = frame[:]

    def emit(self, sock, shape):
        port = self.config.socket_port + self.camera_id

        log_message = (
            f"DcamGrabber.emit(): camera_id {self.camera_id}, "
            f"session {self.session}, frame id {self.frame_id}, "
            f"shape {shape}, port={port}"
        )
        log.debug(log_message)
        shape_channel = shape[2] if len(shape) == 3 else 0
        try:
            self._counter += 1
            sock.sendto(
                struct.pack(
                    "illiii",
                    self.camera_id,
                    self.session,
                    self.frame_id,
                    shape[0],
                    shape[1],
                    shape_channel,
                ),
                (self.config.socket_ip, port),
            )
        except:
            if self._counter % self.config.camera_framerate == 0:
                log.warn(
                    "failed emit frame ready packet to %s:%s",
                    self.config.socket_ip,
                    port,
                )

    def emitForGps(self, sock):
        """
        serve in fase di sviluppo per sincronizzare il GPS generato anche quando si rallentano i frames
        """
        if self.config.gps_sync:
            # print(f"send camera_id={self.camera_id} session={self.session} port={port}")
            try:
                sock.sendto(
                    struct.pack("il", self.camera_id, self.counter),
                    (self.config.socket_ip_gps, self.config.socket_port_gps),
                )
            except:
                log.warn(
                    "failed emitForGps to %s:%s",
                    self.config.socket_ip_gps,
                    self.config.socket_port_gps,
                )

    def close(self):
        self.is_running = False

        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None

    def setup_dc1394(self):
        """
        settiamo lo shutter e lasciamo automatico il gain e exposure

        ['Y8', 'YUV411', 'YUV422', 'YUV444', 'RGB8', 'Y16', 'RAW8', 'RAW16']
        """
        try:
            camera = Camera()
        except Exception as e:
            log.error("no camera: %s" % str(e))
            return False

        camera.mode = camera.modes_dict["FORMAT7_0"]
        # camera.mode.color_coding="YUV444" # 36fps colore

        camera.framerate.setup(value=self.config.camera_framerate)

        # camera.mode.image_size=(2048, 1536)
        camera.mode.image_size = [
            self.config.camera_cols,
            self.config.camera_rows,
        ]  # perdiamo le fascie alte e basse dove c'è sfuocatura
        camera.mode.image_position = [0, self.config.camera_top]

        # camera.mode.color_coding="RGB8" #"MONO8"
        if self.config.camera_color:
            camera.mode.color_coding = "YUV411"
        else:
            camera.mode.color_coding = "MONO8"  # 54fps grigio
        # camera.mode.color_coding="YUV411" # 36fps colore

        camera.shutter.setup(
            value=self.config.camera_shutter, model="manual", absolute=False
        )
        # camera.saturation.setup(mode='auto')

        # camera.sharpness.setup(mode='auto')
        camera.gain.setup(mode="auto")
        camera.exposure.setup(mode="auto")

        # print("Color coding: ", camera.mode.color_codings)
        # print("Mode:", camera.mode)

        return camera


class DcamGrabberNative(DcamGrabber):
    name = "dcam-native"

    def __init__(self, config, camera_id=0):
        Grabber.__init__(self, config)
        self.video = "dcam"
        self.shm = None
        self.camera_id = camera_id
        self.camera = self.setup_dc1394()

    def poll(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        # self.setUpListen()

        self.camera.start_capture()
        self.camera.start_video()

        while self.is_running:
            (do_continue, buff) = self.get()
            self.frame_id += 1
            if not do_continue:
                break
            matrix = self.camera.dequeue()
            frame = matrix.copy()
            # frame = matrix.to_rgb()
            matrix.enqueue()
            self.pushFrame(frame)
            self.emit(sock, frame.shape)
            if self.config.show:
                cv.imshow("image", frame)
        self.camera.stop_video()
        self.camera.stop_capture()

    def close(self):
        self.is_running = False


class DcamGrabberFake(DcamGrabber):
    name = "dcam-fake"

    def __init__(self, config):
        DcamGrabber.__init__(self, config)
        self.video = "dcam"
        self.shm = None
        self.save_process = None
        self.taskqueue = Queue()
        # self.frame = cv.cvtColor(cv.imread(config.camera_fake), cv.COLOR_BGR2GRAY)

    def poll(self):
        def _read():
            ret, frame = self.cap.read()
            self.frame_id += 1
            if frame is not None:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            return frame

        self.cap = cv.VideoCapture(self.config.camera_fake)
        sock_emit = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        # self.setUpListen()

        frame_id_current = 0
        frame = _read()
        while self.is_running:
            if (self.frame_id != frame_id_current) and (
                self.frame_id - 1
            ) % self.config.camera_framerate == 0:
                self.emitForGps(sock_emit)
                frame_id_current = self.frame_id

            start = time.time()
            (do_continue, _) = self.get()

            if not do_continue:
                break

            if self.config.show and self.config.step and self.key in ("n", "."):
                frame = _read()

            if not self.config.step:
                frame = _read()

            if frame is None:
                break
            self.pushFrame(frame)
            self.emit(sock_emit, frame.shape)

            if self.config.show:
                cv.imshow("image", frame)

            h, w = frame.shape[:2]
            if not self.save_process and self.config.save:  # inizio la registrazione
                self.save_process = Process(
                    target=save, args=(self.taskqueue, (w, h), self.config)
                )
                self.save_process.start()

            if self.save_process and not self.config.save:  # finisco la registrazione
                self.taskqueue.put((None, None))
                if self.save_process:
                    self.save_process.join()  # attendiamo
                self.save_process = None

            if self.config.save:
                self.taskqueue.put((frame, None))  # position))

            elapsed = time.time() - start
            time.sleep(max(0, 1.0 / self.config.camera_framerate - elapsed))

        if self.save_process:  # finisco la registrazione
            self.taskqueue.put((None, None))
            self.save_process.join()  # attendiamo

    def close(self):
        self.cap.release()
        self.is_running = False


class SharedMemoryGrabber(Grabber):
    """
    riceve i frames
    """

    name = "shared-memory"

    def __init__(self, config, camera_id=0):
        Grabber.__init__(self, config)
        self.video = "shared"
        self.camera_id = int(camera_id)
        # self.slave = slave
        self.frame = None
        self._task = None

    async def start(self):
        # if self.slave:
        port = self.config.socket_port + self.camera_id
        log.info(f"camera_id={self.camera_id}, listening on port {port}")
        self._local_udp = await aioudp.open_local_endpoint(port=port)

    async def get(self, key=None):
        (do_continue, buff) = super().get(key=key)
        if not do_continue:
            return (None, "")
        # if trigger:
        #     camera_id, session, counter, h, w = trigger
        # else:
        data, address = await self._local_udp.receive()
        camera_id, session, counter, h, w, c = struct.unpack("illiii", data)

        log.debug(
            f"Grabbler.get(): camera_id {camera_id}, session {session}, frame id {counter}, shape {h, w, c} "
        )

        if self.session != session:
            name = "frame_load_%s" % camera_id
            log.info(f"shared memory init new session {session} attach to {name}")
            self.session = session
            shm = SharedMemory(name=name, create=False)
            self.shm_frame = (
                SharedNDArray(shm.name, (h, w, c), np.uint8)
                if c
                else SharedNDArray(shm.name, (h, w), np.uint8)
            )

        # TODO si potrebbe evitare di fare il copy se i tempi di elaborazione successivi sono minori della successiva chiamata a questa funzione
        return (
            self.shm_frame.get().copy(),
            (camera_id, session, counter, h, w),
        )  # copiamo subito in area di memoria separata

    async def pollAsync(self):
        self._task = asyncio.create_task(self._pollAsync())

    async def _pollAsync(self):
        await self.start()
        while self.is_running:
            frame, trigger = await self.get()
            if frame is None:
                break
            if frame is not None:
                self.frame = frame.copy()

    def getCache(self):
        return self.frame

    async def close(self):
        self.is_running = False
        if self._task:
            await self._task

    # def getCurrent(self):
    #     return self.shm_frame


class BaslerGrabber(Grabber):
    name = "baslerdcam"
    _fontsize = 0.7

    def __init__(self, config):
        super().__init__(config)
        self.video = "dcam"
        self.shm = None
        self.sock_listen = None
        self.camera_id = int(self.config.camera_id[0])
        self.taskqueue = Queue()
        self.save_process = None

        self.camera = self.setup()

        labelSize, baseLine = cv.getTextSize(
            "X", cv.FONT_HERSHEY_SIMPLEX, self._fontsize, 2
        )
        self.rowSize = labelSize[0] + 12

    def poll(self):
        sock_emit = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

        converter = pylon.ImageFormatConverter()

        # converting to opencv greyscale or BGR
        converter.OutputPixelFormat = pylon.PixelType_Mono8
        # converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        if not self.camera:
            log.error("no camera detected")
            return

        frame_id_current = 0
        while self.is_running and self.camera.IsGrabbing():
            try:
                grabResult = self.camera.RetrieveResult(
                    5000, pylon.TimeoutHandling_ThrowException
                )
            except SystemError:
                self.is_running = False
                break

            if not grabResult.GrabSucceeded():
                continue

            self.frame_id += 1

            if (self.frame_id != frame_id_current) and (
                self.frame_id - 1
            ) % self.config.camera_framerate == 0:
                self.emitForGps(sock_emit)
                frame_id_current = self.frame_id

            (do_continue, key) = super().get()
            if not do_continue:
                break
            if key == ord(" "):
                self.config.save = not self.config.save
            image = converter.Convert(grabResult)
            frame = image.GetArray()

            self.config.camera_shutter = grabResult.ChunkExposureTime.GetValue() / 1000
            gain = grabResult.ChunkGain.GetValue()

            grabResult.Release()

            # grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            if self.config.save or self.config.show:
                self.drawInfo(frame, gain=gain, roi=not self.config.save)

            if self.config.show:
                cv.imshow("image", frame)
            self.pushFrame(frame)
            self.emit(sock_emit, frame.shape[:2])

            h, w = frame.shape[:2]
            if not self.save_process and self.config.save:  # inizio la registrazione
                self.save_process = Process(
                    target=save, args=(self.taskqueue, (w, h), self.config)
                )
                self.save_process.start()

            if self.save_process and not self.config.save:  # finisco la registrazione
                self.taskqueue.put((None, None))
                if self.save_process:
                    self.save_process.join()  # attendiamo
                self.save_process = None

            if self.config.save:
                self.taskqueue.put((frame, None))  # position))

        if self.save_process:  # finisco la registrazione
            self.taskqueue.put((None, None))
            self.save_process.join()  # attendiamo

        self.close()

    def setup(self):
        """
        settiamo lo shutter e lasciamo in automatico il resto

        """
        log.warn(
            "initialize basler camera id=%s serial=%s",
            self.camera_id,
            self.config.camera_serial,
        )
        info = pylon.DeviceInfo()
        if self.config.camera_serial:
            info.SetSerialNumber(self.config.camera_serial)
        try:
            camera = pylon.InstantCamera(
                pylon.TlFactory.GetInstance().CreateFirstDevice(info)
            )
            camera.Open()
        except Exception as e:
            log.error("no camera: %s" % str(e))
            return None

        # for s in (dir(camera)):
        #     if 'ROI' in s:
        #         print(s)

        camera.UserSetSelector.SetValue("Default")
        camera.UserSetLoad.Execute()

        camera.MaxNumBuffer = 2

        camera.StaticChunkNodeMapPoolSize = camera.MaxNumBuffer.GetValue()
        if genicam.IsWritable(camera.ChunkModeActive):
            camera.ChunkModeActive = True
        camera.ChunkSelector = "ExposureTime"
        camera.ChunkEnable = True
        camera.ChunkSelector = "Gain"
        camera.ChunkEnable = True

        camera.PixelFormat = "Mono8"
        # camera.PixelFormat = "BGR8"

        camera.AcquisitionFrameRate = self.config.camera_framerate
        camera.AcquisitionFrameRateEnable = True

        if self.config.camera_flip_y:
            camera.ReverseY = True
            camera.ReverseX = True

        if self.config.legacy:
            camera.GainAuto.SetValue("Continuous")
            camera.BalanceWhiteAuto.SetValue("Continuous")
            camera.AutoFunctionProfile.SetValue("MinimizeGain")

            camera.Height = self.config.camera_rows
            offsetY = int((camera.Height.Max - camera.Height.GetValue()) / 2)
            if offsetY % 2 == 1:
                offsetY -= 1
            camera.OffsetY = offsetY

            camera.Width = self.config.camera_cols
            camera.OffsetX = 4

            camera.ExposureTime = self.config.camera_shutter * 1000

        else:
            camera.AutoTargetBrightness = (
                self.config.camera_AutoTargetBrightness
            )  # maggiore di .19, vogliamo un'immagine scura
            camera.Gamma.SetValue(
                1
            )  # di solito 1. Se minore l'immagine diventa più chiara
            camera.AutoGainUpperLimit = (
                self.config.camera_AutoGainUpperLimit
            )  # fino a 23
            camera.AutoExposureTimeUpperLimit = (
                self.config.camera_AutoExposureTimeUpperLimit
            )
            camera.BlackLevel = 0  # fino a 30
            log.warn(
                "AutoTargetBrightness=%.2f AutoGainUpperLimit=%.2f AutoExposureTimeUpperLimit=%.2f",
                camera.AutoTargetBrightness.GetValue(),
                camera.AutoGainUpperLimit.GetValue(),
                camera.AutoExposureTimeUpperLimit.GetValue(),
            )
            camera.GainAuto.SetValue("Continuous")
            # camera.BalanceWhiteAuto.SetValue("Continuous")
            try:  # le camere mono non hanno questa prop
                camera.BalanceWhiteAuto.SetValue("Off")
            except Exception:
                pass

            # camera.ExposureTime = self.config.camera_shutter * 1000
            camera.ExposureAuto.SetValue("Continuous")

            # camera.AutoFunctionProfile.SetValue("MinimizeGain")
            camera.AutoFunctionProfile.SetValue(
                "MinimizeExposureTime"
            )  # MinimizeExposureTime")

            camera.AutoFunctionROISelector = "ROI1"

            camera.Height = self.config.camera_rows
            camera.AutoFunctionROIHeight = self.config.camera_roi_rows
            if self.config.camera_top == -1:
                offsetY = int(
                    (camera.Height.Max - camera.Height.GetValue()) / 2
                )  # centriamo
            else:
                offsetY = self.config.camera_top
            if offsetY % 2 == 1:
                offsetY -= 1
            camera.OffsetY = offsetY

            offsetY_roi = int(
                (offsetY + camera.Height.Max - self.config.camera_roi_rows) / 2
            )
            camera.AutoFunctionROIOffsetY = offsetY_roi

            camera.Width = self.config.camera_cols
            camera.AutoFunctionROIWidth = self.config.camera_roi_cols

            offsetX = int((camera.Width.Max - camera.Width.GetValue()) / 2)
            while offsetX % 4 > 0:
                offsetX -= 1
            camera.OffsetX = offsetX

            offsetX_roi = int(
                (offsetX + camera.Width.Max - self.config.camera_roi_cols) / 2
            )
            camera.AutoFunctionROIOffsetX = offsetX_roi

            # camera.AutoFunctionROIUseBrightness.SetValue(True)
            # camera.AutoFunctionROIUseWhiteBalance.SetValue(True)

        pylon.FeaturePersistence.Save(
            "/tmp/dcam%s.pfs" % self.camera_id, camera.GetNodeMap()
        )

        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        return camera

    def drawInfo(self, frame, position=None, box_width=200, gain=0, roi=False):
        box_width = 300  # px
        h, w = frame.shape[:2]

        rows = []
        rows.append(
            f"{self.config.camera_cols}x{self.config.camera_rows}@{self.config.camera_framerate}"
        )
        rows.append("e:%05dus g:%.1f" % (self.config.camera_shutter * 1000, gain))
        if position:
            rows.append("%.5f" % position[0])
            rows.append("%.5f" % position[1])
            rows.append("%2.1fm" % position[2])

        cv.rectangle(
            frame,
            (w - box_width, 0),
            (w, len(rows) * self.rowSize + 8),
            (0, 0, 255) if self.config.save else (0, 0, 0),
            cv.FILLED,
        )  # rettangolo score

        for i, row in enumerate(rows):
            cv.putText(
                frame,
                row,
                (w - box_width + 3, self.rowSize * (i + 1)),
                cv.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )

        if roi:
            cv.rectangle(
                frame,
                (
                    int((w - self.config.camera_roi_cols) / 2),
                    int((h - self.config.camera_roi_rows) / 2),
                ),
                (
                    int((w + self.config.camera_roi_cols) / 2),
                    int((h + self.config.camera_roi_rows) / 2),
                ),
                (0, 0, 255),
                1,
            )

    def pushFrame(self, frame):
        if self.shm is None or (
            self.config.check_shm and self.frame_id % self.config.check_shm == 0
        ):
            name = "frame_load_%s" % self.camera_id
            try:
                shm = SharedMemory(
                    name=name,
                    create=True,
                    size=self.config.camera_rows * self.config.camera_cols * 1,
                )
            except:
                shm = SharedMemory(name=name, create=False)
            finally:
                self.shm = SharedNDArray(shm.name, frame.shape, frame.dtype)
                log.debug(f"shared '{name}' created size={frame.nbytes}")
        shared_memory = self.shm.get()
        shared_memory[:] = frame[:]

    def emit(self, sock, shape):
        port = self.config.socket_port + self.camera_id
        log_message = (
            f"BaslerGrabber.emit(): camera_id {self.camera_id}, "
            f"session {self.session}, frame id {self.frame_id}, "
            f"shape {shape}, port={port}"
        )
        log.debug(log_message)

        shape_channel = shape[2] if len(shape) == 3 else 0
        sock.sendto(
            struct.pack(
                "illiii",
                self.camera_id,
                self.session,
                self.frame_id,
                shape[0],
                shape[1],
                shape_channel,
            ),
            (self.config.socket_ip, port),
        )

    def close(self):
        log.info("closing ...")
        self.is_running = False
        self.camera.StopGrabbing()
        if self.save_process:
            self.save_process.join()

    def dumpCfg(self, filename="settings.txt"):
        pylon.FeaturePersistence.Save(filename, self.camera.GetNodeMap())

    def emitForGps(self, sock):
        """ """
        sock.sendto(
            struct.pack("il", self.camera_id, self.counter),
            (self.config.socket_ip, self.config.socket_port_gps),
        )


def save(taskqueue, size, config):
    now = int(time.time())
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)

    filename = os.path.join(config.save_path, "%d_%s.mkv" % (now, config.camera_id[0]))
    if platform.machine() == "x86_64":
        fourcc = cv.VideoWriter_fourcc(*"MJPG")
        out = cv.VideoWriter(filename, fourcc, config.camera_framerate, size)
    else:
        log.info("arm64 detected, activating hardware encoder: %s", config.save_encoder)
        fourcc = cv.VideoWriter_fourcc(*"H264")
        out = cv.VideoWriter(
            f"""appsrc ! videoconvert ! video/x-raw,format=I420 ! nvvideoconvert ! {config.save_encoder} ! h264parse ! matroskamux ! filesink location={filename}""",
            cv.CAP_GSTREAMER,
            fourcc,
            config.camera_framerate,
            size,
        )
    if not out.isOpened():
        log.error("cannot save")
        return

    log.info("start video record %s", filename)

    try:
        while True:
            image, position = taskqueue.get()
            if image is None:
                break
            out.write(cv.cvtColor(image, cv.COLOR_GRAY2BGR))
    except KeyboardInterrupt:
        pass

    print(f"closing video {filename} ...")
    out.release()
    print(f"... saved {filename}")


def dcam():
    global log

    ap.add_argument("--show", action="store_true", default=False, help="")
    ap.add_argument("--step", action="store_true", default=False, help="step image")
    config = ap.parse_args()
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
        level="DEBUG" if config.debug else "INFO",
    )
    log = logging.getLogger(__name__)

    if config.show:
        cv.namedWindow("image", cv.WINDOW_NORMAL)
        cv.resizeWindow("image", 2048, 1024)

    if config.debug:
        import jurigged

        jurigged.watch()
        import debugpy

        debugpy.listen(("0.0.0.0", 5678))
        log.warning("debug mode: started debugpy and jurigged")

    if config.camera_fake:
        grabber = DcamGrabberFake(config)
    elif config.camera_native:
        grabber = DcamGrabberNative(config)
    elif config.camera_basler:
        grabber = BaslerGrabber(config)
    else:
        grabber = DcamGrabber(config)

    def save_on(signum, stack):
        config.save = True

    def save_off(signum, stack):
        config.save = False

    signal.signal(signal.SIGUSR1, save_on)
    signal.signal(signal.SIGUSR2, save_off)

    if config.delay_start > 0:
        log.warning("delay at start: sleep for %d seconds ...", config.delay_start)
        time.sleep(config.delay_start)
    while True:
        grabber.poll()
        if config.show or not config.camera_loop:
            break


def test_nvidia_encoder():
    now = int(time.time())

    fourcc = cv.VideoWriter_fourcc(*"H264")
    out = cv.VideoWriter(
        f"""appsrc is-live=true ! videoconvert ! video/x-raw,format=I420 ! nvvideoconvert ! nvv4l2h264enc bitrate=8000000 ! h264parse ! matroskamux ! filesink location=/tmp/out.mkv""",
        cv.CAP_GSTREAMER,
        fourcc,
        20,
        (416, 416),
    )
    if not out.isOpened():
        print("cannot save")
        return

    img = np.zeros((416, 416, 1), dtype=np.uint8)

    for i in range(20):
        out.write(cv.cvtColor(img, cv.COLOR_GRAY2BGR))
