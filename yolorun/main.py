import asyncio
import logging
#import time
from pathlib import Path

from .config import CLASSES, COLORS
from .lib.grabber import DummyGrabber, FileGrabber, Grabber, WebcamGrabber, RtspGrabber

log = logging.getLogger(__name__)

from . import models


async def grab(config, grabber: Grabber, model: models.Model) -> None:
    while True:
        try:
            frame, filename, bboxes_truth = await grabber.get()
        except Exception as e:
            raise
            log.error(e)
            break

        if frame is None:
            break

        if config.move:
            if config.filter_classes_strict:
                if bboxes_truth.hasOnly(config.filter_classes_strict):
                    grabber.move(filename, config.move)
            elif config.filter_classes:
                if bboxes_truth.has(config.filter_classes):
                    grabber.move(filename, config.move)

        model.predict(frame)

        bboxes_predicted = model.getBBoxes()

        if config.save:
            if config.merge:
                merged_bboxes = bboxes_truth.merge(bboxes_predicted, config.filter_classes)
                merged_bboxes.save(frame, filename, config.save)
            else:
                if bboxes_predicted.has(config.filter_classes):
                    bboxes_predicted.save(
                        frame, filename, config.save, include=config.filter_classes
                    )

        if config.show:
            frame_show = frame.copy()
            if bboxes_truth:
                frame_show = bboxes_truth.draw(frame_show)
            frame_show = bboxes_predicted.draw(frame_show)
            model.show(frame_show)


def main():
    from .config import get_config

    config = get_config()

    if config.webcam:
        grabber = WebcamGrabber(config)
    elif config.images:
        if "rtsp" in config.images[0]:
            grabber = RtspGrabber(config)
        else:
            grabber = FileGrabber(config, config.images)
    else:
        grabber = DummyGrabber(config)

    model = models.getModel(config)
    log.info("model %s %s", model.__class__, model)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(grab(config, grabber, model))
    loop.close()


if __name__ == "__main__":
    main()
