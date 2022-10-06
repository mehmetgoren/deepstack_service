import sys
import time
from multiprocessing import Process

from core_fr.back_up import BackUp
from common.event_bus.event_bus import EventBus
from common.utilities import logger, config
from core_fr.event_handlers import FrReadServiceEventHandler
from core_fr.train_event_handler import TrainEventHandler
from core_fr.utilities import EventChannels, start_thread
from core_od.deepstack_object_detector import DeepstackObjectDetector
from core_od.event_handlers import OdReadServiceEventHandler
from core_od.utilities import register_detect_service
from docker_manager import DockerManager


def setup_fr():
    def train_event_handler():
        logger.info('DeepStack face training event handler will start soon')
        eb = EventBus(EventChannels.fr_train_request)
        th = TrainEventHandler()
        eb.subscribe_async(th)

    start_thread(fn=train_event_handler, args=[])

    handler = FrReadServiceEventHandler()

    logger.info('DeepStack face recognition service will start soon')
    event_bus = EventBus(EventChannels.read_service)
    event_bus.subscribe_async(handler)
    sys.exit()


def setup_od():
    detector = DeepstackObjectDetector()
    event_bus = EventBus(EventChannels.snapshot_in)
    handler = OdReadServiceEventHandler(detector)
    logger.info('DeepStack service will start soon')
    event_bus.subscribe_async(handler)


def main():
    if len(config.deep_stack.server_url) == 0:
        logger.error('Config.Deepstack.ServerUrl is empty, the deepstack service is now exiting')
        return

    dckr_mngr = None
    backup = None
    try:
        dckr_mngr = DockerManager()
        backup = BackUp()

        dckr_mngr.run()
        time.sleep(3.)
        backup.restore()

        register_detect_service('deepstack_service', 'deepstack_service-instance', 'The Deepstack Object Detection and Facial Recognition Service®')
        ds_config = config.deep_stack

        proc_fr = None
        if ds_config.fr_enabled:
            logger.info('DeepStack Facial Recognition is enabled')
            proc_fr = Process(target=setup_fr, args=())
            proc_fr.daemon = True
            proc_fr.start()
        else:
            logger.warning('DeepStack Facial Recognition is not enabled')

        if ds_config.od_enabled:
            logger.info('DeepStack Object Detection is enabled')
            setup_od()
        else:
            logger.warning('DeepStack Object Detection is not enabled')

        if proc_fr is not None:
            proc_fr.join()
            proc_fr.terminate()
        logger.warning('DeepStack Service has been ended')
    except BaseException as ex:
        logger.error(f'an error occurred on DeepStack Service main function, ex: {ex}')
    finally:
        if backup is not None:
            backup.backup()
            time.sleep(3.)
        if dckr_mngr is not None:
            dckr_mngr.remove()


if __name__ == '__main__':
    main()
