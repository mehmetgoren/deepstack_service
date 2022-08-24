import json

from common.event_bus.event_bus import EventBus
from common.event_bus.event_handler import EventHandler
from common.utilities import logger, config, datetime_now
from core_fr.face_recognizer import FaceRecognizer
from core_fr.utilities import generate_id, start_thread, EventChannels


class FrReadServiceEventHandler(EventHandler):
    def __init__(self):
        self.prob_threshold: float = config.deep_stack.fr_threshold
        self.encoding = 'utf-8'
        self.fr = FaceRecognizer()
        self.publisher = EventBus(EventChannels.fr_service)

    def handle(self, dic: dict):
        if dic is None or dic['type'] != 'message':
            return

        start_thread(self.__handle, [dic])

    def __handle(self, dic: dict):
        try:
            data: bytes = dic['data']
            dic = json.loads(data.decode(self.encoding))
            name = dic['name']
            source_id = dic['source']
            base64_image = dic['img']
            ai_clip_enabled = dic['ai_clip_enabled']

            result = self.fr.predict(base64_image)
            if result is None:
                logger.info(f'image contains no face for camera: {name}')
                return

            detected_faces = []
            face_logs = []
            for face in result.detected_faces:
                prob = face.pred_score
                if prob < self.prob_threshold:
                    logger.info(f'prob ({prob}) is lower than threshold: {self.prob_threshold} for {face.pred_cls_name}')
                    continue
                detected_faces.append({'pred_score': prob, 'pred_cls_idx': int(face.pred_cls_idx), 'pred_cls_name': face.pred_cls_name,
                                       'crop_base64_image': face.crop_base64_image,
                                       'x1': float(face.x1), 'y1': float(face.y1), 'x2': float(face.x2), 'y2': float(face.y2)})
                face_logs.append({'pred_cls_name': face.pred_cls_name, 'pred_score': face.pred_score}),
            if len(detected_faces) == 0:
                logger.info('no detected face prob score is higher than threshold, this event will not be published')
                return

            dic = {'id': generate_id(), 'source_id': source_id, 'created_at': datetime_now(), 'detected_faces': detected_faces,
                   'base64_image': result.base64_image, 'ai_clip_enabled': ai_clip_enabled}
            event = json.dumps(dic)
            self.publisher.publish(event)
            logger.info(f'face: detected {json.dumps(face_logs)}')
        except BaseException as ex:
            logger.error(f'an error occurred while handling an facial-recognition request by DeepStack, err: {ex}')
