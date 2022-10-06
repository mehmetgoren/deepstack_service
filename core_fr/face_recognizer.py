from typing import List
from PIL import Image
from deepstack_sdk.structs import FaceRecognitionResponse

from common.utilities import logger, config
from core_fr.utilities import create_face


class DetectedFace:
    pred_cls_idx: int = 0
    pred_cls_name: str = ''
    pred_score: float = .0
    x1, y1, x2, y2 = 0, 0, 0, 0

    def format(self) -> str:
        return f'{self.pred_cls_idx}_{self.pred_cls_name}_{self.pred_score}'


class FaceRecognizer:
    def __init__(self):
        self.face = create_face()
        self.min_confidence = config.deep_stack.fr_threshold

    def predict(self, pil_img: Image) -> List[DetectedFace]:
        ret: List[DetectedFace] = []
        try:
            response: FaceRecognitionResponse = self.face.recognizeFace(pil_img, min_confidence=self.min_confidence)
            for index, d in enumerate(response.detections):
                df = DetectedFace()
                df.pred_score, df.pred_cls_idx, df.pred_cls_name = d.confidence, index, d.userid
                df.x1, df.y1, df.x2, df.y2 = d.x_min, d.y_min, d.x_max, d.y_max
                ret.append(df)
        except BaseException as ex:
            logger.error(f'an error occurred while face api call, ex: {ex}')

        return ret
