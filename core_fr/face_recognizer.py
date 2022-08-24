from typing import List
import base64
from io import BytesIO
from PIL import Image, ImageDraw
from deepstack_sdk.structs import FaceRecognitionResponse

from common.utilities import logger, config
from core_fr.utilities import create_face, pil_to_base64


class DetectedFace:
    pred_cls_idx: int = 0
    pred_cls_name: str = ''
    pred_score: float = .0
    crop_base64_image: str = ''
    x1, y1, x2, y2 = 0, 0, 0, 0

    def format(self) -> str:
        return f'{self.pred_cls_idx}_{self.pred_cls_name}_{self.pred_score}'


class DetectedFaceImage:
    detected_faces: List[DetectedFace] = []
    base64_image: str = ''


class FaceRecognizer:
    def __init__(self):
        self.face = create_face()
        self.min_confidence = config.deep_stack.fr_threshold
        self.overlay = config.ai.overlay

    def predict(self, base64_img: str) -> DetectedFaceImage:
        ret = DetectedFaceImage()
        ret.base64_image = base64_img
        ret.detected_faces = []
        pil_img = Image.open(BytesIO(base64.b64decode(base64_img)))
        try:
            response: FaceRecognitionResponse = self.face.recognizeFace(pil_img, min_confidence=self.min_confidence)
            for index, d in enumerate(response.detections):
                df = DetectedFace()
                df.pred_score, df.pred_cls_idx, df.pred_cls_name = d.confidence, index, d.userid
                df.x1, df.y1, df.x2, df.y2 = d.x_min, d.y_min, d.x_max, d.y_max
                ret.detected_faces.append(df)
        except BaseException as ex:
            logger.error(f'an error occurred while face api call, ex: {ex}')

        if self.overlay:
            for detected_face in ret.detected_faces:
                xy1 = (detected_face.x1, detected_face.y1)
                xy2 = (detected_face.x2, detected_face.y2)
                draw = ImageDraw.Draw(pil_img)
                draw.rectangle((xy1, xy2), outline='yellow')
                text = detected_face.pred_cls_name
                draw.text(xy1, text)
            ret.base64_image = pil_to_base64(pil_img)
        else:
            ret.base64_image = base64_img

        return ret
