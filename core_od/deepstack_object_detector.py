from typing import List
from deepstack_sdk import ServerConfig, Detection
from deepstack_sdk.structs import DetectionResponse
import numpy.typing as npt

from common.utilities import config, logger
from core_od.models.coco_objects import coco80_object
from core_od.models.detections import DetectionBox, DetectionResult


class DeepstackObjectDetector:
    def __init__(self):
        self.ds_config = config.deep_stack
        ds = ServerConfig(f'{self.ds_config.server_url}:{self.ds_config.server_port}')
        ds.api_key = config.deep_stack.api_key
        self.min_confidence = self.ds_config.od_threshold
        self.detection = Detection(ds)

    def get_results(self, img: npt.NDArray, detected_by: str) -> List[DetectionResult]:
        ret: List[DetectionResult] = []
        try:
            detections: DetectionResponse = self.detection.detectObject(img, min_confidence=self.min_confidence)
            for d in detections:
                cls_idx = coco80_object.get_index(d.label)
                box = DetectionBox()
                box.x1, box.y1, box.x2, box.y2 = d.x_min, d.y_min, d.x_max, d.y_max
                r = DetectionResult()
                r.box = box
                r.pred_cls_name, r.pred_cls_idx, r.pred_score = d.label, cls_idx, d.confidence
                ret.append(r)
        except BaseException as ex:
            logger.error(f'an error occurred while detection api call, source: {detected_by}, ex: {ex}')
        return ret
