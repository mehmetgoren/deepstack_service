from typing import List
import numpy as np
from deepstack_sdk import ServerConfig, Detection
from deepstack_sdk.structs import DetectionResponse

from common.utilities import config, logger
from core_od.models.detected_objects import Coco80DetectedObject, BaseDetectedObject, coco80_info
from core_od.models.object_detector_model import BaseObjectDetectorModel, DetectionBox


class DeepstackObjectDetectorModel(BaseObjectDetectorModel):
    def __init__(self):
        super(DeepstackObjectDetectorModel, self).__init__()
        self.ds_config = config.deep_stack
        ds = ServerConfig(f'{self.ds_config.server_url}:{self.ds_config.server_port}')
        ds.api_key = config.deep_stack.api_key
        self.min_confidence = self.ds_config.od_threshold
        self.detection = Detection(ds)

    def get_detect_boxes(self, img: np.array, detected_by: str) -> List[DetectionBox]:
        boxes: List[DetectionBox] = []
        try:
            detections: DetectionResponse = self.detection.detectObject(img, min_confidence=self.min_confidence)
            for d in detections:
                cls_idx = coco80_info.get_index(d.label)
                boxes.append(DetectionBox(d.x_min, d.y_min, d.x_max, d.y_max, d.confidence, d.label, cls_idx))
        except BaseException as ex:
            logger.error(f'an error occurred while detection api call, source: {detected_by}, ex: {ex}')
        return boxes

    def create_detected_object(self, img: np.array, detected_by: str, box: DetectionBox) -> BaseDetectedObject:
        obj = Coco80DetectedObject(img, box.confidence, box.cls_name)
        obj.detected_by = detected_by
        return obj

    def get_detected_object_class_name(self, cls_idx: int) -> str:
        return coco80_info.get_name(cls_idx)
