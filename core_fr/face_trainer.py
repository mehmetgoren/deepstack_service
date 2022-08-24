import os
import time
from os.path import isfile, join
from typing import List

from common.utilities import logger
from core_fr.back_up import BackUp
from core_fr.utilities import create_face, get_train_dir_path, create_dir_if_not_exist


class FaceTrainer:
    def __init__(self):
        self.face = create_face()
        self.folder_path = get_train_dir_path()
        create_dir_if_not_exist(self.folder_path)

    def fit(self):
        for fc in self.face.listFaces():
            try:
                self.face.deleteFace(fc)
            except BaseException as ex:
                logger.error(f'en error occurred while deleting a DeepStack face, ex: {ex}')

        need_backup = False
        for dir_name in os.listdir(self.folder_path):
            full_path_dir = join(self.folder_path, dir_name)
            files = os.listdir(full_path_dir)
            images: List[str] = []
            for image_path in files:
                full_path_file = join(full_path_dir, image_path)
                if isfile(full_path_file):
                    images.append(full_path_file)
            if len(images) > 0:
                self.face.registerFace(images=images, userid=dir_name)
                need_backup = True
        if need_backup:
            time.sleep(3.)
            backup = BackUp()
            backup.backup()

