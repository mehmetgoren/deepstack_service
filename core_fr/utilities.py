import base64
import os
import uuid
from enum import Enum
from io import BytesIO
from threading import Thread

from deepstack_sdk import ServerConfig, Face

from common.utilities import config


class EventChannels(str, Enum):
    read_service = 'read_service'
    fr_train_request = 'fr_train_request'
    fr_service = 'fr_service'
    fr_train_response = 'fr_train_response'
    frtc = 'frtc'


def create_face() -> Face:
    ds_config = config.deep_stack
    ds = ServerConfig(f'{ds_config.server_url}:{ds_config.server_port}')
    ds.api_key = config.deep_stack.api_key
    face = Face(ds)
    return face


def get_train_dir_path() -> str:
    return os.path.join(config.general.root_folder_path, 'fr', 'ml', 'train')


def create_dir_if_not_exist(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def start_thread(fn, args):
    th = Thread(target=fn, args=args)
    th.daemon = True
    th.start()


def generate_id() -> str:
    return str(uuid.uuid4().hex)


def pil_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
