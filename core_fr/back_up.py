from os import path
import requests
from io import open
import shutil

from common.utilities import config, logger


class BackUp:
    def __init__(self):
        self.server_url = f'{config.deep_stack.server_url}:{config.deep_stack.server_port}'
        self.full_file_path: str = path.join(config.general.root_folder_path, 'deepstack', 'backupdeepstack.zip')

    def backup(self):
        try:
            data = requests.post(f'{self.server_url}/v1/backup', stream=True)
            with open(self.full_file_path, 'wb') as file:
                shutil.copyfileobj(data.raw, file)
            del data
            logger.info('back-up operation has been completed successfully')
        except BaseException as ex:
            logger.warning(f'an error occurred while backing up the DeepStack file, err: {ex}')

    def restore(self):
        try:
            image_data = open(self.full_file_path, 'rb').read()
            _ = requests.post(f'{self.server_url}/v1/restore', files={'file': image_data}).json()
            logger.info('back-up operation has been completed successfully')
        except BaseException as ex:
            logger.warning(f'an error occurred while restoring the DeepStack file, err: {ex}')
