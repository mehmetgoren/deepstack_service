from os import path
from typing import List, Any
import docker
from docker.types import Mount

from common.config import DeepStackDockerType, DeepStackPerformanceMode
from common.utilities import config, logger


# for more info: https://docker-py.readthedocs.io/en/stable/containers.html
class DockerManager:
    def __init__(self):
        self.client = docker.from_env()
        self.container_name = 'deepstack-server'
        self.ds_config = config.deep_stack

    def __get_image_name(self) -> str:
        docker_type = self.ds_config.docker_type
        if docker_type == DeepStackDockerType.CPU:
            return 'deepquestai/deepstack'
        elif docker_type == DeepStackDockerType.GPU:
            return 'deepquestai/deepstack:gpu'
        elif docker_type == DeepStackDockerType.NVIDIA_JETSON:
            return 'deepquestai/deepstack:jetpack'
        elif docker_type == DeepStackDockerType.ARM64:
            return 'deepquestai/deepstack:arm64'
        elif docker_type == DeepStackDockerType.ARM64_SERVER:
            return 'deepquestai/deepstack:arm64-server'
        else:
            return 'deepquestai/deepstack'

    def __init_container(self, all_containers: List):
        for container in all_containers:
            if container.name == self.container_name:
                self.stop_and_remove_container(container)
                logger.warning(f'a previous DeepStack server container has been found and removed.')
                break

        environments = dict()
        if self.ds_config.od_enabled:
            environments['VISION-DETECTION'] = 'True'
        if self.ds_config.fr_enabled:
            environments['VISION-FACE'] = 'True'
        if self.ds_config.performance_mode != DeepStackPerformanceMode.Medium:
            mode: str = 'High' if self.ds_config.performance_mode == DeepStackPerformanceMode.High else 'Low'
            environments['MODE'] = mode

        device_requests = []
        if self.ds_config.docker_type == DeepStackDockerType.GPU:
            device_requests.append(docker.types.DeviceRequest(count=-1, capabilities=[['gpu']]))

        mounts = list()
        mounts.append(Mount(source=f'{path.join(config.general.root_folder_path, "deepstack")}', target='/datastore', type='bind'))

        container = self.client.containers.run(image=self.__get_image_name(), detach=True, restart_policy={'Name': 'unless-stopped'},
                                               name=self.container_name, ports={'5000': str(self.ds_config.server_port)},
                                               environment=environments, mounts=mounts, device_requests=device_requests,
                                               runtime='nvidia' if self.ds_config.docker_type == DeepStackDockerType.NVIDIA_JETSON else '')
        return container

    def run(self) -> Any:
        all_containers = self.get_all_containers()
        container = self.__init_container(all_containers)
        return container

    def remove(self):
        container = self.get_container(self.container_name)
        if container is not None:
            self.stop_and_remove_container(container)

    def get_container(self, container_name: str):
        filters: dict = {'name': container_name}
        containers = self.client.containers.list(filters=filters)
        return containers[0] if len(containers) > 0 else None

    def get_all_containers(self):
        return self.client.containers.list(all=True)

    @staticmethod
    def stop_and_remove_container(container):
        container.stop()
        container.remove()
