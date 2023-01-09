import datetime
from dataclasses import dataclass
from typing import Iterator
import os


@dataclass
class FileManager:
    root_image_path: str = 'StableDiffusion-utils/images/to_improve'
    destination_image_path: str = f"new_image_at_{datetime.datetime.today()}"

    @classmethod
    def get_images_iterator(cls) -> Iterator[str]:
        for image in cls.root_image_path:
            yield os.path.join(cls.root_image_path, image)

    @classmethod
    def get_destination_path(cls, image_name: str) -> str:
        return image_name.replace(cls.root_image_path, cls.destination_image_path)
    
