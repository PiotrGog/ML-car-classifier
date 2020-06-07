import abc


class BaseModel(abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def get_instance(self, img_widht, img_height):
        pass

    @abc.abstractmethod
    def get_image_data_generator(self):
        pass
