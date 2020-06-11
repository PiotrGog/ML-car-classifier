import abc
import os
from tensorflow.python.keras.models import model_from_json


class BaseModel(abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def get_instance(self, img_widht, img_height):
        pass

    @abc.abstractmethod
    def get_image_data_generator(self):
        pass

    def save_model(self, model, model_json_path, model_h5_path):
        model_json = model.to_json()
        with open(model_json_path, "w") as json_file:
            json_file.write(model_json)
        model.save_weights(model_h5_path)

    def load_model(self, model_json_path, model_h5_path):
        json_file = open(model_json_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(model_h5_path)
        return loaded_model
