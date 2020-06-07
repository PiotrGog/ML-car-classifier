from .base_model import BaseModel
from tensorflow.python import keras
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model


class SimplePg(BaseModel):
    def __init__(self):
        super().__init__()

    def get_instance(self, img_width, img_height):
        result_model = keras.Sequential([
            Conv2D(16, 3, padding='same', activation='relu',
                   input_shape=(img_height, img_width, 3)),
            MaxPooling2D(),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(64, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        return result_model

    def get_image_data_generator(self):
        return ImageDataGenerator(rescale=1.0/255)
