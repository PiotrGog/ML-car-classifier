from .base_model import BaseModel
from tensorflow.python.keras import applications
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model


class Vgg16PgTl(BaseModel):
    def __init__(self):
        super().__init__()

    def get_instance(self, img_widht, img_height):
        feature_extractor = applications.vgg16.VGG16(include_top=False, weights='imagenet',
                                                     input_shape=(img_widht, img_height, 3))

        for layer in feature_extractor.layers:
            layer.trainable = False

        dropout = 0.3
        x = feature_extractor.output
        x = Flatten()(x)
        x = Dropout(dropout)(x)
        x = Dense(32, activation='sigmoid')(x)
        output = Dense(1, activation='sigmoid')(x)

        result_model = Model(inputs=feature_extractor.input, outputs=output)

        return result_model

    def get_image_data_generator(self):
        return ImageDataGenerator(
            preprocessing_function=applications.vgg16.preprocess_input)
