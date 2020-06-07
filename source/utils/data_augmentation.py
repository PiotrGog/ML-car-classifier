import os
import math
import numpy as np
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array


def data_augmentation(in_dir, out_dir, result_images):
    images_list = [
        os.path.join(in_dir, f) for f in os.listdir(in_dir)
    ]
    num_images_from_one = math.ceil(result_images/len(images_list))
    aug = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")

    for (image_num, image_file) in enumerate(images_list):
        image = load_img(image_file)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        imageGen = aug.flow(image, batch_size=1, save_to_dir=out_dir,
                            save_prefix="image"+str(image_num), save_format="jpg")
        total = 0
        for image in imageGen:
            total += 1
            if total == num_images_from_one:
                break
