import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, Adam
from source.utils.get_images_num import get_images_num
from source.utils.history_help import *
import argparse
from source.models import (
    resnet50_pg_tl,
    vgg16_pg_tl,
    simple_pg
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", type=str, dest="model_name",
                        action="store", required=True,
                        choices=["resnet", "vgg16", "simple"],
                        help="Model name.")
    parser.add_argument("--test-dir", type=str, dest="test_dir",
                        action="store", required=True,
                        help="Test data directory.")
    parser.add_argument("--model-h5", type=str, dest="model_h5",
                        action="store", required=True,
                        help="Path to save  model weights in h5 file")
    parser.add_argument("--model-json", type=str, dest="model_json",
                        action="store", required=True,
                        help="Path to save  model structure in json format")
    parser.add_argument("--batch-size", type=int, dest="batch_size",
                        action="store", required=False,
                        default=128,
                        help="Batch size.")
    parser.add_argument("--img-size", type=int, dest="img_size",
                        action="store", required=False,
                        default=128,
                        help="Size of images: img-size x img-size.")

    args = parser.parse_args()

    models_dict = {
        "resnet": vgg16_pg_tl.Vgg16PgTl(),
        "vgg16": resnet50_pg_tl.Resnet50PgTl(),
        "simple": simple_pg.SimplePg()
    }

    test_dir = args.test_dir
    test_cars_dir = os.path.join(test_dir, 'car')
    test_others_dir = os.path.join(test_dir, 'other')

    num_cars_tr = get_images_num(test_cars_dir)
    num_others_tr = get_images_num(test_others_dir)

    total_train = num_cars_tr + num_others_tr

    classes = ['other', 'car']
    img_size = args.img_size
    batch_size = args.img_size

    model_builder = models_dict[args.model_name]

    data_gen = model_builder.get_image_data_generator()

    test_generator = data_gen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        classes=classes,
        seed=12345,
        shuffle=True)

    model = model_builder.load_model(args.model_json, args.model_h5)
    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss='binary_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())

    metrics = model.evaluate(
        test_generator,
        steps=total_train // batch_size
    )
