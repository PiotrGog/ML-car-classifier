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
        "vgg16": vgg16_pg_tl.Vgg16PgTl(),
        "resnet": resnet50_pg_tl.Resnet50PgTl(),
        "simple": simple_pg.SimplePg()
    }

    test_dir = args.test_dir

    img_size = args.img_size
    batch_size = args.img_size

    model_builder = models_dict[args.model_name]

    classes = ['other', 'car']
    data_gen = model_builder.get_image_data_generator()
    print("test_dir", os.listdir(test_dir))
    os.listdir(test_dir)
    test_generator = data_gen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=1,
        class_mode=None,
        shuffle=False)

    model = model_builder.load_model(args.model_json, args.model_h5)
    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss='binary_crossentropy',
                  metrics=['accuracy'])
    import numpy as np
    filenames = test_generator.filenames
    nb_samples = len(filenames)
    predict = model.predict(
        test_generator
    )
    predict = model.predict(
        test_generator
    )
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for p, f in zip(predict, filenames):
        if "car" in os.path.split(f)[0].replace("/", ""):
            if(p[0] > 0.5):
                tp += 1
            else:
                fn += 1
        else:
            if(p[0] > 0.5):
                fp += 1
            else:
                tn += 1

        if p[0] > 0.5:
            print(f)
    print("Accuracy: ", (tp+tn)/len(filenames))
