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
    parser.add_argument("--train-dir", type=str, dest="train_dir",
                        action="store", required=True,
                        help="Train data directory.")
    parser.add_argument("--val-dir", type=str, dest="val_dir",
                        action="store", required=True,
                        help="Validation data directory.")
    parser.add_argument("--model-h5", type=str, dest="model_h5",
                        action="store", required=True,
                        help="Path to save model weights in h5 file")
    parser.add_argument("--model-json", type=str, dest="model_json",
                        action="store", required=True,
                        help="Path to save model structure in json format")
    parser.add_argument("--epochs", type=int, dest="epochs",
                        action="store", required=False,
                        default=20,
                        help="Epochs number.")
    parser.add_argument("--lr", type=float, dest="lr",
                        action="store", required=False,
                        default=10e-3,
                        help="Learning rate.")
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

    train_dir = args.train_dir
    train_cars_dir = os.path.join(train_dir, 'car')
    train_others_dir = os.path.join(train_dir, 'other')

    validation_dir = args.val_dir
    validation_cars_dir = os.path.join(validation_dir, 'car')
    validation_others_dir = os.path.join(validation_dir, 'other')

    num_cars_tr = get_images_num(train_cars_dir)
    num_others_tr = get_images_num(train_others_dir)

    num_cars_val = get_images_num(validation_cars_dir)
    num_others_val = get_images_num(validation_others_dir)

    total_train = num_cars_tr + num_others_tr
    total_val = num_cars_val + num_others_val

    classes = ['other', 'car']
    img_size = args.img_size
    batch_size = args.img_size
    epochs = args.epochs
    lr = args.lr

    model_builder = models_dict[args.model_name]

    data_gen = model_builder.get_image_data_generator()

    train_generator = data_gen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        classes=classes,
        seed=12345,
        shuffle=True)

    validate_generator = data_gen.flow_from_directory(
        validation_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        classes=classes,
        seed=12345,
        shuffle=True)

    model = model_builder.get_instance(img_size, img_size)
    opt = Adam(lr=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())
    max_accuracy = 0
    for e in range(epochs):
        print(f"Epoch: {e}/{epochs}")
        history = model.fit(
            train_generator,
            steps_per_epoch=total_train // batch_size,
            epochs=1,
            validation_data=validate_generator,
            validation_steps=total_val // batch_size
        )
        if history.history["val_accuracy"][-1] > max_accuracy:
            print("New accuracy", history.history["val_accuracy"][-1])
            max_accuracy = history.history["val_accuracy"][-1]
            model_builder.save_model(model, args.model_json, args.model_h5)
