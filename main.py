import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, Adam
from source.utils.get_images_num import get_images_num
from source.utils.history_help import *

from source.models import (
    resnet50_pg_tl,
    vgg16_pg_tl,
    simple_pg
)

imgdir = './resources/pa3_images'

train_dir = os.path.join(imgdir, 'train_augmentation')
train_cars_dir = os.path.join(train_dir, 'car')
train_others_dir = os.path.join(train_dir, 'other')

validation_dir = os.path.join(imgdir, 'validation')
validation_cars_dir = os.path.join(validation_dir, 'car')
validation_others_dir = os.path.join(validation_dir, 'other')

num_cars_tr = get_images_num(train_cars_dir)
num_others_tr = get_images_num(train_others_dir)

num_cars_val = get_images_num(validation_cars_dir)
num_others_val = get_images_num(validation_others_dir)

total_train = num_cars_tr + num_others_tr
total_val = num_cars_val + num_others_val

classes = ['other', 'car']
img_size = 64
batch_size = 128
epochs = 20

# model_builder = vgg16_pg_tl.Vgg16PgTl()
# model_builder = resnet50_pg_tl.Resnet50PgTl()
model_builder = simple_pg.SimplePg()

vgg_data_gen = model_builder.get_image_data_generator()

train_generator = vgg_data_gen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    classes=classes,
    seed=12345,
    shuffle=True)

validate_generator = vgg_data_gen.flow_from_directory(
    validation_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    classes=classes,
    seed=12345,
    shuffle=True)

lr = 0.0001

model = model_builder.get_instance(img_size, img_size)
opt = Adam(lr=lr)
model.compile(optimizer=opt, loss='binary_crossentropy',
              metrics=['accuracy'])

print(model.summary())

history = model.fit(
    train_generator,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=validate_generator,
    validation_steps=total_val // batch_size
)

history_name = f"./simple_pg{img_size}_{batch_size}_{epochs}_{lr}.hist"
plot_name = f"./simple_pg{img_size}_{batch_size}_{epochs}_{lr}.png"

save_history(history, history_name)
h = load_history(history_name)
draw_history_plots(h, plot_name)
