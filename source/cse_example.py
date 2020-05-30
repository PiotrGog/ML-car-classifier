from matplotlib import pyplot as plt
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import os
data_gen = ImageDataGenerator(rescale=1.0/255)

imgdir = './resources/pa3_images'  # or wherever you put them...
img_size = 64
batch_size = 32

train_generator = data_gen.flow_from_directory(
    imgdir + '/train',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    classes=['other', 'car'],
    seed=12345,
    shuffle=True)

validate_generator = data_gen.flow_from_directory(
    imgdir + '/validation',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    classes=['other', 'car'],
    seed=12345,
    shuffle=True)

Xbatch, Ybatch = train_generator.next()
print(Xbatch.shape)
print(Ybatch[4])
plt.imshow(Xbatch[4])
plt.show()


def make_convnet(img_height, img_width, channels):
    model = keras.Sequential([
        Conv2D(16, 3, padding='same', activation='relu',
               input_shape=(img_height, img_width, channels)),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1)
    ])
    return model


model = make_convnet(img_size, img_size, 3)
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

print(model.summary())

epochs = 10

train_dir = os.path.join(imgdir, 'train')
validation_dir = os.path.join(imgdir, 'validation')

# directory with our training cat pictures
train_cars_dir = os.path.join(train_dir, 'car')
# directory with our training dog pictures
train_others_dir = os.path.join(train_dir, 'other')
# directory with our validation cat pictures
validation_cars_dir = os.path.join(validation_dir, 'car')
validation_others_dir = os.path.join(
    validation_dir, 'other')  # directory with o

num_cars_tr = len(os.listdir(train_cars_dir))
num_others_tr = len(os.listdir(train_others_dir))

num_cars_val = len(os.listdir(validation_cars_dir))
num_others_val = len(os.listdir(validation_others_dir))

total_train = num_cars_tr + num_others_tr
total_val = num_cars_val + num_others_val

history = model.fit_generator(
    train_generator,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=validate_generator,
    validation_steps=total_val // batch_size
)
