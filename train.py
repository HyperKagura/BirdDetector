import os
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
import matplotlib.pyplot as plt

LABEL_1 = "birds/"
LABEL_2 = "nonbirds/"
TRAINING_DIR = "data/train/"
TEST_DIR = "data/test/"

def split_data(src_dir, training_dst, test_dst, training_part=0.9):
    imglist = []
    for img in os.listdir(src_dir):
        if os.path.getsize(src_dir + img) == 0:
            print(img, "is of zero size, skipping")
        else:
            imglist.append(img)
    num_img = len(imglist)
    shuffled_list = random.sample(imglist, num_img)
    x = int(num_img * training_part + 0.5)
    train_data = shuffled_list[:x]
    test_data = shuffled_list[x:]
    for img in train_data:
        shutil.copyfile(src_dir + img, training_dst + img)
    for img in test_data:
        shutil.copyfile(src_dir + img, test_dst + img)
    print(len(train_data), " files in ", training_dst)
    print(len(test_data), " files in ", test_dst)

def reshuffle_images():
    if os.path.exists(TRAINING_DIR):
        shutil.rmtree(TRAINING_DIR)
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.mkdir(TRAINING_DIR)
    os.mkdir(TRAINING_DIR + LABEL_1)
    os.mkdir(TRAINING_DIR + LABEL_2)
    os.mkdir(TEST_DIR)
    os.mkdir(TEST_DIR + LABEL_1)
    os.mkdir(TEST_DIR + LABEL_2)
    split_data("data/"+ LABEL_1, TRAINING_DIR + LABEL_1, TEST_DIR + LABEL_1)
    split_data("data/"+ LABEL_2, TRAINING_DIR + LABEL_2, TEST_DIR + LABEL_2)

print("Bird images: ", len(os.listdir('data/birds/')))
print("Non-Bird images: ", len(os.listdir('data/nonbirds/')))

#uncomment the following line to generate training and test data
#reshuffle_images()

model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), input_shape=(128,128,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=10,
                                   horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=128,
                                                    target_size=(128,128),
                                                    class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255,
                                  rotation_range=10,
                                  horizontal_flip=True)

test_generator = test_datagen.flow_from_directory(TEST_DIR,
                                                  batch_size=20,
                                                  target_size=(128,128),
                                                  class_mode='binary')

history = model.fit_generator(train_generator,
                              epochs=5,
                              steps_per_epoch=8,
                              verbose=1,
                              validation_data=test_generator)

acc = history.history['acc']
test_acc = history.history['val_acc']
loss = history.history['loss']
test_loss = history.history['val_loss']

x_epochs = range(len(acc))

plt.plot(x_epochs, acc, 'r', "Training Accuracy")
plt.plot(x_epochs, test_acc, 'b', "Test Accuracy")
plt.title('Training and Test Accuracy')
plt.savefig('data/accuracy.png')
plt.figure()

plt.plot(x_epochs, loss, 'r', "Training Loss")
plt.plot(x_epochs, test_loss, 'b', "Test Loss")
plt.title('Training and Test Loss')
plt.savefig('data/loss.png')

model.save('data/model.h5')
