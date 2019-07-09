import os
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import shutil
import matplotlib.pyplot as plt

TRAINING_DIR = "data/train/"
TEST_DIR = "data/test/"

model = load_model('data/model.h5')

#model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])

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

plt.plot(x_epochs, acc, 'r', label="Training Accuracy")
plt.plot(x_epochs, test_acc, 'b', label="Test Accuracy")
plt.title('Training and Test Accuracy')
plt.legend(loc=0)
plt.savefig('data/accuracy.png')
plt.figure()

plt.plot(x_epochs, loss, 'r', label="Training Loss")
plt.plot(x_epochs, test_loss, 'b', label="Test Loss")
plt.title('Training and Test Loss')
plt.legend(loc=0)
plt.savefig('data/loss.png')

model.save('data/model.h5')
