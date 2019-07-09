import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import time
import os

model = load_model('data/model.h5')

test_file = 'data/nonbirds_orig/IMG_0365.jpg'

def test_one(filename):
    img = image.load_img(filename, target_size=(128,128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    start_time = time.time()
    res = model.predict(images)
    end_time = time.time()
    if res[0][0] > 0:
        print(filename + " is a bird")
    else:
        print(filename + " is not a bird")
    print("Prediction time: ", end_time - start_time)

test_one(test_file)

def test_dir(dirpath):
    imgs = os.listdir(dirpath)
    for img in imgs:
        if len(img) > 4 and img[-4:].lower() == '.jpg':
            test_one(dirpath + img)

test_dir('data/only_test/birds/')