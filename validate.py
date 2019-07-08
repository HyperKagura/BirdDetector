import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import time

model = load_model('data/model.h5')

img = image.load_img('data/nonbirds_orig/IMG_0365.jpg', target_size=(128,128))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
start_time = time.time()
res = model.predict(images)
end_time = time.time()
if res[0][0] > 0:
    print("it's a bird")
else:
    print("it's not a bird")
print("Prediction time: ", end_time - start_time)