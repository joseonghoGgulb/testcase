import cv2
import numpy as np
import os
from keras.models import load_model
from keras import backend as K

model_path = './model/CNN_dep4_model_best_weights_15.hdf5'
img_path = './uploads/img/'
names = 'id.png'


def detect_covid(img_path, names):
    # Use the index of the best obtained model according to the test results
    model = load_model(model_path)
    imgArraySize = (88, 39)
    # Loading Images
    images = []

    img = cv2.imread(os.path.join(img_path, names))
    img = cv2.resize(img, imgArraySize)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, dtype=np.float32)
    img = img/255.0
    if img is not None:
        images.append(img)

    images = np.squeeze(images)

    # reshape img
    rows = imgArraySize[1]
    cols = imgArraySize[0]

    if K.image_data_format() == 'channels_first':
        images = images.reshape(1, 3, rows, cols)
        input_shape = (3, rows, cols)
    else:
        images = images.reshape(1, rows, cols, 3)
        input_shape = (rows, cols, 3)

    covPredict = model.predict(images)

    value = covPredict[0][0]
    decimal_places = 6

    rounded_value = round(value * 10**decimal_places) / 10**decimal_places
    formatted_value = "{:.{}f}".format(rounded_value, decimal_places)

    return rounded_value


if __name__ == '__main__':

    print(detect_covid(img_path=img_path, names=names))
