import numpy as np
from PIL import Image
import tensorflow as tf


def clean_image(image):
    image = np.array(image)
    image = np.array(Image.fromarray(
        image).resize((512, 512), Image.ANTIALIAS))
    image = image[np.newaxis, :, :, :3]
    return image


def get_prediction(model, image):

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)
    test = datagen.flow(image)
    predictions = model.predict(test)
    predictions_arr = np.array(np.argmax(predictions))

    return predictions, predictions_arr


def make_results(predictions, predictions_arr):

    result = {}
    if int(predictions_arr) == 0:
        result = {"status": " is Healthy ",
                  "prediction": f"{int(predictions[0][0].round(2)*100)}%"}
    if int(predictions_arr) == 1:
        result = {"status": ' has Multiple Diseases ',
                  "prediction": f"{int(predictions[0][1].round(2)*100)}%"}
    if int(predictions_arr) == 2:
        result = {"status": ' has Rust ',
                  "prediction": f"{int(predictions[0][2].round(2)*100)}%"}
    if int(predictions_arr) == 3:
        result = {"status": ' has Scab ',
                  "prediction": f"{int(predictions[0][3].round(2)*100)}%"}
    return result
