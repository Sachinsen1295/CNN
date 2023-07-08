from keras.models import load_model
import numpy as np
from keras.utils import load_img,img_to_array
from keras.preprocessing import image

class dogcat:
    def __init__(self,filename) -> None:
        self.filename = filename


    def predictdogcat(self):

        model = load_model('model.h5')

        imgname =  self.filename
        test_img = load_img(imgname, target_size = (64,64))
        test_img = img_to_array(test_img)
        test_img = np.expand_dims(test_img, axis=0)

        result = model.predict(test_img)
        # training_set.class_indices

        if result[0][0] == 1:
            prediction = 'dog'
            return [{ "image" : prediction}]
        else:
            prediction = 'cat'
            return [{ "image" : prediction}]
        