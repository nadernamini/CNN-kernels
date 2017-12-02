import numpy as np
from numpy.random import randint
import cv2

import glob


class DataManager(object):
    def __init__(self, image_size, compute_features=None, num_filters=3):
        self.image_size = image_size
        self.batch_size = 1

        self.recent_batch = []

        if not compute_features:
            self.compute_feature = self.compute_features_baseline
        else:
            self.compute_feature = compute_features

        self.load_train_set()

        self.train_data = np.asarray([[d['c_img'], d['features']] for d in self.train_data])

    def get_train_batch(self):
        """
        Compute a training batch for the neural network 
        The batch size should be size 40
        """
        chc = randint(self.train_data.shape[0], size=self.batch_size)

        return self.train_data[chc, 2].tolist(), self.train_data[chc, 1].tolist()

    def compute_features_baseline(self, image):
        """
        computes the featurized on the images. In this case this corresponds
        to rescaling and standardizing.
        """

        image = cv2.resize(image, (self.image_size[0], self.image_size[1]))
        image = (image / 255.0) * 2.0 - 1.0

        return image

    def load_set(self, set_name):
        """
        Given a string which is either 'val' or 'train', the function should load all the
        data into an 

        """

        data = []
        data_paths = glob.glob(set_name + '/train3.png')

        for datum_path in data_paths:
            img = cv2.imread(datum_path)

            features = self.compute_feature(img)

            data.append({'c_img': img, 'features': features})

        np.random.shuffle(data)
        return data

    def load_train_set(self):
        """
        Loads the train set
        """
        self.train_data = self.load_set('train')
