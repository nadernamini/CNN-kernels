from data_manager import DataManager
from cnn import CNN
import random
import numpy as np
from viz_features import Viz_Feat

image_size = [128, 77]

random.seed(0)

dm = DataManager(image_size)

k = np.zeros([3, 3, 1, 1])
k[1, 1, :, :] = 8
k[0, 1, :, :] = -1
k[1, 0, :, :] = -1
k[2, 1, :, :] = -1
k[1, 2, :, :] = -1

cnn = CNN(image_size, kernel=k)

cm = Viz_Feat(dm.train_data)

cm.vizualize_features(cnn, image_size)

# Sharpen
k = np.zeros([3, 3, 1, 1])
k[1, 1, :, :] = 5
k[0, 1, :, :] = -1
k[1, 0, :, :] = -1
k[2, 1, :, :] = -1
k[1, 2, :, :] = -1

cnn = CNN(image_size, kernel=k)

cm = Viz_Feat(dm.train_data)

cm.vizualize_features(cnn, image_size)

# Edge Detection
k = np.zeros([3, 3, 1, 1])
k[1, 1, :, :] = 8
k[0, 1, :, :] = -1
k[0, 2, :, :] = -1
k[1, 0, :, :] = -1
k[2, 1, :, :] = -1
k[1, 2, :, :] = -1
k[2, 2, :, :] = -1
k[2, 0, :, :] = -1
k[0, 0, :, :] = -1

cnn = CNN(image_size, kernel=k)

cm = Viz_Feat(dm.train_data)

cm.vizualize_features(cnn, image_size)

# Gaussian Blur
k = np.zeros([3, 3, 1, 1])
k[1, 1, :, :] = 4
k[0, 1, :, :] = 2
k[0, 2, :, :] = 1
k[1, 0, :, :] = 2
k[2, 1, :, :] = 2
k[1, 2, :, :] = 2
k[2, 2, :, :] = 1
k[0, 0, :, :] = 1
k[2, 0, :, :] = 1
k = k / 16


cnn = CNN(image_size, kernel=k)

cm = Viz_Feat(dm.train_data)

cm.vizualize_features(cnn, image_size)
