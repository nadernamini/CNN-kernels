import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class Viz_Feat(object):
    def __init__(self, train_data):
        self.train_data = train_data
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def vizualize_features(self, net, image_size):
        """
        Compute the response map for the index images
        """
        conv, sigmoid, max_pool, avg_pool = self.sess.run([net.conv, net.sigmoid, net.max_pool, net.avg_pool],
                                                          feed_dict={net.images: self.train_data[0, 1].reshape(
                                                              (1, image_size[0], image_size[1], 3))})
        print(conv.shape)
        for j in range(3):
            img = self.revert_image(conv[0][0, :, :, j])

            plt.title('img-' + str(i) + '-filter-' + str(j+1))
            plt.imsave('figures/img-' + str(i) + '-filter-' + str(j+1) + '.png', img)

    def revert_image(self, img):
        """
        Used to revert images back to a form that can be easily visualized
        """

        img = (img + 1.0) / 2.0 * 255.0

        img = np.array(img, dtype=int)

        blank_img = np.zeros([img.shape[0], img.shape[1], 3])

        blank_img[:, :, 0] = img
        blank_img[:, :, 1] = img
        blank_img[:, :, 2] = img

        img = blank_img.astype("uint8")

        return img
