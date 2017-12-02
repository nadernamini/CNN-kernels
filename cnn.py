import tensorflow as tf

slim = tf.contrib.slim


class CNN(object):
    def __init__(self, image_size, kernel, num_filters=3):
        """
        Initializes the size of the network
        """

        self.image_size = image_size
        self.conv_layer = None

        self.images = tf.placeholder(tf.float32, [1, None, None, num_filters])
        print(self.images.shape)
        self.build_network(self.images, kernel)

        # self.labels = tf.placeholder(tf.float32, [None, self.output_size])
        #
        # self.loss_layer(self.logits, self.labels)
        # self.total_loss = tf.losses.get_total_loss()
        # tf.summary.scalar('total_loss', self.total_loss)

    def build_network(self,
                      images,
                      kernel,
                      scope='old',
                      padding='SAME'):
        with tf.variable_scope(scope):
            w = tf.get_variable('w', initializer=tf.to_float(kernel))
            self.conv = tf.nn.conv2d(images, w, strides=[1, 3, 3, 1], padding=padding)
            self.sigmoid = tf.sigmoid(self.conv)
            self.max_pool = tf.nn.max_pool(self.sigmoid, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding=padding)
            self.avg_pool = tf.nn.avg_pool(self.sigmoid, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding=padding)

