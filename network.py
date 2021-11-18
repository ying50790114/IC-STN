from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
import tensorflow as tf

class IC_STN(tf.keras.Model):
    def __init__(self, warpDim):    # warpDim = opt.warpDim
        super(IC_STN, self).__init__()
        # GP
        self.conv1 = Conv2D(4,
                            kernel_size=(7, 7),
                            strides=(1, 1),
                            padding='valid',
                            activation='relu',
                            name='conv1',
                            trainable=True)
        self.conv2 = Conv2D(8,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='valid',
                            activation='relu',
                            name='conv2',
                            trainable=True)
        self.pool = MaxPool2D(pool_size=(2, 2),
                              strides=(1, 1),
                              padding='valid',
                              name='pool')
        self.flat = Flatten()

        self.fc0 = Dense(48, activation='relu', name='fc0', trainable=True)

        self.fc1 = Dense(warpDim, name='fc1', trainable=True)

    def call(self, inputs, training=True, **kwargs):
        z = self.conv1(inputs)
        z = self.conv2(z)
        z = self.pool(z)
        z = self.flat(z)
        z = self.fc0(z)
        z = self.fc1(z)
        return z

class CNN(tf.keras.Model):
    def __init__(self, labelN):   # labelN = opt.labelN
        super(CNN, self).__init__()
        self.conv = Conv2D(3,
                            kernel_size=(9, 9),
                            strides=(1, 1),
                            padding='valid',
                            activation='relu',
                            name='conv',
                            trainable=True)

        self.flat = Flatten()

        self.fc = Dense(labelN,
                        name='fc',
                        activation='softmax',
                        trainable=True)

    def call(self, inputs, training=True, **kwargs):
        z = self.conv(inputs)
        z = self.flat(z)
        z = self.fc(z)
        return z

