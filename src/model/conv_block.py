from tensorflow import keras


class ConvolutionalBlock(keras.layers.Layer):
    def __init__(self, filters, s):
        super(ConvolutionalBlock, self).__init__()

        f1, f2 = filters

        self.conv2d_1a = keras.layers.Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid')
        self.bn_1a = keras.layers.BatchNormalization()

        self.conv2d_1b = keras.layers.Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn_1b = keras.layers.BatchNormalization()

        self.conv2d_1c = keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid')
        self.bn_1c = keras.layers.BatchNormalization()

        self.conv2d_2 = keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid')
        self.bn_2 = keras.layers.BatchNormalization()

        self.add = keras.layers.Add()
        self.relu = keras.layers.Activation('relu')

    def call(self, inputs):
        x_shortcut = inputs
        x = inputs

        x = self.conv2d_1a(x)
        x = self.bn_1a(x)
        x = self.relu(x)

        x = self.conv2d_1b(x)
        x = self.bn_1b(x)
        x = self.relu(x)

        x = self.conv2d_1c(x)
        x = self.bn_1c(x)

        x_shortcut = self.conv2d_2(x_shortcut)
        x_shortcut = self.bn_2(x_shortcut)

        x = self.add([x, x_shortcut])
        x = self.relu(x)

        return x
