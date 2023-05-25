from tensorflow import keras


class IdentityBlock(keras.layers.Layer):
    def __init__(self, filters):
        super(IdentityBlock, self).__init__()

        f1, f2 = filters

        self.conv2d_a = keras.layers.Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid')
        self.bn_a = keras.layers.BatchNormalization()

        self.conv2d_b = keras.layers.Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn_b = keras.layers.BatchNormalization()

        self.conv2d_c = keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid')
        self.bn_c = keras.layers.BatchNormalization()

        self.add = keras.layers.Add()
        self.relu = keras.layers.Activation('relu')

    def call(self, inputs):
        x_shortcut = inputs
        x = inputs

        x = self.conv2d_a(x)
        x = self.bn_a(x)
        x = self.relu(x)

        x = self.conv2d_b(x)
        x = self.bn_b(x)
        x = self.relu(x)

        x = self.conv2d_c(x)
        x = self.bn_c(x)

        x = self.add([x, x_shortcut])
        x = self.relu(x)

        return x
