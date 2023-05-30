import tensorflow as tf
from tensorflow import keras
from src.model.block import IdentityBlock, AttentionBlockV2, ConvolutionBlock


def _make_stage(filters, num_identity_blocks, use_conv_block=False):
    stage = keras.Sequential()
    if use_conv_block:
        stage.add(ConvolutionBlock(filters=filters, s=2))
    else:
        stage.add(AttentionBlockV2(filters, s=2))
    for i in range(num_identity_blocks):
        stage.add(IdentityBlock(filters=filters))

    return stage


class ACNNv2(keras.Model):
    def __init__(self, hidden_units, num_classes):
        super(ACNNv2, self).__init__()

        self.zero_padding = keras.layers.ZeroPadding2D(padding=(3, 3))

        self.conv2d = keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(1, 1), bias_initializer='glorot_uniform')
        self.bn = keras.layers.BatchNormalization()
        self.relu = keras.layers.Activation('relu')
        self.max_pooling = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))

        self.stage_1 = _make_stage(filters=(64, 256), num_identity_blocks=2)
        self.stage_2 = _make_stage(filters=(128, 512), num_identity_blocks=2)
        self.stage_3 = _make_stage(filters=(256, 1024), num_identity_blocks=3)
        self.stage_4 = _make_stage(filters=(512, 2048), num_identity_blocks=2, use_conv_block=True)

        self.global_avg_pooling = keras.layers.GlobalAveragePooling2D()
        self.dropout = keras.layers.Dropout(0.3)
        self.dense = keras.layers.Dense(hidden_units, activation='relu')
        self.fc = keras.layers.Dense(num_classes)

    def call(self, inputs):
        x = self.zero_padding(inputs)
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.global_avg_pooling(x)
        x = self.dropout(x)
        x = self.dense(x)
        logits = self.fc(x)

        return logits

    def predict(self, inputs, as_int=False):
        logits = self.call(inputs)

        if as_int:
            y_pred = tf.cast(logits > 0.0, dtype=tf.int32)
        else:
            y_pred = tf.cast(logits > 0.0, dtype=tf.float32)

        return y_pred
