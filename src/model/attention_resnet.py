from src.model.block import *


class AttentionResNet(keras.Model):
    def __init__(self, hidden_units, num_classes):
        super(AttentionResNet, self).__init__()

        self.zero_padding = keras.layers.ZeroPadding2D(padding=(3, 3))

        self.conv2d = keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(1, 1), bias_initializer='glorot_uniform')
        self.bn = keras.layers.BatchNormalization()
        self.relu = keras.layers.Activation('relu')
        self.max_pooling = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))

        self.global_avg_pooling = keras.layers.GlobalAveragePooling2D()
        self.dense = keras.layers.Dense(hidden_units, activation='relu')
        self.dropout = keras.layers.Dropout(0.2)
        self.fc = keras.layers.Dense(num_classes)

        self.attn_1 = AttentionBlock(64, 256, 2)
        self.stage_1 = self.__make_stage(filters=(64, 256), num_identity_blocks=2)
        self.attn_2 = AttentionBlock(256, 512, 2)
        self.stage_2 = self.__make_stage(filters=(128, 512), num_identity_blocks=3)
        self.attn_3 = AttentionBlock(512, 1024, 2)
        self.stage_3 = self.__make_stage(filters=(256, 1024), num_identity_blocks=4)
        self.stage_4 = self.__make_stage(filters=(512, 2048), num_identity_blocks=2, use_conv_block=True)

    def __make_stage(self, filters, num_identity_blocks, use_conv_block=False):
        stage = keras.Sequential()
        if use_conv_block:
            stage.add(ConvolutionBlock(filters=filters, s=2))
        for i in range(num_identity_blocks):
            stage.add(IdentityBlock(filters=filters))

        return stage

    def call(self, inputs):
        x = self.zero_padding(inputs)

        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pooling(x)

        x, _ = self.attn_1(x)
        x = self.stage_1(x)
        x, _ = self.attn_2(x)
        x = self.stage_2(x)
        x, _ = self.attn_3(x)
        x = self.stage_3(x)
        x = self.stage_4(x)

        x = self.global_avg_pooling(x)
        x = self.dropout(x)
        x = self.dense(x)
        logits = self.fc(x)

        return logits

    def predict(self, inputs, as_int=False):
        x = self.zero_padding(inputs)

        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pooling(x)

        x, attn_1 = self.attn_1(x)
        x = self.stage_1(x)
        x, attn_2 = self.attn_2(x)
        x = self.stage_2(x)
        x, attn_3 = self.attn_3(x)
        x = self.stage_3(x)
        x = self.stage_4(x)

        x = self.global_avg_pooling(x)
        x = self.dropout(x)
        x = self.dense(x)
        logits = self.fc(x)

        if as_int:
            pred_labels = tf.cast(logits > 0.0, dtype=tf.int32)
        else:
            pred_labels = tf.cast(logits > 0.0, dtype=tf.float32)

        return pred_labels, (attn_1, attn_2, attn_3)
