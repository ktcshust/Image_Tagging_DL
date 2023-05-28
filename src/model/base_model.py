import tensorflow as tf
from tensorflow import keras
from src.utils import load_pretrained


class BaseResNet50V2(keras.Model):
    def __init__(self,hidden_units, num_classes):
        super(BaseResNet50V2, self).__init__()
        # pretrained model for feature extraction
        self.pretrained = load_pretrained()
        # fix pretrained model's weights
        self.preprocess_model.trainable = False
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(hidden_units, activation='relu')
        self.output_fc = keras.layers.Dense(num_classes)

    def call(self, batch_input):
        feature_maps = self.preprocess_model(batch_input)
        flatten_vectors = self.flatten(feature_maps)
        logits = self.dense(flatten_vectors)
        logits = self.output_fc(logits)

        return logits

    def predict(self, batch_input, as_int=True):
        batch_shape = tf.shape(batch_input)
        if len(batch_shape) == 3:
            batch_input = batch_input[tf.newaxis]

        logits = self(batch_input)
        pred = logits > 0.0
        if as_int:
            return tf.cast(pred, dtype=tf.int32)
        return tf.cast(pred, dtype=tf.float32)
