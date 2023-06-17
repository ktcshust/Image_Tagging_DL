import tensorflow as tf
from tensorflow import keras
from src.utils import load_pretrained


class ViTL16(keras.Model):
    def __init__(self, hidden_units, num_classes, patch_size, num_layers, d_model, num_heads, mlp_dim, dropout_rate):
        super(ViTL16, self).__init__()
        # Pretrained model for feature extraction
        self.pretrained = load_pretrained()
        # Fix pretrained model's weights
        self.pretrained.trainable = False
        self.patch_embedding = keras.layers.Conv2D(d_model, kernel_size=patch_size, strides=patch_size, padding='valid')
        self.positional_encoding = PositionalEncoding()
        self.transformer_layers = [TransformerLayer(d_model, num_heads, mlp_dim, dropout_rate) for _ in range(num_layers)]
        self.global_avg_pool = keras.layers.GlobalAveragePooling1D()
        self.dense = keras.layers.Dense(hidden_units, activation='relu')
        self.output_fc = keras.layers.Dense(num_classes)

    def call(self, batch_input):
        feature_maps = self.pretrained(batch_input)
        patches = self.patch_embedding(feature_maps)
        batch_size, num_patches, _, d_model = patches.shape
        patches = tf.reshape(patches, [batch_size, num_patches, d_model])
        patches = self.positional_encoding(patches)
        for layer in self.transformer_layers:
            patches = layer(patches)
        flatten_vectors = self.global_avg_pool(patches)
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
