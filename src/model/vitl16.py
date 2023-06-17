import tensorflow as tf
from tensorflow import keras
from src.utils import load_pretrained


class PositionalEncoding(keras.layers.Layer):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def call(self, inputs):
        batch_size, seq_length, d_model = tf.shape(inputs)
        position_indices = tf.range(seq_length)[:, tf.newaxis]
        div_term = tf.pow(10000.0, 2 * tf.range(d_model // 2, dtype=tf.float32) / d_model)
        positional_encoding = tf.concat([tf.sin(position_indices / div_term), tf.cos(position_indices / div_term)], axis=-1)
        positional_encoding = tf.reshape(positional_encoding, [1, seq_length, d_model])

        return inputs + positional_encoding


class TransformerLayer(keras.layers.Layer):
    def __init__(self, d_model, num_heads, mlp_dim, dropout_rate):
        super(TransformerLayer, self).__init__()
        self.multihead_attention = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.layer_norm1 = keras.layers.LayerNormalization()
        self.mlp = keras.Sequential([
            keras.layers.Dense(mlp_dim, activation='relu'),
            keras.layers.Dense(d_model),
        ])
        self.dropout2 = keras.layers.Dropout(dropout_rate)
        self.layer_norm2 = keras.layers.LayerNormalization()

    def call(self, inputs):
        attention_output = self.multihead_attention(inputs, inputs)
        attention_output = self.dropout1(attention_output)
        attention_output = self.layer_norm1(inputs + attention_output)
        mlp_output = self.mlp(attention_output)
        mlp_output = self.dropout2(mlp_output)
        output = self.layer_norm2(attention_output + mlp_output)

        return output


class ViTL16(keras.Model):
    def __init__(self, hidden_units, num_classes, patch_size=16, num_layers=16, d_model=256, num_heads=8, mlp_dim=512, dropout_rate=0.1):
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
        patches = tf.squeeze(patches, axis=1)
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
        pred = tf.math.argmax(logits, axis=-1)
        if as_int:
            return tf.cast(pred, dtype=tf.int32)
        return tf.cast(pred, dtype=tf.float32)

