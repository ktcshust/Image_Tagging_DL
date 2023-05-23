import tensorflow as tf


def cross_entropy_loss_with_logits(y_true, y_pred):
    loss_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction='none')

    scale = tf.cast(tf.shape(y_true)[-1], dtype=float)
    loss = loss_fn(y_true, y_pred) * scale
    # Return the total.
    return tf.reduce_mean(loss)
