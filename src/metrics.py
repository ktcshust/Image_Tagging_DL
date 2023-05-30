import tensorflow as tf


def accuracy_score(y_true, y_pred):
    # remember: y_pred is logits
    y_true = tf.cast(y_true, dtype=tf.bool)
    y_pred = tf.cast(y_pred > 0, dtype=tf.bool)
    and_match = tf.cast(tf.math.logical_and(y_true, y_pred), dtype=tf.float32)
    or_match = tf.cast(tf.math.logical_or(y_true, y_pred), dtype=tf.float32)
    return tf.reduce_mean(tf.reduce_sum(and_match, axis=1) / tf.reduce_sum(or_match, axis=1))


def precision_score(y_true, y_pred):
    # remember: y_pred is logits
    y_true = tf.cast(y_true, dtype=tf.bool)
    y_pred = tf.cast(y_pred > 0, dtype=tf.bool)

    and_match = tf.math.logical_and(y_true, y_pred)

    if not tf.reduce_all(tf.reduce_any(and_match, axis=1)):
        return tf.constant(0.0)

    and_match = tf.cast(and_match, dtype=tf.float32)
    precisions = tf.reduce_sum(and_match, axis=1) / tf.reduce_sum(tf.cast(y_pred, dtype=tf.float32), axis=1)
    return tf.reduce_mean(precisions)


def recall_score(y_true, y_pred):
    # remember: y_pred is logits
    y_true = tf.cast(y_true, dtype=tf.bool)
    y_pred = tf.cast(y_pred > 0, dtype=tf.bool)

    and_match = tf.cast(tf.math.logical_and(y_true, y_pred), dtype=tf.float32)
    recalls = tf.reduce_sum(and_match, axis=1) / tf.reduce_sum(tf.cast(y_true, dtype=tf.float32), axis=1)
    return tf.reduce_mean(recalls)


def f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.bool)
    y_pred = tf.cast(y_pred > 0, dtype=tf.bool)

    and_match = tf.math.logical_and(y_true, y_pred)

    and_match = tf.cast(and_match, dtype=tf.float32)
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    f1 = 2 * tf.reduce_sum(and_match, axis=1) / tf.reduce_sum(y_true + y_pred, axis=1)
    return tf.reduce_mean(f1)
