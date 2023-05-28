import os
import pandas as pd
import tensorflow as tf
import numpy as np


def decode(image):
    img = tf.io.decode_jpeg(image, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = img / 255.0
    return img


def convert_path(image_paths, labels):
    images = tf.map_fn(fn=tf.io.read_file, elems=image_paths)
    images = tf.map_fn(fn=decode, elems=images, fn_output_signature=tf.float32)
    return images, tf.cast(labels, tf.float32)


def get_data_from_csv(path):
    """
    :return:
        data: list[tuple[str, list[int]]],
    """
    df = pd.read_csv(path)
    data = []
    for _, row in df.iterrows():
        data.append((row['id'].split(os.path.sep)[-1], row['HG':'CVN']))

    return data


def extract_path_and_label(image_dir, image_path_with_label):
    paths = list(map(lambda x: image_dir + x[0], image_path_with_label))
    labels = list(map(lambda x: x[1], image_path_with_label))

    return paths, labels


def get_dataset(image_paths, labels, ratio=0.8, test=False):
    buffer_size = len(image_paths)
    batch_size = 64
    if test:
        test_raw = (tf.data.Dataset
                    .from_tensor_slices((image_paths, labels))
                    .shuffle(buffer_size)
                    .batch(batch_size))
        return test_raw

    is_train = np.random.uniform(size=buffer_size) < ratio

    train_image_paths = []
    train_labels = []
    val_image_paths = []
    val_labels = []

    for i in range(buffer_size):
        if is_train[i]:
            train_image_paths.append(image_paths[i])
            train_labels.append(labels[i])
        else:
            val_image_paths.append(image_paths[i])
            val_labels.append(labels[i])

    train_raw = (tf.data.Dataset
                 .from_tensor_slices((train_image_paths, train_labels))
                 .shuffle(buffer_size)
                 .batch(batch_size))
    val_raw = (tf.data.Dataset
               .from_tensor_slices((val_image_paths, val_labels))
               .shuffle(buffer_size)
               .batch(batch_size))

    return train_raw, val_raw
