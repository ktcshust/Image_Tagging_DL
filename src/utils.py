import tensorflow as tf
import matplotlib.pyplot as plt


def load_pretrained():
    resnet = tf.keras.applications.resnet_v2.ResNet50V2(include_top=True)
    pretrained_model = tf.keras.Model(inputs=resnet.input, outputs=resnet.layers[-3].output)

    return pretrained_model


def plot_history(history):
    plt.savefig("")
    pass
