from src.data import *
from src.utils import *
from src.model.base_model import *
from src.losses import *
from src.metrics import *
import yaml

if __name__ == "__main__":
    with open("config.yml") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    label_name, data = get_data_from_csv(config['csv_file'])
    image_dir = config['image_dir']
    image_paths, labels = extract_path_and_label(image_dir, data)

    NUM_CLASSES = len(label_name)

    train_raw, val_raw = get_dataset(image_paths, labels, ratio=config['train_test_ratio'])
    train_ds = train_raw.map(convert_path, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_raw.map(convert_path, num_parallel_calls=tf.data.AUTOTUNE)

    pretrained = load_pretrained()

    hidden_units = config['hidden_units']

    model = BaseMultiLabelCNN(pretrained, hidden_units, NUM_CLASSES)

    model.compile(optimizer='adam',
                  loss=cross_entropy_loss_with_logits, metrics=[accuracy])

    print("Call model on batch of data for building graph:")
    eval_result = model.evaluate(val_ds, steps=5, return_dict=True)
    print(eval_result)

    print("Training: ")
    history = model.fit(
        train_ds,
        epochs=config['epochs'],
        validation_data=val_ds,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)])

    model.save_weights('tmp/base_model.tf')

    plot_history(history.history)

    # Evaluate model
