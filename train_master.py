import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# from master_model import build_model
from model import build_model # U-NET
# from load_data import load_dataset, tf_dataset
from load_patches import load_dataset, tf_dataset
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping


if __name__ == "__main__":
    """ Hyperparamaters """

    batch_size = 1
    epochs = 99
    lr = 1e-4
    # lr_split = str(lr).split('-')[1]
    model_path = f"models/master_model_{epochs}_{batch_size}_{lr}.h5"
    csv_path = f"csv/master_model_{epochs}_{batch_size}_{lr}.csv"

    """ Load the dataset """

    (train_images, train_masks), (val_images, val_masks) = load_dataset()

    print(f"Train: {len(train_images)} - {len(train_masks)}")
    print(f"Validation: {len(val_images)} - {len(val_masks)}")

    train_dataset = tf_dataset(train_images, train_masks, batch=batch_size)
    val_dataset = tf_dataset(val_images, val_masks, batch=batch_size)

    model = build_model()

    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr),
        metrics=[
            tf.keras.metrics.BinaryIoU(target_class_id=[0, 1], threshold=0.3),
            tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0]),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Precision()
        ]
    )

    callbacks = [
        ModelCheckpoint(model_path, monitor="val_loss", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.1, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor="val_loss", patience=10)
    ]

    train_steps = len(train_images)//batch_size
    if len(train_images) % batch_size != 0:
        train_steps += 1

    val_steps = len(val_images)//batch_size
    if len(val_images) % batch_size != 0:
        val_steps += 1

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=callbacks
    )

    # Plot the training and validation loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure()
    plt.plot(history.epoch, loss, 'r', label='Training loss')
    plt.plot(history.epoch, val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig(f"loss_plot/loss_graph_{epochs}_{batch_size}_{lr}.png")
