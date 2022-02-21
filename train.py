import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from model import build_model
from data import load_dataset, tf_dataset
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping


if __name__ == "__main__":
    """ Hyperparamaters """
    dataset_path = "building-segmentation"
    input_shape = (1024, 1024, 3)
    batch_size = 4
    epochs = 5
    lr = 1e-3
    model_path = f"models/damane_model_{epochs}.h5"
    csv_path = f"csv/damane_model_{epochs}.csv"

    """ Load the dataset """
    (train_images, train_masks), (val_images, val_masks) = load_dataset(dataset_path)
    print(f"Train: {len(train_images)} - {len(train_masks)}")
    print(f"Validation: {len(val_images)} - {len(val_masks)}")

    train_dataset = tf_dataset(train_images, train_masks, batch=batch_size)
    val_dataset = tf_dataset(val_images, val_masks, batch=batch_size)


    model = build_model(input_shape)

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(lr),
        metrics=[
            tf.keras.metrics.MeanIoU(num_classes=2),
            tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0]),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Precision()
        ]
    )


    callbacks = [
        ModelCheckpoint(model_path, monitor="val_loss", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", patience=10, factor=0.1, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor="val_loss", patience=10)
    ]

    train_steps = len(train_images)//batch_size
    if len(train_images) % batch_size != 0:
        train_steps += 1

    test_steps = len(val_images)//batch_size
    if len(val_images) % batch_size != 0:
        test_steps += 1

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=test_steps,
        callbacks=callbacks
    )

    # Plot the training and validation loss

    loss = model.history['loss']
    val_loss = model.history['val_loss']

    plt.figure()
    plt.plot(model.epoch, loss, 'r', label='Training loss')
    plt.plot(model.epoch, val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()