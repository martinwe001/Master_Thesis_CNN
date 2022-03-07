import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf

dataset_path = "dataset"
images_path = "test_images/*"



def read_image(path):
    x = cv2.imread(str(path), cv2.IMREAD_COLOR)
    # x = x / 255.0
    # x = x.astype(np.float32) # usikker om vi trenger denne
    return x


def load_dataset():

    test_images = sorted(glob(os.path.join(dataset_path, images_path)))
    # masks = sorted(glob(os.path.join(dataset_path, masks_path)))

    # train_x, test_x = train_test_split(images, test_size=0.2, random_state=42)
    # train_y, test_y = train_test_split(masks, test_size=0.2, random_state=42)

    return test_images


def preprocess(image_path):
    def f(image_path):
        image_path = image_path.decode()

        x = read_image(image_path)
        x = x.astype(np.float32)

        return x

    image = tf.numpy_function(f, [image_path], [tf.float32])
    # image.set_shape([256, 256, 3])

    return image


get_patches = lambda x: (
    tf.reshape(
        tf.image.extract_patches(
            images=tf.expand_dims(x, 0),
            sizes=[1, 1024, 1024, 1],
            strides=[1, 1024, 1024, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'), [-1, 1024, 1024, 3]))


def tf_dataset(images):
    dataset = tf.data.Dataset.from_tensor_slices(images)
    # dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(preprocess)
    dataset = dataset.map(get_patches)
    dataset = dataset.prefetch(2)
    return dataset


if __name__ == "__main__":
    test_images = load_dataset()
    train_dataset = tf_dataset(test_images)

