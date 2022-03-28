import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf

dataset_path = "dataset"
images_path = "images/*"
masks_path = "masks/*"


def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = x.astype(np.float32) # usikker om vi trenger denne
    return x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = x > 0.5
    x = x.astype(np.float32) # usikker om vi trenger denne
    x = np.expand_dims(x, axis=-1)
    return x


def load_dataset():

    images = sorted(glob(os.path.join(dataset_path, images_path)))
    masks = sorted(glob(os.path.join(dataset_path, masks_path)))

    train_x, test_x = train_test_split(images, test_size=0.2, random_state=42)
    train_y, test_y = train_test_split(masks, test_size=0.2, random_state=42)


    return (train_x, train_y), (test_x, test_y)


def preprocess(image_path, mask_path):
    def f(image_path, mask_path):
        image_path = image_path.decode()
        mask_path = mask_path.decode()

        x = read_image(image_path)
        y = read_mask(mask_path)

        return x, y

    image, mask = tf.numpy_function(f, [image_path, mask_path], [tf.float32, tf.float32])
    image.set_shape([1024, 1024, 3])
    mask.set_shape([1024, 1024, 1])

    return image, mask


get_patches = lambda x, y: (
    tf.reshape(
        tf.image.extract_patches(
            images=tf.expand_dims(x, 0),
            sizes=[1, 512, 512, 1],
            strides=[1, 512, 512, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'), [-1, 512, 512, 3]),
    (tf.reshape(
        tf.image.extract_patches(
            images=tf.expand_dims(y, 0),
            sizes=[1, 512, 512, 1],
            strides=[1, 512, 512, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'), [-1, 512, 512, 1])))


def tf_dataset(images, masks, batch=4):
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(preprocess)
    dataset = dataset.map(get_patches)
    dataset = dataset.repeat(100)
    dataset = dataset.prefetch(2)
    return dataset

if __name__ == "__main__":
    (train_x, train_y), (test_x, test_y) = load_dataset()
    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Validation: {len(test_x)} - {len(test_y)}")

    train_dataset = tf_dataset(train_x, train_y, batch=4)

