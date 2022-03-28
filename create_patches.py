import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt


dataset_path = "test_dataset"
images_path = "images_old/*"
masks_path = "masks_old/*"

image_folder = 'dataset/images_old/'
masks_folder = 'dataset/masks_old/'


def create_patches(image_folder, mask_folder):

    def load_masks_from_folder(folder):
        masks = []
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                masks.append(img)
        return masks

    def load_images_from_folder(folder):
        images = []
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
        return images

    images = load_images_from_folder(image_folder)
    masks = load_masks_from_folder(masks_folder)

    images = tf.cast(images, tf.float32)
    masks = tf.cast(masks, tf.float32)

    masks = tf.reshape(masks, [-1, 256, 256, 1])
    """
    image = read_image(os.path.join(dataset_path,'images_old/trd_1_crop_6.jpg'))
    mask = read_image(os.path.join(dataset_path,'masks_old/trd_1_crop_mask_6.jpg'))
    
    image = tf.expand_dims(image, 0)
    mask = tf.expand_dims(mask, 0)
    """

    image_patches = tf.image.extract_patches(images, sizes=[1, 256, 256, 1], strides=[1, 256, 256, 1], rates=[1, 1, 1, 1], padding='VALID')
    mask_patches = tf.image.extract_patches(masks, sizes=[1, 256, 256, 1], strides=[1, 256, 256, 1], rates=[1, 1, 1, 1], padding='VALID')


    image_patches = tf.reshape(image_patches, [-1, 256, 256, 3])
    mask_patches = tf.reshape(mask_patches, [-1, 256, 256, 1])

    return image_patches, mask_patches


def create_dataset(image_patches, mask_patches):
    dataset = tf.data.Dataset.from_tensor_slices((image_patches, mask_patches))
    # dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.batch(4)
    dataset = dataset.prefetch(2)

    return dataset






