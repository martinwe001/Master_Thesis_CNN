import cv2
import os


data_type = 'images'
folder_path = f'dataset/{data_type}/'


def image_augmentation(path):

    for image in os.listdir(path):
        original_image = cv2.imread(os.path.join(path, image))

        flip_vertical = cv2.flip(original_image, 0)
        flip_horizontal = cv2.flip(original_image, 1)
        flip_both = cv2.flip(original_image, -1)

        name = image.split('.')[0]

        cv2.imwrite(f'dataset/{data_type}/{name}-1' + '.jpg', flip_vertical)
        cv2.imwrite(f'dataset/{data_type}/{name}-2' + '.jpg', flip_horizontal)
        cv2.imwrite(f'dataset/{data_type}/{name}-3' + '.jpg', flip_both)

    cv2.waitKey()


image_augmentation(folder_path)