import cv2
import os



interval = 512
stride = 512

# os.remove("demofile.txt")

def crop_images():
    folder_path = '../dataset/test_images_masks'
    for filename in os.listdir(folder_path):
        count = 0
        img = cv2.imread(os.path.join(folder_path, filename))
        for i in range(0, img.shape[0], interval):
            for j in range(0, img.shape[1], interval):
                cropped_img = img[i:i + stride, j:j + stride]
                count += 1
                name = filename.split('.')[0]
                cv2.imwrite(f'../dataset/masks_cropped/{name}_' + str(count) + '.jpg', cropped_img)
    cv2.waitKey()

crop_images()