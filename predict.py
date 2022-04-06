import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow_addons as tfa
from load_test_images import load_dataset, tf_dataset



if __name__ == "__main__":
    """ Load the model """

    model = 'master_model'
    epochs = 26
    batches = 1
    lr = 1e-4

    model = tf.keras.models.load_model(f"models/{model}_{epochs}_{batches}_{lr}.h5", custom_objects={'MaxUnpooling2D': tfa.layers.MaxUnpooling2D})

    dataset = load_dataset()
    dataset = tf_dataset(dataset)
    for i, element in enumerate(dataset):
        for j, image in enumerate(element):
            original_image = image
            h, w, _ = image.shape

            x = np.expand_dims(image, axis=0)
            pred_mask = model.predict(x)[0]

            pred_mask = np.concatenate(
                [
                    pred_mask,
                    pred_mask,
                    pred_mask
                ], axis=2)
            pred_mask = (pred_mask > 0.3) * 255
            pred_mask = pred_mask.astype(np.float32)
            # pred_mask = cv2.resize(pred_mask, (w, h))
            original_image = np.array(original_image)

            alpha_image = 0
            alpha_mask = 1
            cv2.addWeighted(pred_mask, alpha_mask, original_image, alpha_image, 0, original_image)
            image = tf.data.Dataset.from_tensor_slices(original_image)
            cv2.imwrite(f"pred_test/{i+1}-{j+1}.jpg", original_image)



    """
        for path in tqdm(test_images, total=len(test_images)):
        x = cv2.imread(path, cv2.IMREAD_COLOR)
        original_image = x
        h, w, _ = x.shape

        # x = x/255.0
        x = x.astype(np.float32)

        x = np.expand_dims(x, axis=0)
        pred_mask = model.predict(x)[0]

        pred_mask = np.concatenate(
            [
                pred_mask,
                pred_mask,
                pred_mask
            ], axis=2)
        pred_mask = (pred_mask > 0.5) * 255
        pred_mask = pred_mask.astype(np.float32)
        pred_mask = cv2.resize(pred_mask, (w, h))

        original_image = original_image.astype(np.float32)

        alpha_image = 0.5
        alpha_mask = 1
        cv2.addWeighted(pred_mask, alpha_mask, original_image, alpha_image, 0, original_image)

        name = path.split("/")[-1]
        cv2.imwrite(f"predictions/{name}", original_image)

    """

