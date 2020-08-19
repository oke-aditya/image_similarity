# Creates tensorflow dataset which is used by tensorflow 2.x

__all__ = ["create_image_dataset"]

import tensorflow as tf
import cv2
import os
import numpy as np
from tqdm import tqdm


def create_image_dataset(data_dir, n_images: int = None):
    all_images = []
    count = 0
    for file_name in tqdm(os.listdir(data_dir)):
        img_path = os.path.join(data_dir, file_name)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Scale the image to 0-255 range
        image = image / 255.0
        all_images.append(image)
        image_array = np.array(all_images)
        # print(image_array.shape)

        if count == n_images:
            break
        else:
            count += 1
        # print(image_array.shape)

    # Since we are training auto-encoders, labels are the images itself.
    # We don't have supervised labels.
    tf_dataset = tf.data.Dataset.from_tensor_slices((image_array, image_array))
    # print(tf_dataset)
    # print(tf_dataset)
    return tf_dataset


# if __name__ == "__main__":
#     create_image_dataset("../data/images/", 100)
