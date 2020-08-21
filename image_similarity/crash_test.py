import config
import torch
import numpy as np
import torch_model
from sklearn.neighbors import NearestNeighbors
import torchvision.transforms as T
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2

if __name__ == "__main__":
    image = cv2.imread(config.TEST_IMAGE_PATH)
    # Initialize ORB detector
    orb = cv2.ORB_create(20)

    # Detect features
    kp = orb.detect(image, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(image, kp)
    print(des.shape)
    des = des / 255.0
    des = np.expand_dims(des, axis=0)
    des = np.reshape(des, (des.shape[0], -1))
    print(des.shape)

    img2 = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0), flags=0)
    plt.imshow(img2)
    plt.show()

    # print(type(kp))
    # print(kp)
