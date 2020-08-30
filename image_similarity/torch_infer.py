__all__ = [
    "load_image_tensor",
    "compute_similar_images",
    "compute_similar_features",
    "plot_similar_images",
]

import config
import torch
import numpy as np
import torch_model
from sklearn.neighbors import NearestNeighbors
import torchvision.transforms as T
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def load_image_tensor(image_path, device):
    """
    Loads a given image to device.
    Args:
    image_path: path to image to be loaded.
    device: "cuda" or "cpu"
    """
    image_tensor = T.ToTensor()(Image.open(image_path))
    image_tensor = image_tensor.unsqueeze(0)
    # print(image_tensor.shape)
    # input_images = image_tensor.to(device)
    return image_tensor


def compute_similar_images(image_path, num_images, embedding, device):
    """
    Given an image and number of similar images to generate.
    Returns the num_images closest neares images.

    Args:
    image_path: Path to image whose similar images are to be found.
    num_images: Number of similar images to find.
    embedding : A (num_images, embedding_dim) Embedding of images learnt from auto-encoder.
    device : "cuda" or "cpu" device.
    """

    image_tensor = load_image_tensor(image_path, device)
    # image_tensor = image_tensor.to(device)

    with torch.no_grad():
        image_embedding = encoder(image_tensor).cpu().detach().numpy()

    # print(image_embedding.shape)

    flattened_embedding = image_embedding.reshape((image_embedding.shape[0], -1))
    # print(flattened_embedding.shape)

    knn = NearestNeighbors(n_neighbors=num_images, metric="cosine")
    knn.fit(embedding)

    _, indices = knn.kneighbors(flattened_embedding)
    indices_list = indices.tolist()
    # print(indices_list)
    return indices_list


def plot_similar_images(indices_list):
    """
    Plots images that are similar to indices obtained from computing simliar images.
    Args:
    indices_list : List of List of indexes. E.g. [[1, 2, 3]]
    """

    indices = indices_list[0]
    for index in indices:
        if index == 0:
            # index 0 is a dummy embedding.
            pass
        else:
            img_name = str(index - 1) + ".jpg"
            img_path = os.path.join(config.DATA_PATH + img_name)
            # print(img_path)
            img = Image.open(img_path).convert("RGB")
            plt.imshow(img)
            plt.show()
            img.save(f"../outputs/query_image_3/recommended_{index - 1}.jpg")


def compute_similar_features(image_path, num_images, embedding, nfeatures=30):
    """
    Given a image, it computes features using ORB detector and finds similar images to it
    Args:
    image_path: Path to image whose features and simlar images are required.
    num_images: Number of similar images required.
    embedding: 2 Dimensional Embedding vector.
    nfeatures: (optional) Number of features ORB needs to compute
    """

    image = cv2.imread(image_path)
    orb = cv2.ORB_create(nfeatures=nfeatures)

    # Detect features
    keypoint_features = orb.detect(image)
    # compute the descriptors with ORB
    keypoint_features, des = orb.compute(image, keypoint_features)

    # des contains the description to features

    des = des / 255.0
    des = np.expand_dims(des, axis=0)
    des = np.reshape(des, (des.shape[0], -1))
    # print(des.shape)
    # print(embedding.shape)

    pca = PCA(n_components=des.shape[-1])
    reduced_embedding = pca.fit_transform(
        embedding,
    )
    # print(reduced_embedding.shape)

    knn = NearestNeighbors(n_neighbors=num_images, metric="cosine")
    knn.fit(reduced_embedding)
    _, indices = knn.kneighbors(des)

    indices_list = indices.tolist()
    # print(indices_list)
    return indices_list


if __name__ == "__main__":
    # Loads the model

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    encoder = torch_model.ConvEncoder()

    # Load the state dict of encoder
    encoder.load_state_dict(torch.load(config.ENCODER_MODEL_PATH, map_location=device))
    encoder.eval()
    encoder.to(device)

    # Loads the embedding
    embedding = np.load(config.EMBEDDING_PATH)

    indices_list = compute_similar_images(
        config.TEST_IMAGE_PATH, config.NUM_IMAGES, embedding, device
    )
    plot_similar_images(indices_list)
    indices_list = compute_similar_features(config.TEST_IMAGE_PATH, 5, embedding)
    plot_similar_images(indices_list)
