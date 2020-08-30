# Let's make the app

from flask import Flask, request, json
import torch_model
import config
import torch
import numpy as np

from sklearn.neighbors import NearestNeighbors
import torchvision.transforms as T
import os
import cv2
from PIL import Image
from sklearn.decomposition import PCA

app = Flask(__name__)

print("App started")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load the model before we start the server
encoder = torch_model.ConvEncoder()
# Load the state dict of encoder
encoder.load_state_dict(torch.load(config.ENCODER_MODEL_PATH, map_location=device))
encoder.eval()
encoder.to(device)
# Loads the embedding
embedding = np.load(config.EMBEDDING_PATH)

print("Loaded model and embeddings")


def compute_similar_images(image_tensor, num_images, embedding, device):
    """
    Given an image and number of similar images to generate.
    Returns the num_images closest neares images.

    Args:
    image_tenosr: PIL read image_tensor whose similar images are needed.
    num_images: Number of similar images to find.
    embedding : A (num_images, embedding_dim) Embedding of images learnt from auto-encoder.
    device : "cuda" or "cpu" device.
    """

    image_tensor = image_tensor.to(device)

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


def compute_similar_features(image, num_images, embedding, nfeatures=30):
    """
    Given a image, it computes features using ORB detector and finds similar images to it
    Args:
    image: Opencv read Image whose features and simlar images are required.
    num_images: Number of similar images required.
    embedding: 2 Dimensional Embedding vector.
    nfeatures: (optional) Number of features ORB needs to compute
    """

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


# For the home route and health check
@app.route("/")
def index():
    return "App is Up"


@app.route("/simfeat", methods=["POST"])
def simfeat():
    r = request.files["image"]
    # print("Hi")
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    indices_list = compute_similar_features(img, num_images=5, embedding=embedding)
    # Need to display the images
    return (
        json.dumps({"indices_list": indices_list}),
        200,
        {"ContentType": "application/json"},
    )


@app.route("/simimages", methods=["POST"])
def simimages():
    image = request.files["image"]
    # print("Hi")
    image = Image.open(image)
    image_tensor = T.ToTensor()(image)
    image_tensor = image_tensor.unsqueeze(0)
    indices_list = compute_similar_images(
        image_tensor, num_images=5, embedding=embedding, device=device
    )
    # Need to display the images
    return (
        json.dumps({"indices_list": indices_list}),
        200,
        {"ContentType": "application/json"},
    )


if __name__ == "__main__":
    app.run(debug=False, port=9000)
