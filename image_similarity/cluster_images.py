##### Clusters the Embeddings using PCA and T-SNE #########

__all__ = ["cluster_images", "vizualise_tsne"]

import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import config
import matplotlib.pyplot as plt


def cluster_images(embedding, pca_num_components: int, tsne_num_components: int):
    """
    Clusters and plots the images using PCA + T-SNE approach.
    Args:

    embedding: A 2D Vector of image embeddings.
    pca_num_components: Number of componenets PCA should reduce.
    tsne_num_components: Number of componenets T-SNE should reduce to. Suggested: 2
    """
    pca_file_name = f"..//data//models//pca_{pca_num_components}.pkl"
    tsne_file_name = f"..//data//models//tsne_{tsne_num_components}.pkl"
    tsne_embeddings_file_name = (
        f"..//data//models//tsne_embeddings_{tsne_num_components}.pkl"
    )

    print("Reducing Dimensions using PCA")

    pca = PCA(n_components=pca_num_components, random_state=42)
    reduced_embedding = pca.fit_transform(embedding)
    # print(reduced_embedding.shape)

    # Cluster them using T-SNE.
    print("Using T-SNE to cluster them")
    tsne_obj = TSNE(
        n_components=tsne_num_components,
        verbose=1,
        random_state=42,
        perplexity=200,
        n_iter=1000,
        n_jobs=-1,
    )

    tsne_embedding = tsne_obj.fit_transform(reduced_embedding)

    # print(tsne_embedding.shape)

    # Dump the TSNE and PCA object.
    pickle.dump(pca, open(pca_file_name, "wb"))
    # pickle.dump(tsne_embedding)
    pickle.dump(tsne_obj, open(tsne_file_name, "wb"))

    # Vizualize the TSNE.
    vizualise_tsne(tsne_embedding)

    # Save the embeddings.
    pickle.dump(tsne_embedding, open(tsne_embeddings_file_name, "wb"))


def vizualise_tsne(tsne_embedding):
    """
    Plots the T-SNE embedding.
    Args:
    tsne_embedding: 2 Dimensional T-SNE embedding.
    """

    x = tsne_embedding[:, 0]
    y = tsne_embedding[:, 1]

    plt.scatter(x, y, c=y)
    plt.show()


if __name__ == "__main__":
    # Loads the embedding
    embedding = np.load(config.EMBEDDING_PATH)

    # print(embedding.shape)

    # Reshape Back to Encoder Embeddings.
    # NUM_IMAGES = (4739, )
    # embedding_shape = NUM_IMAGES + config.EMBEDDING_SHAPE[1:]
    # print(embedding_shape)

    # embedding = np.reshape(embedding, embedding_shape)
    # print(embedding.shape)

    cluster_images(embedding, pca_num_components=50, tsne_num_components=2)
