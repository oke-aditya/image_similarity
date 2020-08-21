## Image Clustering

Embeddings which are learnt from convolutional Auto-encoder are used to cluster the images.

Since the dimensionality of Embeddings is big. We first reduce it by fast dimensionality reduction technique such as PCA.

This is required as T-SNE is much slower and would take lot of time and memory in clustering huge embeddings.

After that we use T-SNE (T-Stochastic Nearest Embedding) to reduce the dimensionality further.

Since these are unsupervised embeddings. Clustering might help us to find classes.

## Clustering output. 

The clusters are note quite clear as model used in very simple one.

T-SNE is takes time to converge and needs lot of tuning.

Also the embeddings can be learnt much better with pretrained models, etc.

![Cluster Output](images/image_clusters.png)

The clustering script can be found [here](https://github.com/oke-aditya/image_similarity/blob/master/image_similarity/cluster_images.py)

It can be used with any arbitrary 2 dimensional embedding learnt using Auto-Encoders.

