## Image similarity searching.

We have to solve problem given in the below figure

![Image Similarity](images/image_similarity_diagram.png)

Given a new Query Image, we need to find most similar images from embeddings and return them.

Here, we make use of embeddings that we have learnt. These embeddings contain representation of images on which the convolutional encoder was trained on.

## Creating Embedding for Query Image and Searching

We convert the query Image to query_embedding using the encoder.

This query_embedding is used to search for similar images from embedding which we learnt before using autoencoder.

Now, we use K-Nearest Neighbors to find `K` number of nearest embeddings.
These nearest embeddings are those in pre-learnt embedding which are nearest to query_embedding.

Use the indices of these nearest embeddings, we display the most similar image from our images folder.

## Models

- Some models trained using a sample dataset are given in [this](https://drive.google.com/drive/folders/1T6uS4O7LiuGMtXlRSMFmKAEDNQ08Ry25?usp=sharing) link.
- It contains encoder, decoder and numpy embeddings generated from auto-enocoders.
- These are obtained by running `torch_train.py` script.

## Outputs

- Query Image 

![Query Image](images/query_image_3699.jpg)


- Recommended Images

![Recommended Images](images/recommended_835.jpg)


![Recommended Images](images/recommended_1215.jpg)


![Recommended Images](images/recommended_2300.jpg)

