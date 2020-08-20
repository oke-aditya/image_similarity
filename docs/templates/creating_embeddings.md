## Creating Embeddings.

To find similar images, we first need to create embeddings from given images. To create embeddings we make use of the convolutional auto-encoder.

This is an unsupervised problem where we use auto-encoders to reconstruct the image.
In tihs porcess the encoder learns embeddings of given images while decoder helps to reconstruct.

It can be summarized as follows

```
encoded_image = encoder(input_image)
reoncstructed_image = decoder(encoded_image)
```

We calculate loss between reconstructed image and input image.

```
loss_fn = nn.MSELoss()
loss = loss_fn(input_image, reconstructed_image)
loss.backward()
optimizer.step()
```

Optimizer updates both encoder and decoder.

## Storing Embeddings

The output of convolutional encoder is collected across all the image batches into a single tensor. It's dimension depends on number of CNN encoding layers used which is the embedding dimension.

E.g. For 4000 images if embedding dimension is (16, 256, 256)

We get representation as (4000, 16, 256, 256)

We flatten these embeddings into numpy arrays of dimension (number_of_images, c * h * w)

Where c, h, w are channels, height and width of the images.

We save these embeddings in numpy "npy" format.
