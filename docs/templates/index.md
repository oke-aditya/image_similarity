## Auto-encoder based Image-Similarity Engine

- Builds a simple Convolutional Auto-encoder based Image similarity engine.
- Dataset is available over Kaggle https://www.kaggle.com/datasets/okeaditya/animals-data

## Convolutional Auto-encoder
![Autoencoder](images/conv_autoencoder.png)

Convolutional autoencoder consists of two parts, encoder and decoer.

Encoder

- CNN Encoder converts the given input image into an embedding representation of size (bs, c, h, w)
- It contains of several CNN, RELU, MaxPool2d layers on top of each other.

Decoder

- CNN Decoder converts the image generated from Encoder back to the input image.
- It consists of Conv2DTranspose layers, which is transposed convulation operation, helping to upsample the size.
- Alternatively, it can also be made using Bilinear interpolation or Upsampling layer to upsample to orignal image size.

### Auto-Encoder

Auto-encoder combines both encoder and decoder to learn a feature representation of input images.
Both the parameters are combined and trained with a single common loss function and optimizer.

The encoder layers give us a resultant latent representation of images through convolutional layers.


