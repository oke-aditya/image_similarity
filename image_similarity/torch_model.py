__all__ = ["ConvEncoder", "ConvDecoder"]

import torch
import torch.nn as nn

# import config


class ConvEncoder(nn.Module):
    """
    A simple Convolutional Encoder Model
    """

    def __init__(self):
        super().__init__()
        # self.img_size = img_size
        self.conv1 = nn.Conv2d(3, 16, (3, 3), padding=(1, 1))
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=(1, 1))
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding=(1, 1))
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d((2, 2))

        self.conv4 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1))
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d((2, 2))

        self.conv5 = nn.Conv2d(128, 256, (3, 3), padding=(1, 1))
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d((2, 2))

    def forward(self, x):
        # Downscale the image with conv maxpool etc.
        # print(x.shape)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # print(x.shape)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # print(x.shape)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        # print(x.shape)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)

        # print(x.shape)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)

        # print(x.shape)
        return x


class ConvDecoder(nn.Module):
    """
    A simple Convolutional Decoder Model
    """

    def __init__(self):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(256, 128, (2, 2), stride=(2, 2))
        # self.upsamp1 = nn.UpsamplingBilinear2d(2)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(128, 64, (2, 2), stride=(2, 2))
        # self.upsamp1 = nn.UpsamplingBilinear2d(2)
        self.relu2 = nn.ReLU(inplace=True)

        self.deconv3 = nn.ConvTranspose2d(64, 32, (2, 2), stride=(2, 2))
        # self.upsamp1 = nn.UpsamplingBilinear2d(2)
        self.relu3 = nn.ReLU(inplace=True)

        self.deconv4 = nn.ConvTranspose2d(32, 16, (2, 2), stride=(2, 2))
        # self.upsamp1 = nn.UpsamplingBilinear2d(2)
        self.relu4 = nn.ReLU(inplace=True)

        self.deconv5 = nn.ConvTranspose2d(16, 3, (2, 2), stride=(2, 2))
        # self.upsamp1 = nn.UpsamplingBilinear2d(2)
        self.relu5 = nn.ReLU(inplace=True)

    def forward(self, x):
        # print(x.shape)
        x = self.deconv1(x)
        x = self.relu1(x)
        # print(x.shape)

        x = self.deconv2(x)
        x = self.relu2(x)
        # print(x.shape)

        x = self.deconv3(x)
        x = self.relu3(x)
        # print(x.shape)

        x = self.deconv4(x)
        x = self.relu4(x)
        # print(x.shape)

        x = self.deconv5(x)
        x = self.relu5(x)
        # print(x.shape)
        return x


# if __name__ == "__main__":
# img_random = torch.randn(1, 3, 512, 512)
# img_random2 = torch.randn(1, 3, 512, 512)
# print(img_random.shape)

# enc = ConvEncoder()
# dec = ConvDecoder()

# enc_out = enc(img_random)
# enc_out2 = enc(img_random2)
# print(enc_out.shape)
# print(enc_out2.shape)

# emb = torch.cat((enc_out, enc_out2), 0)
# print(emb.shape)

# # embedding = torch.randn(config.EMBEDDING_SHAPE)
# # print(embedding.shape)

# dec_out = dec(enc_out)
# print(dec_out.shape)

# embedding_dim = config.EMBEDDING_SHAPE
# embedding = torch.randn(embedding_dim)
# print(embedding.shape)

# img_random = torch.randn(32, 3, 512, 512)
# img_random2 = torch.randn(32, 3, 512, 512)

# encoder = ConvEncoder()
# enc_output = encoder(img_random).cpu()
# embedding = torch.cat((embedding, enc_output), 0)
# print(embedding.shape)

# enc_output2 = encoder(img_random2).cpu()
# embedding = torch.cat((embedding, enc_output2), 0)
# print(embedding.shape)

# print(embedding.numpy().shape)
