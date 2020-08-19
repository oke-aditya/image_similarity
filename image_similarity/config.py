IMG_PATH = "../data/images/"
IMG_HEIGHT = 512  # The images are already resized here
IMG_WIDTH = 512  # The images are already resized here

SEED = 42
TRAIN_RATIO = 0.75
VAL_RATIO = 1 - TRAIN_RATIO
SHUFFLE_BUFFER_SIZE = 100

LEARNING_RATE = 1e-3
EPOCHS = 10
TRAIN_BATCH_SIZE = 32  # Let's see, I don't have GPU, Google Colab is best hope
TEST_BATCH_SIZE = 32  # Let's see, I don't have GPU, Google Colab is best hope

AUTOENCODER_MODEL_PATH = "baseline_autoencoder.pt"
ENCODER_MODEL_PATH = "baseline_encoder.pt"
DECODER_MODEL_PATH = "baseline_decoder.pt"

# TEST_RATIO = 0.2
