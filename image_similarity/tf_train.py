import tensorflow as tf
import data
import model
import config

if __name__ == "__main__":
    # Our main training script
    print("----- Creating Dataset ----- ")
    full_dataset = data.create_image_dataset(config.IMG_PATH, n_images=100)
    train_size = int(config.TRAIN_RATIO * len(full_dataset))
    # val_size = int(config.VAL_RATIO * len(full_dataset))

    full_dataset = full_dataset.shuffle(config.SHUFFLE_BUFFER_SIZE)
    train_dataset = full_dataset.take(train_size)
    val_dataset = full_dataset.skip(train_size)

    train_dataset = train_dataset.batch(config.TRAIN_BATCH_SIZE)
    val_dataset = val_dataset.batch(config.TEST_BATCH_SIZE)

    print(f"Train Dataset Length : {len(train_dataset)}")
    print(f"Validation Dataset Length : {len(val_dataset)}")
    print(train_dataset)
    print(val_dataset)

    print("-------- Creating models --------- ")

    autoencoder, encoder, decoder = model.create_model()

    print("-------- Models Created ---------- ")

    loss = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)

    # Computer mean average error as metric.
    autoencoder.compile(optimizer=optimizer, loss=loss, metrics=["mae"])

    callbacks_l = []

    history = autoencoder.fit(
        train_dataset, epochs=config.EPOCHS, validation_data=val_dataset
    )

    print("Training Done")
