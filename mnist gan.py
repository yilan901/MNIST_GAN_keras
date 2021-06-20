import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


class GAN(keras.Model):

    def __init__(self):
        super(GAN, self).__init__()
        self.input_dims = 128
        self.image_dims = (28, 28, 1)
        self.discriminator = self.create_discriminator()
        self.generator = self.create_generator()

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, images):
        batch_size = tf.shape(images)[0]
        random_input_vectors = tf.random.normal(shape=(batch_size, self.input_dims))
        fake_images = self.generator(random_input_vectors)
        all_images = tf.concat([fake_images, images], axis=0)
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        # Adding random noise to labels, supposedly this improves performance
        labels += 0.05 * tf.random.uniform(tf.shape(labels))
        # Training the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(all_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        random_input_vectors = tf.random.normal(shape=(batch_size, self.input_dims))
        misleading_labels = tf.zeros((batch_size, 1))
        # Training the generator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_input_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {'d_loss': d_loss, 'g_loss': g_loss}

    def create_generator(self):
        generator = keras.Sequential()
        generator.add(keras.Input(shape=(self.input_dims,)))
        generator.add(layers.Dense(7 * 7 * self.input_dims))
        generator.add(layers.LeakyReLU(alpha=0.2))
        generator.add(layers.Reshape((7, 7, self.input_dims)))
        generator.add(layers.Conv2DTranspose(self.input_dims, (4, 4), strides=(2, 2), padding="same"))
        generator.add(layers.LeakyReLU(alpha=0.2))
        generator.add(layers.Conv2DTranspose(self.input_dims, (4, 4), strides=(2, 2), padding="same"))
        generator.add(layers.LeakyReLU(alpha=0.2))
        generator.add(layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"))
        generator._name = 'generator'
        return generator

    def create_discriminator(self):
        discriminator = keras.Sequential()
        discriminator.add(keras.Input(shape=self.image_dims))
        discriminator.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
        discriminator.add(layers.LeakyReLU(alpha=0.2))
        discriminator.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
        discriminator.add(layers.LeakyReLU(alpha=0.2))
        discriminator.add(layers.GlobalMaxPooling2D())
        discriminator.add(layers.Dense(1))
        discriminator._name = 'discriminator'
        return discriminator


if __name__ == '__main__':
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
    digits = np.concatenate([x_train, x_test])
    digits = digits.astype('float32') / 255.0
    digits = digits.reshape((-1, 28, 28, 1))
    dataset = tf.data.Dataset.from_tensor_slices(digits)
    dataset = dataset.shuffle(buffer_size=42).batch(32)
    gan = GAN()
    gan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
                g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
                loss_fn=keras.losses.BinaryCrossentropy(from_logits=True))
    gan.fit(dataset, epochs=10)
    fig = plt.figure(figsize=(8, 8))
    columns = 1
    rows = 5
    for i in range(1, columns * rows + 1):
        test_image = gan.generator(tf.random.normal(shape=(1, gan.input_dims)))
        test_image = test_image.numpy().squeeze()
        fig.add_subplot(rows, columns, i)
        plt.imshow(test_image)
    plt.show()
