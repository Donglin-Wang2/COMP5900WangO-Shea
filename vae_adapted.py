import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Layer, LeakyReLU
import tensorflow.keras.layers as layers
class VAEAdapted(keras.Model):
    
    def __init__(self, latent_dim, **kwargs):
        super(VAEAdapted, self).__init__(**kwargs)
        self.concat_layer = layers.Concatenate(axis=-1)
        self.fc_log_var = tf.keras.Sequential([
            layers.Dense(latent_dim),
            LeakyReLU(),
            layers.Dense(latent_dim),
            LeakyReLU(),
            layers.Dense(latent_dim)
        ])
        self.fc_mean = tf.keras.Sequential([
            layers.Dense(latent_dim),
            LeakyReLU(),
            layers.Dense(latent_dim),
            LeakyReLU(),
            layers.Dense(latent_dim)
        ])
       
        self.decoder = tf.keras.Sequential([
            keras.Input(shape=(latent_dim,)),
            layers.Dense(8 * 8 * 128),
            layers.Reshape((8, 8, 128)),
            layers.Conv2DTranspose(128, 3, strides=2, padding="same"),
            layers.Conv2DTranspose(64, 3, strides=2, padding="same"),
            layers.Conv2DTranspose(32, 3, strides=2, padding="same"),
            layers.Conv2DTranspose(16, 3, strides=2, padding="same"),
            layers.Conv2DTranspose(1, 3, activation='sigmoid', strides=2, padding="same"),
        ])
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean
    def encode(self, img_batch, depth_batch):
        z = self.concat_layer([img_batch, depth_batch])
        z_mean, z_log_var = self.fc_mean(z), self.fc_log_var(z)
        z = self.reparameterize(z_mean, z_log_var)
        return z, z_mean, z_log_var
    def sample(self, data):
        img_batch, depth_batch, _ = data
        z, z_mean, z_log_var = self.encode(img_batch, depth_batch)
        reconstruction = self.decoder(z)
        return reconstruction
    def train_step(self, data):
        img_batch, depth_batch, mask_batch = data
        with tf.GradientTape() as tape:
            z, z_mean, z_log_var = self.encode(img_batch, depth_batch)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_sum(
                keras.losses.binary_crossentropy(mask_batch, reconstruction)
            )
            kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = reconstruction_loss + kl_loss
    
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }