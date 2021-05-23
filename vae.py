import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from librosa.feature.inverse import mfcc_to_audio
from scipy.io import wavfile
import featureextraction
import soundfile as sf

# Create a sampling layer
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Build the encoder
latent_dim = 32 

encoder_inputs = keras.Input(shape=(60, 13, 1))
x = layers.Conv2D(32, 3, activation="tanh", strides=(2, 1), padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="tanh", strides=(2, 1), padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="tanh")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

# Build the decoder
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(15 * 13 * 64, activation="tanh")(latent_inputs)
x = layers.Reshape((15, 13, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="tanh", strides=(2, 1), padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="tanh", strides=(2, 1), padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="tanh", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()


# Define the VAE as a Model with a custom train_step
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
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

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            #reconstruction_loss = keras.losses.mean_squared_error(data, reconstruction)
            reconstruction_loss = keras.losses.binary_crossentropy(data, reconstruction)
            #reconstruction_loss *= 60 * 13
            #reconstruction_loss = tf.reduce_mean(
            #    tf.reduce_sum(
            #        keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
            #    )
            #)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
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

data = np.load('lab2_data.npz', allow_pickle=True)['data']
X_train = np.zeros((44, 60, 13))

# Store everything for normalising
all_data = None 
for d in data:
    if all_data is None:
        all_data = d["lmfcc"][:60, :]
    else:
        all_data = np.vstack([all_data, d["lmfcc"][:60, :]])

print(all_data.shape)
scaler = StandardScaler()
scaler.fit(all_data)

for i, d in enumerate(data): 
    X_train[i] = scaler.transform(d["lmfcc"][:60, :])

X_train = np.expand_dims(X_train, -1) 
print(X_train.shape)

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
#vae.fit(X_train, epochs=150, batch_size=22)

example_digit = data[18]["lmfcc"][:60, :]

plt.figure()
plt.pcolormesh(example_digit)
plt.colorbar()
plt.savefig("input.png")

example = example_digit 
example = scaler.transform(example)
example = np.expand_dims(example, 0)

z_mean, z_log_var, z = vae.encoder(example)
reconstruction = vae.decoder(z)
reconstruction = tf.reshape(reconstruction, (60, 13))
reconstruction = scaler.inverse_transform(reconstruction) 

plt.figure()
plt.pcolormesh(reconstruction)
plt.colorbar()
plt.savefig("output.png")

audio = mfcc_to_audio(reconstruction.T, n_mels=128, lifter=22, n_fft=512, hop_length=160, win_length=320)
example_digit_audio = mfcc_to_audio(example_digit.T, n_mels=128, lifter=22, n_fft=512, hop_length=160, win_length=320)

sf.write("input.wav", samplerate=16000, data=example_digit_audio)
sf.write("output.wav", samplerate=16000, data=audio)
"""
data = np.load('lab2_data.npz', allow_pickle=True)['data']
X_train = None

# preprocess
for d in data:
    lmfcc = np.ndarray.flatten(d["lmfcc"][:60, :]).reshape((1, 780)) # fix time steps
    if X_train is None:
        X_train = lmfcc
    else:
        X_train = np.vstack((X_train,lmfcc))
# X_train == (44, 780)


#§X_train = np.expand_dims(X_train, 0)

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(X_train, epochs=10, batch_size=1)

first_datapoint = X_train[0].reshape((1, 780))

z_mean, z_log_var, z = vae.encoder(first_datapoint)
reconstruction = vae.decoder(z)

output = tf.reshape(reconstruction, ((60, 13)))

plt.pcolormesh(output)
plt.savefig("output.png")
"""