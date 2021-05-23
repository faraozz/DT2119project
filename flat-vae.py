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
from classifier import Classifier

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
input_dim = 101 * 13 #98 * 13
latent_dim = 128 
labels = 35
n_epochs = 25 
n_classifier_epochs = 50 
batch_size = 128
classifier_batch_size = 32

if not os.path.exists("./model-structures"):
    os.mkdir("./model-structures")

if not os.path.exists("./model-results"):
    os.mkdir("./model-results")

if not os.path.exists("./generated-samples"):
    os.mkdir("./generated-samples")

encoder_inputs = keras.Input(shape=(input_dim))
x = layers.Dense(512, activation="tanh")(encoder_inputs)
x = layers.Dense(256, activation="tanh")(x)
x = layers.Dense(128, activation="tanh")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()
keras.utils.plot_model(encoder, "model-structures/encoder-model-plot.png", show_shapes=True)

# Build the decoder
latent_inputs = keras.Input(shape=(latent_dim))
x = layers.Dense(128, activation="tanh")(latent_inputs)
x = layers.Dense(256, activation="tanh")(x)
x = layers.Dense(512, activation="tanh")(x)
decoder_outputs = layers.Dense(input_dim, activation="tanh")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()
keras.utils.plot_model(decoder, "model-structures/decoder-model-plot.png", show_shapes=True)

#classifier 
classifier_inputs = keras.Input(shape=(latent_dim))
x = layers.Dense(64, activation="relu", name="hidden3")(classifier_inputs)
x = layers.Dense(labels, activation=tf.nn.softmax)(x)
classifier = keras.Model(classifier_inputs, x, name="classifier")
classifier.summary()
keras.utils.plot_model(classifier, "model-structures/classifier-model-plot.png", show_shapes=True)

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

        self.val_total_loss_tracker = keras.metrics.Mean(name="val_total_loss")
        self.val_reconstruction_loss_tracker = keras.metrics.Mean(
            name="val_reconstruction_loss"
        )
        self.val_kl_loss_tracker = keras.metrics.Mean(name="val_kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.val_total_loss_tracker,
            self.val_reconstruction_loss_tracker,
            self.val_kl_loss_tracker

        ]

    def train_step(self, data):
        x = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)
            reconstruction_loss = keras.losses.mean_squared_error(x, reconstruction)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)) / input_dim
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

    def test_step(self, data):
        x = data
        if isinstance(data, tuple):
            x = data[0]
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        reconstruction_loss = keras.losses.mean_squared_error(x, reconstruction)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)) / input_dim
        total_loss = reconstruction_loss + kl_loss 
        self.val_total_loss_tracker.update_state(total_loss)
        self.val_reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.val_kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.val_total_loss_tracker.result(),
            "reconstruction_loss": self.val_reconstruction_loss_tracker.result(),
            "kl_loss": self.val_kl_loss_tracker.result(),
        }

def evaluate_utterance(utterance, utterance_name="", counter=0): 
    if not os.path.exists(f"./generated-samples/{utterance_name}"):
        os.mkdir(f"./generated-samples/{utterance_name}")
    example = utterance 
    example_correct_shape = featureextraction.denormalization(example, mean, std, 101, 13)
    print(utterance_name, "max / min", f"{np.max(example_correct_shape)} / {np.min(example_correct_shape)}")

    plt.figure()
    plt.pcolormesh(example_correct_shape.T)
    plt.title("MFCCs over timesteps")
    plt.xlabel("Time step")
    plt.ylabel("MFCC")
    plt.colorbar()
    plt.savefig(f"generated-samples/{utterance_name}/{utterance_name}-input-{counter}.png")

    example = np.expand_dims(example, 0)
    _, _, z = vae.encoder(example)

    example_digit_audio = mfcc_to_audio(example_correct_shape.T, n_mels=128, n_fft=512, hop_length=160, win_length=320)
    sf.write(f"generated-samples/{utterance_name}/{utterance_name}-input-{counter}.wav", samplerate=16000, data=example_digit_audio)
    
    reconstruct(z, utterance_name, counter=counter)

def reconstruct(z, utterance_name="", random=False, counter=0):
    if random and not os.path.exists("./generated-samples/random-samples"):
        os.mkdir("./generated-samples/random-samples")
    reconstruction = vae.decoder(z)
    reconstruction = reconstruction.numpy() 
    reconstruction = featureextraction.denormalization(reconstruction, mean, std, 101, 13) #scaler.inverse_transform(reconstruction) 
    print(utterance_name, "max / min", f"{np.max(reconstruction)} / {np.min(reconstruction)}")

    plt.figure()
    plt.pcolormesh(reconstruction.T)
    plt.title("MFCCs over timesteps (reconstruction)")
    plt.xlabel("Time step")
    plt.ylabel("MFCC")
    plt.colorbar()

    audio = mfcc_to_audio(reconstruction.T, n_mels=128, n_fft=512, hop_length=160, win_length=320)

    if random:
        plt.savefig(f"generated-samples/random-samples/{utterance_name}-output-{counter}.png")
        sf.write(f"generated-samples/random-samples/{utterance_name}-output-{counter}.wav", samplerate=16000, data=audio)
    else:
        plt.savefig(f"generated-samples/{utterance_name}/{utterance_name}-output-{counter}.png")
        sf.write(f"generated-samples/{utterance_name}/{utterance_name}-output-{counter}.wav", samplerate=16000, data=audio)



print("Loading data")
data = featureextraction.loaddata() 
print("Splitting dataset")
training_set, validation_set, testing_set = featureextraction.splitset(0.8, 0.1 , 0.1, data)
print("Computing mean & std")
mean, std = featureextraction.computeMeanSTD(training_set)
print("Normalization")
returnlist = featureextraction.normalization(training_set, validation_set, testing_set, mean, std)

X_train, X_train_labels = returnlist[0]
X_validation, X_validation_labels = returnlist[1]
X_test, X_test_labels = returnlist[2]

print("x train", X_train.shape)
print("x valid", X_validation.shape)
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae_history = vae.fit(X_train, epochs=n_epochs, batch_size=batch_size, validation_data=(X_validation, ), validation_freq=1)

print(vae_history.history.keys())
print("loss shape", np.asarray(vae_history.history["loss"]))
print("validation loss shape", np.asarray(vae_history.history["val_loss"]))
plt.figure()
# create integer xaxis ticks
epoch_range = np.arange(1, n_epochs+1, step=1)
plt.plot(epoch_range,vae_history.history["loss"], label="Training loss")
plt.plot(epoch_range,vae_history.history["val_loss"], label="Validation loss")
plt.title("Variational Autoencoder loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("model-results/vae_loss.png")

""" generate actual sounds """
generated_labels = {} 
for i, label in enumerate(X_test_labels):
    label_string = featureextraction.labeltoname1digit(label)  
    if label_string not in generated_labels:
        generated_labels[label_string] = 0 
    else:
        generated_labels[label_string] += 1
    
    if generated_labels[label_string] < 5:
        evaluate_utterance(X_test[i], f"{label_string}", counter=generated_labels[label_string])



for i in range(5):
    random_sample = tf.random.normal([latent_dim], 0, 1, tf.float32)
    z = np.expand_dims(random_sample, 0)
    reconstruct(z, f"random-sample", random=True, counter=i)

"""Train the classifier by training on the saved z's and labels"""

classifier_data = []
_, _, z_train = vae.encoder(X_train)
categorical_labels = keras.utils.to_categorical(X_train_labels, num_classes=labels)

_, _, z_validation = vae.encoder(X_validation)
validation_categorical_labels = keras.utils.to_categorical(X_validation_labels, num_classes=labels)

classifier = Classifier(input_dim=latent_dim, output_dim=labels, model=classifier)
classifier.compile(optimizer=keras.optimizers.Adam())
classifier_history = classifier.fit(z_train, categorical_labels, epochs=n_classifier_epochs, batch_size=classifier_batch_size, validation_data=(z_validation, validation_categorical_labels))
plt.figure()
# create integer xaxis ticks
epoch_range = np.arange(1, n_classifier_epochs+1, step=1)
plt.plot(epoch_range,classifier_history.history["loss"], label="Training loss")
plt.plot(epoch_range,classifier_history.history["val_loss"], label="Validation loss")
plt.title("Classifier loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("model-results/classifier_loss.png")

# accuracy
plt.figure()
plt.plot(epoch_range,classifier_history.history["accuracy"], label="Training acc.")
plt.plot(epoch_range,classifier_history.history["val_accuracy"], label="Validation acc.")
plt.title("Classifier accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("model-results/classifier_accuracy.png")

# print test accuracy
_, _, z_test = vae.encoder(X_test)
test_categorical_labels = keras.utils.to_categorical(X_test_labels, num_classes=labels)
test_results = classifier.test_step((z_test, test_categorical_labels))
print("Classifier test results: loss", test_results["loss"].numpy(), ", accuracy", test_results["accuracy"].numpy())
