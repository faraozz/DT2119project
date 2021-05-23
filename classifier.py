import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Classifier(keras.Model):
    def __init__(self, input_dim, output_dim, model, **kwargs) -> None:
        super(Classifier, self).__init__(**kwargs)
        self.model = model
        self.nr_labels = output_dim
        
        self.loss_tracker = keras.metrics.CategoricalCrossentropy()
        self.accuracy_tracker = keras.metrics.CategoricalAccuracy()
        
        self.val_loss_tracker = keras.metrics.CategoricalCrossentropy()
        self.val_accuracy_tracker = keras.metrics.CategoricalAccuracy()

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.accuracy_tracker,
            self.val_loss_tracker,
            self.val_accuracy_tracker
        ]

    def train_step(self, data):
        z, label = data
        with tf.GradientTape() as tape:
            classifier_output = self.model(z)
            classifier_loss = keras.losses.categorical_crossentropy(label, classifier_output)
            #classifier_accuracy = keras.accura

        grads = tape.gradient(classifier_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(label, classifier_output)
        self.accuracy_tracker.update_state(label, classifier_output)
        return {
            "loss": self.loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result()
        }
        
    def test_step(self, data):
        z, label = data
        classifier_output = self.model(z)
        classifier_loss = keras.losses.categorical_crossentropy(label, classifier_output)

        self.val_loss_tracker.update_state(label, classifier_output)
        self.val_accuracy_tracker.update_state(label, classifier_output)
        return {
            "loss": self.val_loss_tracker.result(),
            "accuracy": self.val_accuracy_tracker.result()
        }








