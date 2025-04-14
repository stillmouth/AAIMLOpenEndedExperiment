import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import fashion_mnist

# Load Fashion MNIST dataset (contains clothing images)
(X_train, _), (X_test, _) = fashion_mnist.load_data()

# Normalize and flatten images
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# Define the autoencoder
input_dim = 784  
input_img = Input(shape=(input_dim,))
encoded = Dense(32, activation="relu")(input_img)
decoded = Dense(input_dim, activation="sigmoid")(encoded)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=10, batch_size=256, shuffle=True, validation_data=(X_test, X_test), verbose=1)

# Test reconstruction
n = 5  # Number of images to display
reconstructed = autoencoder.predict(X_test[:n])

# Display original and reconstructed images
plt.figure(figsize=(10, 4))
for i in range(n):
    # Original images
    plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap="gray")
    plt.axis("off")

    # Reconstructed images
    plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructed[i].reshape(28, 28), cmap="gray")
    plt.axis("off")

plt.show()
