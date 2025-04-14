🧪 Experiment 1: Clothing Item Color Detection
(i) Problem Statement:
Detect the dominant color from clothing images (or live camera feed) to assist in fashion cataloging or e-commerce tagging.
________________


(ii) Objectives:
1. Identify the most frequently occurring color in a clothing item image.

2. Enable real-time color detection using a webcam feed.

________________


(iii) Methodology Used:
   * Use OpenCV to read and process images (static or live).

   * Reshape image pixels into a 2D array of RGB values.

   * Apply KMeans clustering (from sklearn) to group similar colors.

   * Extract the most dominant cluster center as the primary color.

   * Display or print the detected color for further usage.

________________


(iv) Outcomes:
      * Successfully detected and displayed the dominant RGB color from static images.

      * Real-time color detection was implemented using a live camera feed, updating per frame.

      * The system can now be used for color-based sorting, filtering, or tagging in fashion datasets.

________________




🤖 Experiment 2: Clothing Texture Classification Using Autoencoders
(i) Problem Statement:
Use autoencoders to learn compressed representations of clothing textures for future classification or feature extraction.
________________


(ii) Objectives:
         1. Train an autoencoder to learn low-dimensional representations of clothing item textures.

         2. Visually compare reconstructed outputs to assess how well the model preserves texture details.

________________


(iii) Methodology Used:
            * Load and normalize Fashion MNIST dataset (28×28 grayscale clothing images).

            * Flatten images into 784-length vectors for model input.

            * Build an autoencoder with:

               * Encoder: Dense layer to compress to 32 units.

               * Decoder: Dense layer to reconstruct to 784 units.

                  * Use binary cross entropy loss with the Adam optimizer.

                  * Train for 10 epochs and validate on test data.

                  * Visualize original vs. reconstructed textures using matplotlib.

________________


(iv) Outcomes:
                     * The autoencoder successfully learned to reconstruct clothing textures with reasonable accuracy.

                     * Reconstructed images preserved major shape and texture features of original inputs.

                     * The encoded representations can now be used as texture features for clustering or classification.
