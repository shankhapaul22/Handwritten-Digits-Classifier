# Context

Within the field of machine learning and pattern recognition, image classification (especially for handwritten text) is towards the difficult end of the spectrum. There are a few reasons for this.

First, each image in a training set is high dimensional. Each pixel in an image is a feature and a separate column. This means that a 128 x 128 image has 16384 features.

Second, images are often downsampled to lower resolutions and transformed to grayscale (no color). This is a limitation of compute power unfortunately. The resolution of a 8 megapixel photo has 3264 by 2448 pixels, for a total of 7,990,272 features (or about 8 million). Images of this resolution are usually scaled down to between 128 and 512 pixels in either direction for significantly faster processing. This often results in a loss of detail that's available for training and pattern matching.

Third, the features in an image don't have an obvious linear or nonlinear relationship that can be learned with a model like linear or logistic regression. In grayscale, each pixel is just represented as a brightness value ranging from 0 to 256.

---

Deep neural networks have been used to reach state-of-the-art performance on image classification tasks in the last decade. 
For some image classification tasks, deep neural networks actually perform as well as or slightly better than the human benchmark. Deep learning is effective in image classification because of the models' ability to learn hierarchical representations. At a high level, an effective deep learning model learns intermediate representations at each layer in the model and uses them in the prediction process.

# Objective

In this project, we'll:

- explore image classification.
- observe the limitation of traditional machine learning.
- train, test, and improve a few different deep neural networks for image classification.

# Process

Scikit-learn contains a number of datasets pre-loaded with the library, within the namespace of `sklearn.datasets`. The `load_digits()` function returns a copy of the hand-written digits dataset from UCI.

Each image is represented as a row of pixel values. To visualize the image, we need to reshape these pixel values back into the 28 by 28 and plot them on a coordinate grid.

The different models we attempt:
- We use the K-nearest neighbors algorithm compares every unseen observation in the test set to all (or many, as some implementations constrain the search space) training observations to look for similar (or the "nearest") observations. Then, the algorithm finds the label with the most nearby observations and assigns that as the prediction for the unseen observation.

- Then we use a neural network with a single hidden layer.
- Then we use two hidden layers and continued to increase the number of neurons in each layer.
- Lastly, we increase the number of folds we use for k-fold cross validation to 6 (originally we were using 4) while testing networks with 3 hidden layers.

# Results

There are a few downsides to K-nearest neighbors including high memory usage and no model representation to debug. In the first two models of neural networks ( one hidden layer, two hidden layers), the accuracy increases as we included more neurons per layer, however there is little difference in accuracy between the models.

For our last model (three hidden layers with 6 K-Folds), the accuracy decrease but variance and chance of overfitting decrease as well.
