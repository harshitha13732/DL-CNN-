# DL-CNN

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: GARLAPATI HARSHITHA

*INTERN ID*: CT08DZ2406

*DOMAIN*: DATA SCIENCE

*DURATION*: 6 WEEKS (AUG-2ND 2025 TO SEP-17TH 2025)

*MENTOR*: NEELA SANTOSH

*DESCRIPTION*: 

Introduction

The goal of this project was to design and implement a deep learning model for image classification using the CIFAR-10 dataset and TensorFlow. The CIFAR-10 dataset is widely used for benchmarking computer vision models and consists of 60,000 color images in 10 different categories such as airplanes, automobiles, birds, cats, dogs, and ships. Each image is only 32×32 pixels in size, which makes classification both challenging and insightful. This project provided hands-on experience in building and training a Convolutional Neural Network (CNN), evaluating its performance, and visualizing the results.

Dataset and Preprocessing

CIFAR-10 comes built-in with TensorFlow/Keras, making it easy to load. The dataset contains 50,000 training images and 10,000 testing images. Before feeding them into the model, we normalized the pixel values from the original range of [0–255] to [0–1]. This normalization helps the neural network train faster and more effectively. We also mapped numeric labels to human-readable class names (e.g., “0 → airplane, 1 → automobile”), which allowed us to interpret the predictions more easily.

Model Architecture

We built a Convolutional Neural Network (CNN) step by step. The architecture included:

Convolutional layers: To automatically extract features such as edges, textures, and patterns.

Pooling layers (MaxPooling2D): To reduce spatial dimensions and retain key information.

Flattening layer: To convert feature maps into a one-dimensional vector.

Dense layers: Fully connected layers for learning complex patterns and making predictions.

Output layer: A dense layer with 10 neurons (for 10 classes).

The ReLU activation function was used in hidden layers, while the output layer was designed to predict class probabilities.

Model Training

The model was compiled using the Adam optimizer and Sparse Categorical Crossentropy loss function, which is suitable for multi-class classification tasks. Training was done for 10 epochs with both training and validation accuracy being tracked. This step allowed us to measure how well the model generalized to unseen data. By the end of training, the model reached an accuracy of approximately 70–75% on the test dataset.

Evaluation and Visualization

After training, the model was evaluated on the test set. To gain deeper insights, we plotted the accuracy and loss curves across epochs, which showed steady learning and highlighted opportunities for improvement. We also made predictions on random sample test images and displayed them with their true and predicted labels. While the model performed well in most cases, some misclassifications occurred, such as predicting a “frog” as a “deer.” These errors are common due to the small size and complexity of CIFAR-10 images.

Future Improvements

Although the model performed reasonably well, there is room for improvement. Some possible enhancements include:

Dropout layers to reduce overfitting.

Batch Normalization to stabilize and speed up training.

Data Augmentation (rotation, flipping, zooming) to expose the model to more variations.

Transfer Learning using pretrained models like ResNet or VGG, which can boost accuracy above 90%.

This project successfully implemented a deep learning model for image classification using TensorFlow. It covered data preprocessing, model building, training, evaluation, and visualization. The experience not only met the project requirements but also provided practical exposure to CNNs, one of the most powerful techniques in computer vision. The knowledge gained will be valuable for future research, internships, and interviews in data science and AI fields.

*OUTPUT*:
<img width="1853" height="745" alt="Image" src="https://github.com/user-attachments/assets/499a10f7-e806-4be1-b01a-0c2d1d28be15" />

