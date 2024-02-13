Diabetic Retinopathy Prediction using Convolutional Neural Network (CNN) using low-power devices.


Abstract:

Diabetic Retinopathy (DR) is a leading cause of vision loss around the world.  The goal of this project is to apply deep learning (DL) concepts of Artificial Intelligence to build a real-life application that can run a low-power device, while experimenting with various combinations of datasets, convolutional neural network (CNN) models, hyperparameters and predict the severity of diabetic retinopathy.

In this project, I evaluated if deep learning, specifically convolutional neural networks (CNN) can be used for assessment and prediction of diabetic retinopathy. The purpose of this project was to focus on three key areas to get optimal results  – Data sets, deep learning models, and AI inference.

Datasets: I used Retinal Photographs and 5 classes of animal images as inputs to the models.

Models: I tried numerous models with combinations of feature learning layers (Convolutions and Pooling) and classification layers (flattened, fully connected layers). In addition, I used VGG16(4) and ResNet(5) models to obtain baseline results. Finally, I studied the influence of hyperparameters on overfitting and underfitting training curves as well as stride lengths and epochs. I used Google Colab, Jupyter, Tensorflow, and Python for the development.

AI inference on battery-powered devices: I exported trained models and used the TensorFlow Lite interpreter. To perform inference I used an Android phone camera to capture the image of the retina and passed that to the model as input. The model then showed an array of probabilities between 0 and 1 for each class or level of the DR. I used TensorFlow Lite Java API to perform the Inference.



