Diabetic Retinopathy Prediction using Convolutional Neural Network (CNN) using low-power devices.


Abstract:

Diabetic Retinopathy (DR) is a leading cause of vision loss around the world.  The goal of this project is to apply deep learning (DL) concepts of Artificial Intelligence to build a real-life application that can run a low-power device, while experimenting with various combinations of datasets, convolutional neural network (CNN) models, hyperparameters and predict the severity of diabetic retinopathy.

We obtained thousands of labeled Color Fundus Retinal Photographs (CFPs) from Kaggle(2) for training and classification. Initially, the predictions were about 20% accurate. To improve results we balanced the dataset, pre-processed, trained all layers of the model, and searched for Hyper-Parameters using HyperOpt(6). 
Finally, using Android Studio we built an app to run on an Android phone to continuously scan and predict the Diabetic Retinopathy level. 

We were able to successfully build an application to predict diabetic retinopathy. We are confident that additional research, training with larger high-quality data, and improved models can yield even more accurate results and potentially be integrated with the camera and deployed to prevent blindness.

Introduction
As per IAPB (International Agency for the Prevention of Blindness), in 2015, out of 415 million people living with diabetes over 145 million had diabetic retinopathy (DR), and that number is expected to grow to 224 million by the year 2040(1). Early identification and treatment can prevent almost all blindness; however, DR often remains undetected until it gets too severe.  Ophthalmologists examine CFPs to document the presence, progression, and severity of disorders. Unfortunately, a large population does not have access to these experts and hence many patients remain undiagnosed and lose their vision.
 
In this project, we evaluated if deep learning, specifically convolutional neural networks (CNN) can be used for assessment and prediction of diabetic retinopathy. The purpose of this project was to focus on three key areas to get optimal results  – Data sets, deep learning models, and AI inference.
Datasets: We used Retinal Photographs and 5 classes of animal images as inputs to the models.
Models: We tried numerous models with combinations of feature learning layers (Convolutions and Pooling) and classification layers (flattened, fully connected layers). In addition, we used VGG16(4) and ResNet(5) models to obtain baseline results. Finally, we studied the influence of hyperparameters on overfitting and underfitting training curves as well as stride lengths and epochs. We used Google Colab, Jupyter, Tensorflow, and Python for the development.
AI inference on battery-powered devices: We exported trained models and used the TensorFlow Lite interpreter. To perform inference we used an Android phone camera to capture the image of the retina and passed that to the model as input. The model then showed an array of probabilities between 0 and 1 for each class or level of the DR. We used TensorFlow Lite(7) Java API to perform the Inference.



