# Diabetes Prediction Using Artificial Neural Networks (ANN)

This project aims to predict whether a patient has diabetes using an Artificial Neural Network (ANN) based on diagnostic data. The dataset used is the well-known [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database).

## Project Overview
The goal of this project is to use an ANN to predict whether a patient has diabetes based on their medical data. We will build a neural network with multiple layers, using TensorFlow and Keras, to classify the outcome as either diabetic (1) or non-diabetic (0).

## Technologies Used
- Python
- Pandas
- Numpy
- TensorFlow
- Keras
- scikit-learn

## Dataset
The dataset used for this project is the [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database), which consists of 768 samples and the following 8 features:
- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age (years)
- **Outcome**: 0 for non-diabetic, 1 for diabetic

## Model Training
The model architecture consists of:
1. An input layer with 12 neurons corresponding to the 8 features from the dataset.
2. Two hidden layers with 12 and 8 neurons respectively, using the ReLU activation function.
3. An output layer with 1 neuron using the sigmoid activation function to classify the outcome.

The model is trained using the stochastic gradient descent (SGD) optimizer and binary cross-entropy loss function for 100 epochs.

## Evaluation
After training the model, the following metrics are computed:
- **Test Accuracy**: Accuracy of the model on the test dataset.
- **Test Loss**: Loss function value on the test dataset.

