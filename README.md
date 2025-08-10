# Deep Learning Projects

This repository contains a collection of deep learning projects implemented using Python and popular libraries like PyTorch. Each project explores a different application of deep learning, from computer vision to natural language processing.

---

### 1. Facial Emotion Recognition

*   **`Facial_Emotion_Recognition_CNN.ipynb`**: This Jupyter Notebook implements a Convolutional Neural Network (CNN) to recognize human facial emotions. The model is trained on the FER2013 dataset to classify faces into one of seven emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise, or Neutral.

*   **`Opencv Emotion Detection/`**: This directory contains a real-time implementation of the emotion recognition model.
    *   **`Emotion_recognition_opencv.py`**: A Python script that uses OpenCV to capture video from a webcam, detect faces using a Haar Cascade classifier, and then uses the trained CNN model to predict and display the emotion of the detected face in real-time.
    *   **`emotion_cnn.pth`**: The saved, trained PyTorch model weights for the emotion recognition CNN.
    *   **`haarcascade_frontalface_default.xml`**: An OpenCV Haar Cascade file used for detecting frontal faces in an image or video stream.

Dataset: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

### 2. Fake News Detection

*   **`Fake_News_Detection_(NLP_RNN_LSTM).ipynb`**: This notebook tackles the problem of fake news detection using Natural Language Processing (NLP). It employs a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) units to classify news articles as either "real" or "fake" based on their textual content.

Dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
<br>Download the two files: Fake.csv and True.csv. Upload them to your Colab environment.

### 3. Handwritten Digit Recognition

*   **`Handwritten_Digit_Recognition_MNIST_CNN.ipynb`**: A classic deep learning project that builds a CNN to recognize handwritten digits. The model is trained on the famous MNIST dataset, which consists of thousands of images of handwritten digits from 0 to 9.

### 4. Human Activity Recognition

*   **`Human_Activity_Recognition_SensorML.ipynb`**: This project focuses on recognizing human activities (e.g., walking, sitting, standing) based on sensor data. The notebook uses a hybrid model combining a CNN and an LSTM to effectively capture both spatial and temporal features from the sensor readings.

Dataset: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones <br>
 Download the data from the UCI HAR Dataset page and unzip it into your project directory. Upload the unzipped UCI HAR Dataset folder to your Colab environment.

### 5. Legal Document Classification

*   **`Legal_Document_Classification_Fine_tuning_DistilBERT.ipynb`**: This notebook demonstrates how to fine-tune a pre-trained DistilBERT model for the specific task of classifying legal documents. This is a powerful technique for achieving high accuracy on specialized NLP tasks without training a model from scratch.

### 6. Stock Price Trend Prediction

*   **`Stock_Price_Trend_Prediction(RNN_LSTM).ipynb`**: An application of time-series analysis, this notebook uses an RNN with LSTM cells to predict future stock price trends. The model is trained on historical stock data to learn patterns and make forecasts.

This project requires a few more libraries specifically for data handling (pandas, yfinance) and processing (scikit-learn)

---

