# MathGlyph ( HandWritten Digit Recognition)
## Live Link - https://mathglyphs.onrender.com/

This project is a deep learning-based web application designed to recognize handwritten digits (0-9) from user inputs. It leverages Convolutional Neural Networks (CNN) for accurate digit classification, ensuring reliable predictions. The application features an interactive interface built using Streamlit, deployed seamlessly on Render for real-time access.

## Technologies Used:
- Python
- Data Preprocessing & Feature Engineering
- Deep Learning Algorithms
- Convolutional Neural Network (CNN)-based model
- Streamlit (Web Interface)
- Render (deployment)

## Introduction
The Handwritten Digit Recognition project aims to classify digits drawn or uploaded by users. Using a robust CNN-based model trained on the popular MNIST dataset, the application ensures high accuracy for recognizing digits. This tool is beneficial for digit classification tasks in various domains, such as education, banking, and postal services.

## Objectives
- To accurately recognize handwritten digits (0-9) using deep learning techniques.
- To provide an intuitive and user-friendly interface for digit input (draw or upload).
- To ensure a robust and scalable model with consistent performance across datasets.

## Features
- High Accuracy: Built using a CNN-based architecture optimized for digit recognition.
- Interactive Web Interface: Users can draw or upload handwritten digits and receive instant predictions.
- Real-Time Results: The deployed application ensures quick and precise predictions.
- Model Optimization: Includes advanced training techniques, dropout layers, and optimizers to improve accuracy and prevent overfitting.

## Models/Algorithms
- Convolutional Neural Network (CNN):
- Optimizer: Adam for adaptive learning rates.
- Activation Functions: ReLU and Softmax for non-linearity and classification.
- Dataset: MNIST (Modified National Institute of Standards and Technology) dataset, containing 60,000 training and 10,000 testing images of handwritten digits.

## Installation
To set up this project locally, follow these steps:

1.Clone this repository by running git clone : https://github.com/aniketkumar101/MathGlyphs.git

2.Install the required dependencies:
Use the provided requirements.txt file to install dependencies : pip install -r requirements.txt

3.Run the application:
Navigate to the project directory and start the Streamlit application : streamlit run app.py

## Tools
- Python: Programming language for model development.
- Matplotlib/Seaborn: Visualization tools for analyzing training and testing performance.
- Keras/TensorFlow: Frameworks for building and training the CNN model.
- Streamlit: For creating an interactive web-based interface.
- Render: Deployment platform for hosting the application.

## Conclusion
The Handwritten Digit Recognition project showcases the power of deep learning in image classification tasks. Its accurate and efficient CNN-based model makes it ideal for practical applications in digit recognition. The interactive interface provides a seamless experience for users, making this tool a valuable resource for educational and commercial purposes.
