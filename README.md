## Image Classification using Deep Learning

This project implements an image classification model using a Convolutional Neural Network (CNN) in Python. The model is trained to classify images of fruits and vegetables into multiple categories using the Fruits-360 dataset. It demonstrates a complete deep learning pipeline from data preprocessing to model evaluation.


### Project Objectives

- Develop a deep learning model using CNN for image classification
- Preprocess and augment image data for improved performance
- Train and evaluate the model with performance metrics
- Visualize accuracy and loss during training
- Save and reuse the trained model for inference


### Dataset

- **Dataset Name**: Fruits-360 (Fruits and Vegetables Image Dataset)
- **Number of Classes**: 131
- **Image Size**: 100x100 pixels
- **Download Link**: [Click here to download from Google Drive](https://drive.google.com/file/d/1CGiAWso43GCsNo_faRq4jdDIlmwy7YI4/view?usp=sharing)

>  After downloading, extract the dataset and place the contents inside a `/data/` directory in the project root.


### Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn
- Google Colab / Jupyter Notebook

### Model Architecture

The model uses a typical CNN structure, consisting of:

-  Convolutional Layers (for feature extraction)
-  MaxPooling Layers (for downsampling)
-  Dropout Layers (to reduce overfitting)
-  Flatten Layer (to transition from 2D to 1D)
-  Fully Connected Dense Layers
-  Softmax Output Layer (for multi-class classification)


### Model Performance

- Accuracy: 92.5%

> Training and validation accuracy/loss graphs are included in the notebook.
