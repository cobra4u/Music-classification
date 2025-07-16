# Music Genre Classification using ML & Deep Learning

This project implements and compares various machine learning and deep learning models to classify music tracks into 10 different genres based on audio features. It serves as a comprehensive pipeline, from audio feature extraction to model training, evaluation, and logging.

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg)](https://scikit-learn.org/stable/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents
- [Project Goal](#project-goal)
- [Features](#features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Technologies Used](#technologies-used)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)

## Project Goal
The primary objective of this project is to build an accurate music genre classifier. The project explores the effectiveness of both classical machine learning algorithms and more complex deep learning architectures on features extracted from audio signals. This comparative analysis helps in understanding the strengths of different modeling approaches for audio classification tasks.

## Features
- **Efficient Data Processing**: Audio feature extraction is performed only once. The resulting dataset is cached as `dataset.csv` for significantly faster subsequent runs.
- **Multiple Model Comparison**: Implements and evaluates a wide range of classifiers:
  - **Support Vector Machines (SVM)**: with both "One-vs-One" and "One-vs-Rest" strategies.
  - **Tree-Based Models**: Decision Tree and Random Forest.
  - **Deep Neural Network (DNN)**: A fully-connected multi-layer perceptron.
  - **Convolutional Neural Network (CNN)**: A 1D CNN tailored for sequence data.
- **Interactive Training**: A command-line interface allows you to choose which model to train and evaluate.
- **Advanced Deep Learning Integration**:
  - **TensorBoard Logging**: For visualizing training metrics, model graphs, and performance.
  - **Model Checkpointing**: Automatically saves the best-performing model during training to prevent overfitting and save progress.
  - **Performance Visualization**: Automatically generates and saves accuracy plots for deep learning models.

## Dataset
This project uses the famous **GTZAN Genre Collection** dataset. It consists of 1000 audio tracks, each 30 seconds long. There are 100 tracks for each of the 10 genres:
- Blues
- Classical
- Country
- Disco
- Hiphop
- Jazz
- Metal
- Pop
- Reggae
- Rock

*Note: You will need to download this dataset separately and place it in the correct directory as described in the installation instructions.*

## Methodology
The project follows a standard machine learning workflow:

1.  **Feature Extraction**: The `librosa` library is used to load each `.wav` file. For each file, the Short-Time Fourier Transform (STFT) is computed. The mean and standard deviation of the resulting spectrogram are used as features, representing the audio's frequency profile.
2.  **Data Caching**: The extracted features and their corresponding labels are stored in `dataset.csv` to avoid redundant processing.
3.  **Data Preparation**: The dataset is split into training (80%) and testing (20%) sets. The genre labels are numerically encoded using `LabelEncoder`.
4.  **Model Training**: Based on the user's interactive choice, one of the implemented models is trained on the training data.
5.  **Model Evaluation**: The trained model's performance is evaluated on the unseen test set, with accuracy being the primary metric. For neural networks, loss is also reported, and performance plots are saved.

## Technologies Used
- **Python 3**
- **TensorFlow & Keras**: For building and training the DNN and CNN models.
- **Scikit-learn**: For classical ML models (SVM, Decision Tree, Random Forest) and data preprocessing (`train_test_split`, `LabelEncoder`).
- **Librosa**: For audio processing and feature extraction.
- **Pandas**: For data manipulation and management.
- **NumPy**: For numerical operations.
- **Matplotlib**: For plotting results.

## Setup & Installation

Follow these steps to set up the project environment:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/cobra4u/Music-classification.git
    cd Music-classification
    ```
2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```
3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download the Data:**
    - Download the GTZAN Genre Collection dataset (e.g., from [Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)).
    - Unzip the file and place the `genres_original` folder inside a `Data` directory at the root of the project. The final path structure should be: `.../Music-classification/Data/genres_original/`.

## Usage
To run the project, execute the main training script from the root directory:
```bash
python creation_prediction_training.py
