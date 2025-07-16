## Music Genre Classification using ML & Deep Learning

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
- [Results & Analysis](#results--analysis)
- [Future Improvements](#future-improvements)

## Project Goal
The primary objective of this project is to build an accurate music genre classifier. The project explores the effectiveness of both classical machine learning algorithms and more complex deep learning architectures on features extracted from audio signals. This comparative analysis helps in understanding the strengths of different modeling approaches for audio classification tasks.

## Features
- **Efficient Data Processing**: Audio feature extraction is performed only once. The resulting dataset is cached as `dataset.csv` for significantly faster subsequent runs.
- **Multiple Model Comparison**: Implements and evaluates a wide range of classifiers:
  - **Support Vector Machines (SVM)**: with both "One-vs-One" and "One-vs-Rest" strategies.
  - **Tree-Based Models**: Decision Tree and Random Forest.
  - **Deep Neural Network (DNN)**: A fully-connected multi-layer perceptron (MPP).
  - **Convolutional Neural Network (CNN)**: A 1D CNN tailored for sequence data.
- **Interactive Training**: A command-line interface allows you to choose which model to train and evaluate.
- **Advanced Deep Learning Integration**:
  - **TensorBoard Logging**: For visualizing training metrics, model graphs, and performance.
  - **Model Checkpointing**: Automatically saves the best-performing model during training.
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

1.  **Feature Extraction**: The `librosa` library is used to load each `.wav` file. For each file, the Short-Time Fourier Transform (STFT) is computed. The mean and standard deviation of the resulting spectrogram are used as features.
2.  **Data Caching**: The extracted features and their corresponding labels are stored in `dataset.csv` to avoid redundant processing.
3.  **Data Preparation**: The dataset is split into training (80%) and testing (20%) sets. Genre labels are numerically encoded using `LabelEncoder`.
4.  **Model Training**: Based on the user's interactive choice, a model is trained on the training data.
5.  **Model Evaluation**: The model's performance is evaluated on the unseen test set. For deep learning models, accuracy and loss history are plotted and saved.

## Technologies Used
- **Python 3**
- **TensorFlow & Keras**: For building and training the DNN and CNN models.
- **Scikit-learn**: For classical ML models and data preprocessing.
- **Librosa**: For audio processing and feature extraction.
- **Pandas & NumPy**: For data manipulation and numerical operations.
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
    - Download the GTZAN Genre Collection dataset from a source like [Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).
    - Unzip the file and place the `genres_original` folder inside a `Data` directory at the root of the project. The final path should be: `.../Music-classification/Data/genres_original/`.

## Usage
To run the project, execute the main training script from the root directory:
```bash
python creation_prediction_training.py
```
You will be prompted with an interactive menu to select the model you wish to train:
```
SVM(1), Tree(2), NN(3), CNN(4) or finish(other):
```
Follow the on-screen prompts. Any generated plots or logs will be saved in the `plots/` and `logs/` directories.

## Results & Analysis

The performance of the deep learning models was tracked over 100 epochs.




| Model               | Validation Accuracy | Observations                                           |
| :------------------ | :-----------------: | :----------------------------------------------------- |
| **MPP (DNN) Model** |     **~73%**        | Shows strong learning but significant overfitting.     |
| **CNN Model**       |     **~66%**        | Overfits heavily after epoch 20.                       |
| Random Forest       | *(TBD)*             | *To be determined by running the script.*              |
| SVM (One-vs-Rest)   | *(TBD)*             | *To be determined by running the script.*              |

### Analysis
Both the MPP (DNN) and CNN models show clear signs of **overfitting**. The training accuracy (blue line) reaches nearly 100%, while the validation accuracy (orange line) stagnates at a much lower level. This indicates that the models have memorized the training data very well but struggle to generalize to new, unseen data.

Interestingly, the simpler MPP model achieved a higher validation accuracy (~73%) than the more complex CNN (~66%). This could suggest that the features extracted (mean and std of the spectrogram) may not have enough spatial information for the CNN to be effective, or that the CNN architecture itself requires further tuning and regularization.

## Future Improvements
Based on the results, the following steps would be logical next improvements:

- **Combat Overfitting**:
  - Implement **Dropout layers** in the DNN and CNN to randomly deactivate neurons during training.
  - Use **Early Stopping** to halt training when the validation accuracy stops improving.
  - Add **L1/L2 Regularization** to penalize large weights in the models.
- **Advanced Feature Engineering**: Incorporate more sophisticated audio features like **Mel-Frequency Cepstral Coefficients (MFCCs)**, Chroma Features, and Spectral Contrast, which are standard in audio classification.
- **Hyperparameter Tuning**: Use techniques like KerasTuner or Optuna to systematically find the optimal set of hyperparameters (e.g., learning rate, number of layers, filter sizes) for the deep learning models.
- **Deployment**: Build a simple web application using **Flask** or **Streamlit** where a user can upload a `.wav` file and get a genre prediction in real-time.
