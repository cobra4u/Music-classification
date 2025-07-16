Table of Contents:
Project Goal
Features
Dataset
Methodology
Technologies Used
Setup & Installation
Usage
Results
Future Improvements
Project Goal
The primary objective of this project is to build an accurate music genre classifier. The project explores the effectiveness of both classical machine learning algorithms and more complex deep learning architectures on features extracted from audio signals. This comparative analysis helps in understanding the strengths of different modeling approaches for audio classification tasks.
Features
Efficient Data Processing: Audio feature extraction is performed only once. The resulting dataset is cached as dataset.csv for significantly faster subsequent runs.
Multiple Model Comparison: Implements and evaluates a wide range of classifiers:
Support Vector Machines (SVM): with both "One-vs-One" and "One-vs-Rest" strategies.
Tree-Based Models: Decision Tree and Random Forest.
Deep Neural Network (DNN): A fully-connected multi-layer perceptron.
Convolutional Neural Network (CNN): A 1D CNN tailored for sequence data.
Interactive Training: A command-line interface allows you to choose which model to train and evaluate.
Advanced Deep Learning Integration:
TensorBoard Logging: For visualizing training metrics, model graphs, and performance.
Model Checkpointing: Automatically saves the best-performing model during training to prevent overfitting and save progress.
Performance Visualization: Automatically generates and saves accuracy plots for deep learning models.
Dataset
This project uses the famous GTZAN Genre Collection dataset. It consists of 1000 audio tracks, each 30 seconds long. There are 100 tracks for each of the 10 genres:
Blues
Classical
Country
Disco
Hiphop
Jazz
Metal
Pop
Reggae
Rock
Note: You will need to download this dataset separately and place it in the correct directory as described in the installation instructions.
Methodology
The project follows a standard machine learning workflow:
Feature Extraction: The librosa library is used to load each .wav file. For each file, the Short-Time Fourier Transform (STFT) is computed. The mean and standard deviation of the resulting spectrogram are used as features, representing the audio's frequency profile.
Data Caching: The extracted features and their corresponding labels are stored in dataset.csv to avoid redundant processing.
Data Preparation: The dataset is split into training (80%) and testing (20%) sets. The genre labels are numerically encoded using LabelEncoder.
Model Training: Based on the user's interactive choice, one of the implemented models is trained on the training data.
Model Evaluation: The trained model's performance is evaluated on the unseen test set, with accuracy being the primary metric. For neural networks, loss is also reported, and performance plots are saved.
Technologies Used
Python 3
TensorFlow & Keras: For building and training the DNN and CNN models.
Scikit-learn: For classical ML models (SVM, Decision Tree, Random Forest) and data preprocessing (train_test_split, LabelEncoder).
Librosa: For audio processing and feature extraction.
Pandas: For data manipulation and management.
NumPy: For numerical operations.
Matplotlib: For plotting results.
Setup & Installation
Follow these steps to set up the project environment:
Clone the repository:
Generated bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Use code with caution.
Bash
Create and activate a virtual environment:
Generated bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
Use code with caution.
Bash
Install the required libraries:
Generated bash
pip install -r requirements.txt
Use code with caution.
Bash
Download the Data:
Download the GTZAN Genre Collection dataset (e.g., from Kaggle or the official source).
Unzip the file and place the genres_original folder inside a Data directory at the root of the project. The final path structure should be: .../your-repo-name/Data/genres_original/.
Usage
To run the project, execute the main training script from the root directory:
Generated bash
python creation_prediction_training.py
Use code with caution.
Bash
You will be prompted with an interactive menu to select the model you wish to train:
Generated code
SVM(1), Tree(2), NN(3), CNN(4) or finish(other):
Use code with caution.
Follow the on-screen prompts to select sub-options if available. The results will be printed to the console, and any generated plots or logs will be saved in the plots and logs directories, respectively.

**Results**
The performance of each model is measured by its accuracy on the test set. Here is a sample table of expected results:
Model	Test Accuracy
SVM (One-vs-Rest)	~55% - 65%
Random Forest	~60% - 70%
DNN (MPP)	~65% - 75%
1D CNN	~70% - 80%
(Note: These are estimates. Your actual results may vary. The CNN is expected to perform best due to its ability to capture spatial hierarchies in sequence data).
Training logs for the deep learning models can be viewed using TensorBoard:
Generated bash
tensorboard --logdir logs
Use code with caution.
Bash

**Future Improvements**
Advanced Feature Engineering: Incorporate more sophisticated audio features like Mel-Frequency Cepstral Coefficients (MFCCs), Chroma Features, and Spectral Contrast.
Hyperparameter Tuning: Use techniques like Grid Search, Random Search, or KerasTuner to find the optimal hyperparameters for each model.
Data Augmentation: Create more training data by applying transformations to the audio files (e.g., adding noise, time-stretching, pitch-shifting) to improve model robustness.
Deployment: Build a simple web interface using Flask or Streamlit where a user can upload a .wav file and get a genre prediction in real-time.
