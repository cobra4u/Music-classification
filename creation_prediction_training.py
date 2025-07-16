from reading import *
from NN import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import os
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow as tf
import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

hparams = {  # Hyperparameters to design
    'hidden_units': 256,
    'num_layers': 4,
    'activation': tf.nn.relu,
    'epochs': 100,
    'optimizer': 'adam',
    'loss': 'sparse_categorical_crossentropy'
}

def save_accuracy_plot(history, model_name): # Save the accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'{plot_dir}/{model_name}_accuracy.png')
    plt.close()

# Load the dataset............................................................................................

path = 'Data/genres_original/'
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

feature_list = []
label_list = []

if 'dataset.csv' in os.listdir():         # Load the dataset if it exists
    dataset = pd.read_csv('dataset.csv')
    print('Dataset loaded successfully')

else:                                     # Create the dataset if it does not exist
    for genre in genres:
        for i in range(100):
            file = path + genre + '/' + genre + '.' + str(i).zfill(5) + '.wav'
            y, sr = loading(file)
            print("Extraction of", genre, "at", (i + 1), "%")
            mean_freq, std_freq, S = spectrum(y)
            feature_list.append(mean_freq + std_freq)
            label_list.append(genre)

    dataset = pd.DataFrame(feature_list)
    dataset['label'] = label_list
    dataset.to_csv('dataset.csv', index=False) 

# Split into training and test sets and encode the labels.....................................................

train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

train_features = train_dataset.iloc[:, :-1].values
train_labels = train_dataset.iloc[:, -1].values

test_features = test_dataset.iloc[:, :-1].values
test_labels = test_dataset.iloc[:, -1].values

label_encoder = LabelEncoder()
train_encoded = label_encoder.fit_transform(train_labels)
test_encoded = label_encoder.transform(test_labels)

# Training and prediction.....................................................................................

while True:

    a = input("SVM(1), Tree(2), NN(3), CNN(4) or finish(other):")

    if a == '1': # SVM

        b = input('one vs one (1) or one vs all(other)')
        
        if b == '1': # One vs One

            svm_model = SVC(kernel='linear')
            svm_model.fit(train_features, train_labels)
            y_pred = svm_model.predict(test_features)
            accuracy = accuracy_score(test_labels, y_pred)
            print(f'Accuracy for SVM (one vs one): {accuracy * 100:.2f}%')

        else: # One vs All

            svm_model = SVC(kernel='linear')
            ova_svm_model = OneVsRestClassifier(svm_model)
            ova_svm_model.fit(train_features, train_labels)
            y_pred = ova_svm_model.predict(test_features)
            accuracy = accuracy_score(test_labels, y_pred)
            print(f'Accuracy for SVM (one vs all): {accuracy * 100:.2f}%')

    elif a == '2': # Decision Tree or Random Forest

        b = input('Decision tree(1) or Random Forest(other)')

        if b == '1': # Decision Tree

            decision_tree_model = DecisionTreeClassifier()
            decision_tree_model.fit(train_features, train_labels)
            y_pred = decision_tree_model.predict(test_features)
            accuracy = accuracy_score(test_labels, y_pred)
            print(f'Accuracy for Decision tree: {accuracy * 100:.2f}%')
            
        else: # Random Forest

            random_forest_model = RandomForestClassifier()
            random_forest_model.fit(train_features, train_labels)
            y_pred = random_forest_model.predict(test_features)
            accuracy = accuracy_score(test_labels, y_pred)
            print(f'Accuracy for Random Forest: {accuracy * 100:.2f}%')

    elif a == '3':  # Basic MPP

        # Create the logs and plots directories

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        log_dir = os.path.join('logs/mpp', current_time)
        plot_dir = os.path.join('plots/mpp', current_time)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        checkpoint_callback_mpp = ModelCheckpoint('mpp_model_checkpoint.keras', save_best_only=True, monitor='val_loss', mode='min', verbose=1)

        model = MPP(train_features, train_encoded, hparams)
        history = model.fit(train_features, train_encoded, epochs=hparams['epochs'], validation_data=(test_features, test_encoded), callbacks=[tensorboard_callback,checkpoint_callback_mpp])

        test_loss, test_accuracy = model.evaluate(test_features, test_encoded)
        save_accuracy_plot(history, 'MPP_Model')
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}") 

    elif a == '4':  # Convolutional Neural Network

        # Create the logs and plots directories

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        log_dir = os.path.join('logs/cnn', current_time)
        plot_dir = os.path.join('plots/cnn', current_time)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        checkpoint_callback_cnn = ModelCheckpoint('cnn_model_checkpoint.keras', save_best_only=True, monitor='val_loss', mode='min', verbose=1)

        train_features_cnn = train_features[..., np.newaxis]
        test_features_cnn = test_features[..., np.newaxis]

        cnn_model = CNN(input_shape=train_features_cnn.shape[1:], num_classes=len(label_encoder.classes_))
        history=cnn_model.fit(train_features_cnn, train_encoded, epochs=hparams['epochs'], validation_data=(test_features_cnn, test_encoded), callbacks=[tensorboard_callback,checkpoint_callback_cnn])

        test_loss, test_accuracy = cnn_model.evaluate(test_features_cnn, test_encoded)
        save_accuracy_plot(history, 'CNN_Model')
        print(f"Test Loss (CNN): {test_loss:.4f}")
        print(f"Test Accuracy (CNN): {test_accuracy:.4f}")

    else: 
        print('End')
        break