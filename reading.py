import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

path = 'Data/genres_original/'
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def loading(namefile):
    # print("Loading the audio file...")
    y, sr = librosa.load(namefile, sr=None) # Default sampling rate
    # print("Loading done")
    return y, sr


def spectrum(y):
    S = np.abs(librosa.stft(y))  # Magnitude of the spectrogram
    mean_frequency = np.mean(S, axis=1)  # Mean as a function of frequency
    std_frequency = np.std(S, axis=1)    # Standard deviation as a function of frequency
    return mean_frequency, std_frequency,S


if __name__ == "__main__":
    print("Loading the audio file...")
    y, sr = loading('Data/genres_original/blues/blues.00000.wav')
    print("Loading done")
    S,mean_frequency, std_frequency = spectrum(y)

    # Spectrogram calculation
    S = np.abs(librosa.stft(y))  # Amplitude spectrogram
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (dB) for a blues extract')
    plt.show()

    # Display the mean as a function of frequency
    plt.figure()
    plt.plot(mean_frequency)
    plt.title('Mean as a function of frequency')
    plt.xlabel('Frequency (index)')
    plt.ylabel('Mean amplitude')
    plt.xscale('log')  # Use logarithmic scale for frequencies
    plt.show()

    # Display the standard deviation as a function of frequency
    plt.figure()
    plt.plot(std_frequency)
    plt.title('Standard deviation as a function of frequency')
    plt.xlabel('Frequency (index)')
    plt.ylabel('Standard deviation')
    plt.xscale('log')  # Use logarithmic scale for frequencies
    plt.show()