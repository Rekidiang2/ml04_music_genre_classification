import numpy as np
import pandas as pd
import librosa, librosa.display
import matplotlib.pyplot as plt

def audio_properties(file, sr=22050):
    signal, sample_rate = librosa.load(file, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(signal, sample_rate)
    print("Some audio properties\n"+'--'*10)
    print(f"|Audio Signal Length : {signal.shape}\n|Sample Rate: {sample_rate}\n|Tempo : {tempo} ")
    bf = pd.DataFrame({"Beat Frame":list(beat_frames)})
    return bf.T


def waveform(file):
    # WAVEFORM
    # display waveform
    signal, sample_rate = librosa.load(file, sr=22050)
    FIG_SIZE = (13,5)
    plt.figure(figsize=FIG_SIZE)
    librosa.display.waveplot(signal, sample_rate, alpha=0.4)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    
    
def spectrum(file) :
    signal, sample_rate = librosa.load(file, sr=22050)
    # perform Fourier transform
    fft = np.fft.fft(signal)

    # calculate abs values on complex numbers to get magnitude
    spectrum = np.abs(fft)

    # create frequency variable
    f = np.linspace(0, sample_rate, len(spectrum))

    # take half of the spectrum and frequency
    left_spectrum = spectrum[:int(len(spectrum)/2)]
    left_f = f[:int(len(spectrum)/2)]
    # plot spectrum (entire)
    plt.figure(figsize=(13,5))
    plt.subplot(2,1,1)
    
    plt.plot(f, spectrum, alpha=0.4)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.title("Power spectrum (Entire)")
    
    # plot spectrum (left)
    plt.figure(figsize=(13,5))
    plt.subplot(2,1,2)
    
    plt.plot(left_f, left_spectrum, alpha=0.4)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.title("Power spectrum (left)")
    
def spectogram(file, hop_length=512, n_fft = 2048):
    signal, sample_rate = librosa.load(file, sr=22050)
    # STFT -> spectrogram
    # calculate duration hop length and window in seconds
    hop_length_duration = float(hop_length)/sample_rate
    n_fft_duration = float(n_fft)/sample_rate
    print(f"STFT : Hop Dength Duration = {hop_length_duration} | Window Duration = {n_fft_duration}")
    # perform stft
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    # calculate abs values on complex numbers to get magnitude
    spectrogram = np.abs(stft)
    #Plot
    plt.figure(figsize=(15,8))
    # display spectrogram
    plt.subplot(2,1,1)
    librosa.display.specshow(spectrogram, sr=sample_rate, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar()
    plt.title("Spectrogram")
    
    # display spectrogram
    # apply logarithm to cast amplitude to Decibels
    log_spectrogram = librosa.amplitude_to_db(spectrogram)
    plt.subplot(2,1,2)
    librosa.display.specshow(log_spectrogram, sr=sample_rate, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram (dB)")
    
def mfccs(file, hop_length=512, n_fft = 2048,  n_mfcc=13):
    signal, sample_rate = librosa.load(file, sr=22050)
    MFCCs = librosa.feature.mfcc(signal, sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
    # display MFCCs
    plt.figure(figsize=(16,8))
    librosa.display.specshow(MFCCs, sr=sample_rate, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("MFCC coefficients")
    plt.colorbar()
    plt.title("MFCCs")
    # show plots
    plt.show()
    
#keyword spotting system
import librosa
import os
import json


DATASET_PATH = "data/sub_raw_data"
JSON_PATH = "data/prepared_data.json"
SAMPLES_TO_CONSIDER = 22050 # 1 sec. of audio // sample rate number of sample per second


def preprocess_dataset(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512):
    """Extracts MFCCs from music dataset and saves them into a json file.
    :param dataset_path (str): Path to dataset
    :param json_path (str): Path to json file used to save MFCCs
    :param num_mfcc (int): Number of coefficients to extract
    :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
    :param hop_length (int): Sliding window for FFT. Measured in # of samples
    :return:
    """

    # dictionary where we'll store mapping, labels, MFCCs and filenames
    data = {
        "mapping": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    # loop through all sub-dirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're at sub-folder level
        if dirpath is not dataset_path:

            # save label (i.e., sub-folder name) in the mapping
            label = dirpath.split("/")[-1]
            data["mapping"].append(label)
            print("\nProcessing: '{}'".format(label))

            # process all audio files in sub-dir and store MFCCs
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # load audio file and slice it to ensure length consistency among different files
                signal, sample_rate = librosa.load(file_path)

                # drop audio files with less than pre-decided number of samples
                if len(signal) >= SAMPLES_TO_CONSIDER:

                    # ensure consistency of the length of the signal
                    signal = signal[:SAMPLES_TO_CONSIDER]

                    # extract MFCCs
                    MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                                 hop_length=hop_length)

                    # store data for analysed track
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["labels"].append(i-1)
                    data["files"].append(file_path)
                    print("{}: {}".format(file_path, i-1))

    # save data in json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        
    print(" >>> Data Extracted for Training <<< ")


#if __name__ == "__main__":
   # preprocess_dataset(DATASET_PATH, JSON_PATH)

