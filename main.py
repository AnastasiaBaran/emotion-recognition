print("STARTED")
import os
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

DATASET_PATH = "dataset"

EMOTIONS = [
    "happy",
    "sad",
    "angry",
    "neutral",
    "fear",
    "disgust"
]

SR = 22050
N_MELS = 128

def extract_features(file_path):

    audio, sr = librosa.load(file_path, sr=SR)

    audio = audio / np.max(np.abs(audio))

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    return mel_db

files = []
labels = []

for i, emotion in enumerate(EMOTIONS):

    folder = os.path.join(DATASET_PATH, emotion)

    for file in os.listdir(folder):

        files.append(os.path.join(folder, file))
        labels.append(i)

print(f"Loaded files: {len(files)}")

sample = extract_features(files[0])

print("Mel spectrogram shape:")
print(sample.shape)
print("DONE ")