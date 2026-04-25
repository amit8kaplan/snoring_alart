import csv
import urllib.request
import numpy as np
import tensorflow_hub as hub
import scipy.io.wavfile as wavfile
from scipy.signal import resample_poly
import os

print("Loading YAMNet...")
model = hub.load("https://tfhub.dev/google/yamnet/1")

class_map_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
csv_path = "yamnet_class_map.csv"
urllib.request.urlretrieve(class_map_url, csv_path)

class_names = []
with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        class_names.append(row["display_name"])

snore_index = class_names.index("Snoring")


def load_wav_16k_mono(filename):
    sample_rate, data = wavfile.read(filename)

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if data.dtype != np.float32:
        data = data.astype(np.float32)

        if np.max(np.abs(data)) > 1:
            data = data / np.max(np.abs(data))

    if sample_rate != 16000:
        data = resample_poly(data, 16000, sample_rate)

    return data.astype(np.float32)


base_dir = os.path.dirname(os.path.abspath(__file__))
audio_path = os.path.join(base_dir, "..", "snore_in.wav")

audio = load_wav_16k_mono(audio_path)

scores, embeddings, spectrogram = model(audio)
mean_scores = np.mean(scores.numpy(), axis=0)

top_indexes = np.argsort(mean_scores)[::-1][:10]

print("\nTop predictions:")
for i in top_indexes:
    print(f"{class_names[i]}: {mean_scores[i]:.3f}")

print("\nSnoring score:")
print(mean_scores[snore_index])
