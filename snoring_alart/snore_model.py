
import csv
import urllib.request
import numpy as np
import tensorflow_hub as hub

from audio_utils import SAMPLE_RATE, load_wav_16k_mono


WINDOW_SECONDS = 1.0
THRESHOLD = 0.35


def load_yamnet_model():
    print("Loading YAMNet...")
    return hub.load("https://tfhub.dev/google/yamnet/1")


def load_class_names():
    class_map_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
    csv_path = "yamnet_class_map.csv"

    urllib.request.urlretrieve(class_map_url, csv_path)

    class_names = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_names.append(row["display_name"])

    return class_names


def predict_snoring_windows(audio_path: str):
    model = load_yamnet_model()
    class_names = load_class_names()

    snore_index = class_names.index("Snoring")

    audio = load_wav_16k_mono(audio_path)

    window_size = int(SAMPLE_RATE * WINDOW_SECONDS)
    num_windows = max(1, int(np.ceil(len(audio) / window_size)))

    results = []
    snore_windows = 0

    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        window = audio[start:end]

        if len(window) < window_size:
            window = np.pad(window, (0, window_size - len(window)))

        scores, embeddings, spectrogram = model(window)
        mean_scores = np.mean(scores.numpy(), axis=0)

        snore_score = float(mean_scores[snore_index])
        is_snore = snore_score >= THRESHOLD

        if is_snore:
            snore_windows += 1

        results.append({
            "start_sec": i,
            "end_sec": i + 1,
            "snore_score": snore_score,
            "is_snore": is_snore
        })

    summary = {
        "total_windows": num_windows,
        "snore_windows": snore_windows,
        "snore_ratio": snore_windows / num_windows
    }

    return results, summary