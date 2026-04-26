import csv
import urllib.request
import numpy as np
import sounddevice as sd
import tensorflow_hub as hub
import keyboard

from realtime_config import (
    SAMPLE_RATE,
    WINDOW_SECONDS,
    CHANNELS,
    SNORE_THRESHOLD,
    MIN_AUDIO_RMS,
    CONSECUTIVE_SNORE_WINDOWS_TO_ALARM,
    ALARM_COOLDOWN_SECONDS
)

from snore_decision import SnoreDecisionEngine
from alarm_utils import AlarmController


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


def get_snore_score(model, class_names, audio_window: np.ndarray) -> float:
    snore_index = class_names.index("Snoring")

    audio_window = audio_window.astype(np.float32)

    if len(audio_window.shape) > 1:
        audio_window = np.squeeze(audio_window)

    scores, embeddings, spectrogram = model(audio_window)
    mean_scores = np.mean(scores.numpy(), axis=0)

    return float(mean_scores[snore_index])


def record_audio_window() -> np.ndarray:
    samples = int(SAMPLE_RATE * WINDOW_SECONDS)

    try:
        audio = sd.rec(
            samples,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32"
        )
        sd.wait()

        audio = np.squeeze(audio)

        if audio is None or len(audio) == 0:
            return np.zeros(samples, dtype=np.float32)

        return audio.astype(np.float32)

    except Exception as e:
        print(f"Audio input error: {e}")
        return np.zeros(samples, dtype=np.float32)


def main():
    model = load_yamnet_model()
    class_names = load_class_names()

    decision_engine = SnoreDecisionEngine(
        snore_threshold=SNORE_THRESHOLD,
        min_audio_rms=MIN_AUDIO_RMS,
        consecutive_windows_to_alarm=CONSECUTIVE_SNORE_WINDOWS_TO_ALARM
    )

    alarm = AlarmController(cooldown_seconds=ALARM_COOLDOWN_SECONDS)

    print("\nReal-time snoring detector started.")
    print("Press Ctrl+C to stop.")
    print("----------------------------------")

    try:
        while True:
            audio_window = record_audio_window()

            snore_score = 0.0
            rms = decision_engine.calculate_rms(audio_window)

            if rms >= MIN_AUDIO_RMS:
                snore_score = get_snore_score(model, class_names, audio_window)

            result = decision_engine.update(snore_score, audio_window)

            print(
                f'STATE={result["state"]:18s} | '
                f'RMS={result["rms"]:.5f} | '
                f'score={result["snore_score"]:.3f} | '
                f'count={result["consecutive_snore_count"]} | '
                f'not_snore_count={result["not_snore_counter"]}'
            )

            if result["should_alarm"]:
                alarm.trigger_alarm()

            if keyboard.is_pressed("q"):
                print("Stopping...")
                break

    except KeyboardInterrupt:
        print("\nStopped real-time detector.")


if __name__ == "__main__":
    main()