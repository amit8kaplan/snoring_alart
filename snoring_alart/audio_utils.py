import os
import shutil
import uuid
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import resample_poly
from pydub import AudioSegment

SAMPLE_RATE = 16000

FFMPEG_PATH = r"C:\Users\amit8\ffmpeg-2026-04-16-git-5abc240a27-essentials_build\bin\ffmpeg.exe"
AudioSegment.converter = FFMPEG_PATH


def detect_audio_file_type(file_path: str) -> str:
    file_path = file_path.strip().strip('"')

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = os.path.splitext(file_path)[1].lower().replace(".", "")

    if extension == "":
        raise ValueError("File has no extension")

    return extension


def copy_and_convert_user_audio_to_wav(
    user_path: str,
    destination_folder: str = "input_audio"
) -> str:
    user_path = user_path.strip().strip('"')

    if not os.path.isfile(user_path):
        raise FileNotFoundError(f"File not found: {user_path}")

    os.makedirs(destination_folder, exist_ok=True)

    file_type = detect_audio_file_type(user_path)
    print(f"Detected file type: {file_type}")

    unique_name = f"audio_{uuid.uuid4().hex}.wav"
    destination_wav_path = os.path.join(destination_folder, unique_name)

    if file_type == "wav":
        shutil.copy2(user_path, destination_wav_path)
    else:
        audio = AudioSegment.from_file(user_path, format=file_type)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(SAMPLE_RATE)
        audio.export(destination_wav_path, format="wav")

    return destination_wav_path


def load_wav_16k_mono(filename: str) -> np.ndarray:
    sample_rate, data = wavfile.read(filename)

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    data = data.astype(np.float32)

    max_value = np.max(np.abs(data))
    if max_value > 1:
        data = data / max_value

    if sample_rate != SAMPLE_RATE:
        data = resample_poly(data, SAMPLE_RATE, sample_rate)

    return data.astype(np.float32)