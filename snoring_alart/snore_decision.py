import numpy as np


class SnoreDecisionEngine:
    def __init__(
        self,
        snore_threshold: float,
        min_audio_rms: float,
        consecutive_windows_to_alarm: int
    ):
        self.snore_threshold = snore_threshold
        self.min_audio_rms = min_audio_rms
        self.consecutive_windows_to_alarm = consecutive_windows_to_alarm
        self.consecutive_snore_count = 0
        self.not_snore_counter = 0
        self.reset_after_not_snore = 120

    def calculate_rms(self, audio_window: np.ndarray) -> float:
        if audio_window is None or len(audio_window) == 0:
            return 0.0

        return float(np.sqrt(np.mean(np.square(audio_window))))

    def update(self, snore_score: float, audio_window: np.ndarray):
        rms = self.calculate_rms(audio_window)
        is_snore = snore_score >= self.snore_threshold

        should_alarm = False  # ✅ always defined

        if rms < self.min_audio_rms:
            self.not_snore_counter += 1

        
        if is_snore:
            print(f"Snore detected! Score: {snore_score:.3f}, RMS: {rms:.3f}")
            self.consecutive_snore_count += 1
            self.not_snore_counter = 0
        else:
            self.not_snore_counter += 1

        if self.not_snore_counter >= self.reset_after_not_snore:
            self.consecutive_snore_count = 0
            
        if self.not_snore_counter >= self.reset_after_not_snore:
            self.consecutive_snore_count = 0

            return {
                "state": "NO_AUDIO_OR_SILENCE",
                "rms": rms,
                "snore_score": snore_score,
                "consecutive_snore_count": self.consecutive_snore_count,
                "not_snore_counter": self.not_snore_counter,
                "should_alarm": False
            }



        should_alarm = self.consecutive_snore_count >= self.consecutive_windows_to_alarm

        return {
            "state": "SNORE" if is_snore else "NOT_SNORE",
            "rms": rms,
            "snore_score": snore_score,
            "consecutive_snore_count": self.consecutive_snore_count,
            "not_snore_counter": self.not_snore_counter,
            "should_alarm": should_alarm
        }