import time
import winsound


class AlarmController:
    def __init__(self, cooldown_seconds: int = 10):
        self.cooldown_seconds = cooldown_seconds
        self.last_alarm_time = 0

    def trigger_alarm(self):
        now = time.time()

        if now - self.last_alarm_time < self.cooldown_seconds:
            return

        self.last_alarm_time = now

        print("ALARM: Snoring detected. Move position.")

        try:
            winsound.Beep(1200, 500)
            winsound.Beep(1600, 500)
        except Exception as e:
            print(f"Alarm failed: {e}")