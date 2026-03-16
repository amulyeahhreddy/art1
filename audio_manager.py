import threading
import time
from playsound import playsound


class AudioManager:
    def __init__(self):
        self._thread = None
        self._stop_flag = threading.Event()
        self._pause_flag = threading.Event()
        self._pause_flag.set()  # not paused by default
        self._volume = 1.0
        self.is_muted = False
        self.is_playing = False
        self._current_file = None
        self._loop = False

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, value):
        self.set_volume(value)

    def load_and_play(self, audio_file, loop=True):
        self.stop()
        self._current_file = audio_file
        self._loop = loop
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._play_loop, daemon=True)
        self._thread.start()
        self.is_playing = True
        print(f"▶️  Playing: {audio_file}")

    def _play_loop(self):
        while not self._stop_flag.is_set():
            self._pause_flag.wait()  # blocks here if paused
            if self._stop_flag.is_set():
                break
            try:
                playsound(self._current_file, block=True)
            except Exception as e:
                print(f"❌ Playback error: {e}")
                break
            if not self._loop:
                break
        self.is_playing = False

    def pause(self):
        self._pause_flag.clear()
        print("⏸️  Paused")

    def resume(self):
        self._pause_flag.set()
        print("▶️  Resumed")

    def toggle_pause(self):
        if self._pause_flag.is_set():
            self.pause()
        else:
            self.resume()

    def stop(self):
        self._stop_flag.set()
        self._pause_flag.set()  # unblock thread so it can exit
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self.is_playing = False
        print("⏹️  Stopped")

    def set_volume(self, volume):
        self._volume = max(0.0, min(1.0, volume))
        print(f"🔊 Volume: {int(self._volume * 100)}%")
        # Note: playsound doesn't support runtime volume change
        # For volume control, we recommend converting to WAV with adjusted amplitude

    def volume_up(self, step=0.1):
        self.set_volume(self._volume + step)

    def volume_down(self, step=0.1):
        self.set_volume(self._volume - step)

    def toggle_mute(self):
        self.is_muted = not self.is_muted
        print(f"{'🔇 Muted' if self.is_muted else f'🔊 Unmuted ({int(self._volume * 100)}%)'}")

    def cleanup(self):
        self.stop()
        print("🧹 Audio cleaned up")
