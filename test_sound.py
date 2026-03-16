import sounddevice as sd
import numpy as np

print("Available devices:")
print(sd.query_devices())
print(f"\nDefault output: {sd.query_devices(kind='output')['name']}")

# Play a 1-second beep directly
sample_rate = 44100
t = np.linspace(0, 1, sample_rate)
tone = (np.sin(2 * np.pi * 440 * t) * 0.5).astype('float32')
tone_stereo = np.stack([tone, tone], axis=1)

print("\nPlaying test tone... (you should hear a beep)")
sd.play(tone_stereo, sample_rate)
sd.wait()
print("Done. Did you hear anything? (y/n)")
