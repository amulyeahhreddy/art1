import numpy as np
import pygame

# Create a simple sine wave tone that pygame can play
def create_tone(frequency=440, duration=1.0, sample_rate=22050):
    frames = int(duration * sample_rate)
    arr = np.zeros((frames, 2), dtype=np.int16)
    for i in range(frames):
        t = float(i) / sample_rate
        val = int(32767.0 * np.sin(2.0 * np.pi * frequency * t))
        arr[i] = [val, val]
    return arr

# Create a simple melody
pygame.mixer.init()

# Create different tones for a simple pattern
tones = [
    create_tone(261.63, 0.5),  # C
    create_tone(293.66, 0.5),  # D
    create_tone(329.63, 0.5),  # E
    create_tone(261.63, 0.5),  # C
    create_tone(329.63, 0.5),  # E
    create_tone(392.00, 0.5),  # G
    create_tone(261.63, 1.0),  # C (longer)
]

# Combine all tones
full_melody = np.concatenate(tones)

# Save as WAV file
sound = pygame.sndarray.make_sound(full_melody)
pygame.mixer.Sound.save(sound, "simple_background.wav")

print("Created simple_background.wav - a basic tone loop for testing")
