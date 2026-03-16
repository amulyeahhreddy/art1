import numpy as np
import wave
import struct

# Create a simple sine wave tone
def create_simple_wav():
    sample_rate = 44100
    duration = 2.0  # 2 seconds
    frequency = 440  # A4 note
    
    # Generate sine wave
    num_samples = int(sample_rate * duration)
    samples = []
    
    for i in range(num_samples):
        t = float(i) / sample_rate
        value = int(32767 * np.sin(2 * np.pi * frequency * t))
        samples.append(value)
    
    # Create WAV file
    wav_file = wave.open('simple_background.wav', 'w')
    wav_file.setnchannels(1)  # Mono
    wav_file.setsampwidth(2)  # 16-bit
    wav_file.setframerate(sample_rate)
    
    # Write samples
    for sample in samples:
        wav_file.writeframes(struct.pack('<h', sample))
    
    wav_file.close()
    print("Created simple_background.wav")

if __name__ == "__main__":
    create_simple_wav()
