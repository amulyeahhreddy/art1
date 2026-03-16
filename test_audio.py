import pygame
import numpy as np
import wave
import struct

def create_loud_wav():
    """Create a loud test audio file"""
    sample_rate = 44100
    duration = 1.0  # 1 second
    frequency = 880  # Higher pitch A5 note
    
    # Generate louder sine wave
    num_samples = int(sample_rate * duration)
    samples = []
    
    for i in range(num_samples):
        t = float(i) / sample_rate
        # Use full amplitude for maximum volume
        value = int(32767 * np.sin(2 * np.pi * frequency * t))
        samples.append(value)
    
    # Create WAV file
    wav_file = wave.open('loud_test.wav', 'w')
    wav_file.setnchannels(2)  # Stereo
    wav_file.setsampwidth(2)  # 16-bit
    wav_file.setframerate(sample_rate)
    
    # Write samples (duplicate for stereo)
    for sample in samples:
        wav_file.writeframes(struct.pack('<hh', sample, sample))
    
    wav_file.close()
    print("Created loud_test.wav")

def test_audio_directly():
    """Test pygame audio directly"""
    pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
    pygame.mixer.init()
    
    print(f"Pygame mixer initialized: {pygame.mixer.get_init()}")
    
    try:
        pygame.mixer.music.load('loud_test.wav')
        pygame.mixer.music.set_volume(1.0)  # Full volume
        pygame.mixer.music.play(-1)  # Loop
        
        print("Playing loud test audio...")
        print(f"Music busy: {pygame.mixer.music.get_busy()}")
        print("You should hear a loud 880Hz tone")
        print("Press Ctrl+C to stop")
        
        # Keep playing for 10 seconds
        import time
        time.sleep(10)
        
    except Exception as e:
        print(f"Audio error: {e}")
    finally:
        pygame.mixer.quit()

if __name__ == "__main__":
    create_loud_wav()
    test_audio_directly()
