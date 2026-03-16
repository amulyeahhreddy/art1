import pygame
import sys

def check_audio_system():
    """Check the audio system and provide diagnostic info"""
    
    print("=== Audio System Diagnostic ===")
    print()
    
    # Check pygame version
    print(f"Pygame version: {pygame.version.ver}")
    print(f"SDL version: {pygame.version.SDL}")
    print()
    
    # Initialize audio
    pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
    pygame.mixer.init()
    
    print(f"Mixer initialized: {pygame.mixer.get_init()}")
    print()
    
    # Check available audio devices
    try:
        import pygame._sdl2.audio as sdl2_audio
        print("SDL2 audio available")
        devices = sdl2_audio.get_audio_device_names(False)  # False = output devices
        if devices:
            print("Available audio devices:")
            for i, device in enumerate(devices):
                print(f"  {i}: {device}")
        else:
            print("No audio devices found")
    except ImportError:
        print("SDL2 audio not available")
    except Exception as e:
        print(f"Error checking audio devices: {e}")
    
    print()
    
    # Test playing a sound
    try:
        print("Testing loud_test.wav...")
        pygame.mixer.music.load('loud_test.wav')
        pygame.mixer.music.set_volume(1.0)
        pygame.mixer.music.play()
        
        print(f"Music playing: {pygame.mixer.music.get_busy()}")
        print("You should hear a loud 880Hz tone for 1 second")
        
        # Wait for it to play
        import time
        time.sleep(2)
        
        print("Test complete")
        
    except Exception as e:
        print(f"Error playing test: {e}")
    
    pygame.mixer.quit()

if __name__ == "__main__":
    check_audio_system()
