import winsound
import time

print("Testing Windows system beep...")
print("You should hear several beeps")

# Test different frequencies
frequencies = [440, 880, 1760, 2000]  # A4, A5, A6, high pitch

for freq in frequencies:
    print(f"Playing {freq} Hz beep...")
    winsound.Beep(freq, 500)  # 500ms duration
    time.sleep(0.2)

print("Beep test complete")
print("If you didn't hear anything, check:")
print("1. System volume is up")
print("2. Speakers/headphones are connected")
print("3. Windows audio services are running")
