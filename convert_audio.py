from pydub import AudioSegment
import os

# Load the webm file
webm_file = "Carnatic Music ｜ Jayanthi Kumaresh ｜ Raga Kapi - Thillana (Pt. 2) ｜ Music of India [4yv4ea1pFp4].webm"

try:
    print("Loading webm file...")
    audio = AudioSegment.from_file(webm_file, format="webm")
    
    # Export as wav
    print("Converting to wav...")
    audio.export("carnatic_music.wav", format="wav")
    print("Successfully converted to carnatic_music.wav")
    
    # Also export as mp3 for backup
    print("Converting to mp3...")
    audio.export("carnatic_music.mp3", format="mp3")
    print("Successfully converted to carnatic_music.mp3")
    
except Exception as e:
    print(f"Error converting audio: {e}")
