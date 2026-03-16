# Audio Setup for Mudra Detection System

## Current Status
✅ Audio system is working with a simple test tone

## How to Use Your Own Music

### Option 1: Use the Downloaded Carnatic Music
The YouTube audio has been downloaded as:
```
Carnatic Music ｜ Jayanthi Kumaresh ｜ Raga Kapi - Thillana (Pt. 2) ｜ Music of India [4yv4ea1pFp4].webm
```

To use it:
1. Open `main.py`
2. Find line 81:
   ```python
   audio_manager.load_and_play("simple_background.wav", loop=True)
   ```
3. Replace it with:
   ```python
   audio_manager.load_and_play("Carnatic Music ｜ Jayanthi Kumaresh ｜ Raga Kapi - Thillana (Pt. 2) ｜ Music of India [4yv4ea1pFp4].webm", loop=True)
   ```

### Option 2: Convert to WAV/MP3 (Recommended)
For better compatibility, convert the webm file to WAV or MP3:
1. Use an online converter or audio software
2. Save as `carnatic_music.wav` or `carnatic_music.mp3`
3. Update main.py to use the new filename

### Option 3: Use Any Audio File
Place any audio file in the project folder and update the filename in main.py.

## Audio Controls

When running the application:

- **A** - Toggle audio play/pause
- **V** - Volume up (+10%)
- **B** - Volume down (-10%)  
- **U** - Mute/Unmute

## Supported Audio Formats
- ✅ WAV (recommended)
- ✅ MP3 (may require additional setup)
- ❌ WEBM (limited pygame support)

## Troubleshooting

### "Audio file not found" error
- Check that the audio file is in the same folder as main.py
- Verify the exact filename (including spaces and special characters)

### "ModPlug_Load failed" error
- This means pygame can't play the audio format
- Convert the file to WAV format for best compatibility

### No sound playing
- Check system volume
- Try pressing 'V' to increase volume in the app
- Verify audio works with other applications
