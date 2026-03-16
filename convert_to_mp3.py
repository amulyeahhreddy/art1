#!/usr/bin/env python3
"""
Convert the existing .webm file to .mp3 using pydub
This bypasses the ffmpeg requirement for yt-dlp postprocessing
"""

from pydub import AudioSegment
import os

def convert_webm_to_mp3(input_file, output_file):
    """Convert webm to mp3 using pydub"""
    try:
        print(f"🔄 Converting {input_file} to {output_file}...")
        
        # Load the webm file
        audio = AudioSegment.from_file(input_file)
        
        # Export as mp3
        audio.export(output_file, format="mp3")
        
        print(f"✅ Successfully converted to {output_file}")
        print(f"📁 File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting file: {e}")
        return False

if __name__ == "__main__":
    # Convert the downloaded webm file to mp3
    webm_file = "carnatic.webm"
    mp3_file = "carnatic.mp3"
    
    if os.path.exists(webm_file):
        success = convert_webm_to_mp3(webm_file, mp3_file)
        if success:
            print(f"🎵 Ready to use: {mp3_file}")
        else:
            print("❌ Conversion failed")
    else:
        print(f"❌ Input file not found: {webm_file}")
